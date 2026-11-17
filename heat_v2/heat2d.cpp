// heat2d.cpp -- explicit 2D heat equation (5-point stencil), MPI + OpenMP.
//
// Parallelization variants
//   --mode   serial       single process, no MPI traffic (baseline)
//            blocking     MPI_Sendrecv halo exchange, then compute
//            nonblocking  MPI_Isend/Irecv, compute interior while halos are in flight,
//                         then compute boundary strips (communication/computation overlap)
//            hybrid       nonblocking + OpenMP threads inside each rank
//   --decomp slab         1D row decomposition  (px = P, py = 1)
//            cart         2D Cartesian blocks   (MPI_Dims_create); column halos use MPI_Type_vector
//   --px K                override rows of the process grid (py = P / K)
//   --init   gauss | sine (sine has a closed-form discrete solution -> independent validation)
//   --validate            gather result, compare with serial reference (and analytic if sine)
#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

struct Args {
  int n = 1024, steps = 200, px = 0;
  double r = 0.20;
  std::string mode = "blocking", decomp = "cart", init = "gauss";
  bool validate = false, affinity = false;
};

static Args parse(int argc, char** argv) {
  Args a;
  for (int k = 1; k < argc; ++k) {
    std::string s = argv[k];
    auto next = [&]() { if (k + 1 >= argc) throw std::runtime_error("Missing value after " + s); return std::string(argv[++k]); };
    if (s == "--n") a.n = std::stoi(next());
    else if (s == "--steps") a.steps = std::stoi(next());
    else if (s == "--r") a.r = std::stod(next());
    else if (s == "--mode") a.mode = next();
    else if (s == "--decomp") a.decomp = next();
    else if (s == "--init") a.init = next();
    else if (s == "--px") a.px = std::stoi(next());
    else if (s == "--validate") a.validate = true;
    else if (s == "--affinity") a.affinity = true;
    else throw std::runtime_error("Unknown argument: " + s);
  }
  if (a.n < 3 || a.steps < 0 || !(a.r > 0 && a.r <= 0.25)) throw std::runtime_error("Require n>=3, steps>=0, 0<r<=0.25");
  if (a.mode != "serial" && a.mode != "blocking" && a.mode != "nonblocking" && a.mode != "hybrid") throw std::runtime_error("Invalid --mode");
  if (a.decomp != "slab" && a.decomp != "cart") throw std::runtime_error("Invalid --decomp");
  if (a.init != "gauss" && a.init != "sine") throw std::runtime_error("Invalid --init");
  return a;
}

static inline void split(int n, int p, int c, int& cnt, int& off) {  // balanced block partition
  cnt = n / p + (c < n % p);
  off = c * (n / p) + std::min(c, n % p);
}

static inline double initial(int gi, int gj, int n, bool sine) {
  if (gi <= 0 || gj <= 0 || gi >= n - 1 || gj >= n - 1) return 0.0;  // Dirichlet boundary
  const double x = double(gi) / (n - 1), y = double(gj) / (n - 1);
  if (sine) return std::sin(M_PI * x) * std::sin(M_PI * y);
  return std::exp(-80.0 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)));
}

// v = u + r*Laplacian(u) on local rows [ilo,ihi], cols [jlo,jhi]; row stride ld.
static void update(const double* __restrict u, double* __restrict v, int ilo, int ihi, int jlo, int jhi,
                   int ld, double r, bool threaded) {
  if (ilo > ihi || jlo > jhi) return;
#pragma omp parallel for schedule(static) if (threaded)
  for (int i = ilo; i <= ihi; ++i) {
    const double* __restrict c = u + size_t(i) * ld;
    const double* __restrict nn = c - ld;
    const double* __restrict ss = c + ld;
    double* __restrict o = v + size_t(i) * ld;
    for (int j = jlo; j <= jhi; ++j) o[j] = c[j] + r * (nn[j] + ss[j] + c[j - 1] + c[j + 1] - 4 * c[j]);
  }
}

// Plain global serial reference (independent of the MPI data layout).
static std::vector<double> reference(const Args& a) {
  const int n = a.n; const bool sine = a.init == "sine";
  std::vector<double> u(size_t(n) * n), v(size_t(n) * n, 0.0);
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) u[size_t(i) * n + j] = initial(i, j, n, sine);
  for (int t = 0; t < a.steps; ++t) {
    for (int i = 1; i < n - 1; ++i) for (int j = 1; j < n - 1; ++j) {
      size_t p = size_t(i) * n + j;
      v[p] = u[p] + a.r * (u[p - n] + u[p + n] + u[p - 1] + u[p + 1] - 4 * u[p]);
    }
    u.swap(v);
  }
  return u;
}

int main(int argc, char** argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);  // only the master thread calls MPI
  int wrank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  try {
    Args a = parse(argc, argv);
    const int n = a.n;
    if (a.mode == "serial" && size != 1) throw std::runtime_error("serial mode requires -np 1");
    if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI lacks MPI_THREAD_FUNNELED");

    // ---- process grid -------------------------------------------------------------
    int dims[2] = {0, 0};
    if (a.decomp == "slab") { dims[0] = size; dims[1] = 1; }
    else if (a.px > 0) {
      if (size % a.px) throw std::runtime_error("--px must divide the number of ranks");
      dims[0] = a.px; dims[1] = size / a.px;
    } else MPI_Dims_create(size, 2, dims);
    if (dims[0] > n || dims[1] > n) throw std::runtime_error("process grid larger than problem");
    int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);
    int rank, co[2];
    MPI_Comm_rank(cart, &rank);
    MPI_Cart_coords(cart, rank, 2, co);
    int up, down, left, right;
    MPI_Cart_shift(cart, 0, 1, &up, &down);     // dim 0 = rows
    MPI_Cart_shift(cart, 1, 1, &left, &right);  // dim 1 = columns

    int nr, ro, nc, cof;  // local rows/cols and global offsets
    split(n, dims[0], co[0], nr, ro);
    split(n, dims[1], co[1], nc, cof);
    const int ld = nc + 2;  // 1-cell halo on every side
    const size_t cells = size_t(nr + 2) * ld;
    const bool threaded = a.mode == "hybrid";
    const bool overlap = a.mode == "nonblocking" || a.mode == "hybrid";
    const bool sine = a.init == "sine";

    // Raw arrays + first-touch initialisation with the same static schedule as the update loop (NUMA-friendly).
    std::unique_ptr<double[]> ubuf(new double[cells]), vbuf(new double[cells]);
    double* u = ubuf.get(); double* v = vbuf.get();
#pragma omp parallel for schedule(static) if (threaded)
    for (int i = 0; i < nr + 2; ++i)
      for (int j = 0; j < ld; ++j) {
        const bool inside = i >= 1 && i <= nr && j >= 1 && j <= nc;
        u[size_t(i) * ld + j] = inside ? initial(ro + i - 1, cof + j - 1, n, sine) : 0.0;
        v[size_t(i) * ld + j] = 0.0;
      }

    // Update ranges: global boundary rows/cols (index 0 and n-1) are fixed.
    const int ilo = ro == 0 ? 2 : 1, ihi = ro + nr == n ? nr - 1 : nr;
    const int jlo = cof == 0 ? 2 : 1, jhi = cof + nc == n ? nc - 1 : nc;

    MPI_Datatype coltype;  // one local column (strided)
    MPI_Type_vector(nr, 1, ld, MPI_DOUBLE, &coltype);
    MPI_Type_commit(&coltype);

    const long haloDoubles = (up != MPI_PROC_NULL ? nc : 0) + (down != MPI_PROC_NULL ? nc : 0) +
                             (left != MPI_PROC_NULL ? nr : 0) + (right != MPI_PROC_NULL ? nr : 0);
    const bool comm_on = a.mode != "serial";

    if (a.affinity) {
#ifdef __linux__
      char host[64]; gethostname(host, sizeof host);
#pragma omp parallel num_threads(threaded ? omp_get_max_threads() : 1)
#pragma omp critical
      std::cerr << "affinity rank=" << rank << " thread=" << omp_get_thread_num() << " cpu=" << sched_getcpu() << " host=" << host << "\n";
#endif
    }

    // ---- time stepping --------------------------------------------------------------
    double tcomm = 0.0, tcomp = 0.0;
    MPI_Barrier(cart);
    const double begin = MPI_Wtime();
    for (int step = 0; step < a.steps; ++step) {
      if (!comm_on) {
        double t = MPI_Wtime(); update(u, v, ilo, ihi, jlo, jhi, ld, a.r, false); tcomp += MPI_Wtime() - t;
      } else if (overlap) {
        MPI_Request req[8];
        double t = MPI_Wtime();
        // tags: 10 = data moving down, 11 = up, 12 = right, 13 = left
        MPI_Irecv(u + 1,                    nc, MPI_DOUBLE, up,    10, cart, &req[0]);
        MPI_Irecv(u + size_t(nr + 1) * ld + 1, nc, MPI_DOUBLE, down,  11, cart, &req[1]);
        MPI_Irecv(u + ld,                   1, coltype,    left,  12, cart, &req[2]);
        MPI_Irecv(u + ld + nc + 1,          1, coltype,    right, 13, cart, &req[3]);
        MPI_Isend(u + size_t(nr) * ld + 1,  nc, MPI_DOUBLE, down,  10, cart, &req[4]);
        MPI_Isend(u + ld + 1,               nc, MPI_DOUBLE, up,    11, cart, &req[5]);
        MPI_Isend(u + ld + nc,              1, coltype,    right, 12, cart, &req[6]);
        MPI_Isend(u + ld + 1,               1, coltype,    left,  13, cart, &req[7]);
        tcomm += MPI_Wtime() - t;
        // interior cells (rows 2..nr-1, cols 2..nc-1) need no remote data
        const int ii0 = std::max(ilo, 2), ii1 = std::min(ihi, nr - 1);
        const int jj0 = std::max(jlo, 2), jj1 = std::min(jhi, nc - 1);
        t = MPI_Wtime(); update(u, v, ii0, ii1, jj0, jj1, ld, a.r, threaded); tcomp += MPI_Wtime() - t;
        t = MPI_Wtime(); MPI_Waitall(8, req, MPI_STATUSES_IGNORE); tcomm += MPI_Wtime() - t;
        t = MPI_Wtime();  // boundary strips of the local block
        if (ilo <= 1 && ihi >= 1) update(u, v, 1, 1, jlo, jhi, ld, a.r, threaded);
        if (nr > 1 && ilo <= nr && ihi >= nr) update(u, v, nr, nr, jlo, jhi, ld, a.r, threaded);
        if (jlo <= 1 && jhi >= 1) update(u, v, ii0, ii1, 1, 1, ld, a.r, threaded);
        if (nc > 1 && jlo <= nc && jhi >= nc) update(u, v, ii0, ii1, nc, nc, ld, a.r, threaded);
        tcomp += MPI_Wtime() - t;
      } else {  // blocking
        double t = MPI_Wtime();
        MPI_Sendrecv(u + ld + 1,              nc, MPI_DOUBLE, up,    11, u + size_t(nr + 1) * ld + 1, nc, MPI_DOUBLE, down,  11, cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(u + size_t(nr) * ld + 1, nc, MPI_DOUBLE, down,  10, u + 1,                       nc, MPI_DOUBLE, up,    10, cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(u + ld + nc,             1, coltype,    right, 12, u + ld,                       1, coltype,    left,  12, cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(u + ld + 1,              1, coltype,    left,  13, u + ld + nc + 1,              1, coltype,    right, 13, cart, MPI_STATUS_IGNORE);
        tcomm += MPI_Wtime() - t;
        t = MPI_Wtime(); update(u, v, ilo, ihi, jlo, jhi, ld, a.r, false); tcomp += MPI_Wtime() - t;
      }
      std::swap(u, v);
    }
    const double elapsed = MPI_Wtime() - begin;

    double mx[3] = {elapsed, tcomm, tcomp}, gmx[3] = {0, 0, 0};
    MPI_Reduce(mx, gmx, 3, MPI_DOUBLE, MPI_MAX, 0, cart);
    double localSum = 0;
    for (int i = 1; i <= nr; ++i) for (int j = 1; j <= nc; ++j) localSum += u[size_t(i) * ld + j];
    double sum = 0; MPI_Reduce(&localSum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
    long hmax = 0; MPI_Reduce(&haloDoubles, &hmax, 1, MPI_LONG, MPI_MAX, 0, cart);

    // ---- validation ------------------------------------------------------------------
    double linf = -1, relL2 = -1, analytic = -1; int ok = 1;
    if (a.validate) {
      std::vector<double> pack(size_t(nr) * nc);
      for (int i = 0; i < nr; ++i) for (int j = 0; j < nc; ++j) pack[size_t(i) * nc + j] = u[size_t(i + 1) * ld + j + 1];
      std::vector<int> counts, displs; std::vector<double> all;
      if (rank == 0) {
        counts.resize(size); displs.resize(size); all.resize(size_t(n) * n);
        int d = 0;
        for (int q = 0; q < size; ++q) {
          int c[2], r0, o0, c0, o1; MPI_Cart_coords(cart, q, 2, c);
          split(n, dims[0], c[0], r0, o0); split(n, dims[1], c[1], c0, o1);
          counts[q] = r0 * c0; displs[q] = d; d += counts[q];
        }
      }
      MPI_Gatherv(pack.data(), nr * nc, MPI_DOUBLE, rank == 0 ? all.data() : nullptr,
                  rank == 0 ? counts.data() : nullptr, rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, cart);
      if (sine) {  // analytic discrete solution: u = lambda^steps * u0, lambda = 1 - 8 r sin^2(pi h / 2)
        const double h = 1.0 / (n - 1), s = std::sin(M_PI * h / 2);
        const double lam = std::pow(1.0 - 8 * a.r * s * s, a.steps);
        double loc = 0;
        for (int i = 0; i < nr; ++i) for (int j = 0; j < nc; ++j)
          loc = std::max(loc, std::abs(pack[size_t(i) * nc + j] - lam * initial(ro + i, cof + j, n, true)));
        MPI_Allreduce(&loc, &analytic, 1, MPI_DOUBLE, MPI_MAX, cart);
      }
      if (rank == 0) {
        auto ref = reference(a);
        double err2 = 0, ref2 = 0; linf = 0;
        for (int q = 0; q < size; ++q) {
          int c[2], r0, o0, c0, o1; MPI_Cart_coords(cart, q, 2, c);
          split(n, dims[0], c[0], r0, o0); split(n, dims[1], c[1], c0, o1);
          for (int i = 0; i < r0; ++i) for (int j = 0; j < c0; ++j) {
            double g = all[displs[q] + size_t(i) * c0 + j], f = ref[size_t(o0 + i) * n + o1 + j], e = g - f;
            linf = std::max(linf, std::abs(e)); err2 += e * e; ref2 += f * f;
          }
        }
        relL2 = std::sqrt(err2) / std::max(std::sqrt(ref2), 1e-300);
        ok = (linf < 1e-10 && relL2 < 1e-10 && (!sine || analytic < 1e-10)) ? 1 : 0;
      }
    }

    if (rank == 0) {
      const double updates = double(n - 2) * (n - 2) * a.steps;
      const double mcups = gmx[0] > 0 ? updates / gmx[0] / 1e6 : 0;
      std::cout << std::setprecision(12)
                << "mode,decomp,init,n,steps,r,ranks,px,py,threads,seconds,comm_max,compute_max,checksum,halo_doubles_max,mcups,gflops,linf,rel_l2,analytic_err,valid\n"
                << a.mode << ',' << (comm_on ? a.decomp : "none") << ',' << a.init << ',' << n << ',' << a.steps << ',' << a.r << ','
                << size << ',' << dims[0] << ',' << dims[1] << ',' << (threaded ? omp_get_max_threads() : 1) << ','
                << gmx[0] << ',' << gmx[1] << ',' << gmx[2] << ',' << sum << ',' << hmax << ',' << mcups << ',' << 7 * mcups / 1e3 << ','
                << linf << ',' << relL2 << ',' << analytic << ',' << (a.validate ? ok : -1) << '\n';
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, cart);
    MPI_Type_free(&coltype);
    MPI_Finalize();
    return ok ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "Rank " << wrank << ": " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
