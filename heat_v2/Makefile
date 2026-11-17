CXX = mpicxx
# Add e.g. ARCH=-march=native for peak single-core numbers (keep it off if nodes differ).
ARCH ?=
CXXFLAGS ?= -O3 -std=c++17 -fopenmp -Wall -Wextra $(ARCH)

all: heat2d
heat2d: heat2d.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Correctness sweep: every mode x decomposition x rank count x awkward (non-divisible) sizes.
test: heat2d
	bash run_validation.sh

clean:
	rm -f heat2d
.PHONY: all test clean
