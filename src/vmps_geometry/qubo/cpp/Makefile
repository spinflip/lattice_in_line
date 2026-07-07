CXX ?= g++
CXXFLAGS ?= -O3 -std=gnu++17 -Wall -Wextra
EIGEN_CXXFLAGS := $(shell \
	if [ -d /opt/homebrew/include/eigen3 ]; then printf '%s' '-I/opt/homebrew/include/eigen3'; \
	elif [ -d /usr/local/include/eigen3 ]; then printf '%s' '-I/usr/local/include/eigen3'; \
	elif [ -d /usr/include/eigen3 ]; then printf '%s' '-I/usr/include/eigen3'; \
	fi)

.PHONY: all clean

all: bandwidth_qubo_tsq_cpp bandwidth_classical_compare_cpp

bandwidth_qubo_tsq_cpp: bandwidth_qubo_tsq.cpp libtsqubo/tsqubo.h
	$(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) bandwidth_qubo_tsq.cpp -o $@

bandwidth_classical_compare_cpp: bandwidth_classical_compare.cpp
	$(CXX) $(CXXFLAGS) bandwidth_classical_compare.cpp -o $@

clean:
	rm -f bandwidth_qubo_tsq_cpp bandwidth_classical_compare_cpp
