#pragma once

#include <cstddef>

namespace bench
{
	// Times each table's build on the CPU and the GPU and prints the comparison (bench_table_builds.cpp).
	// Must run before any other CUDA call, so that the float basis table's first build shows the cost of
	// starting CUDA.
	void compareTableBuilds(int degree, std::size_t numKnots);
}  // namespace bench
