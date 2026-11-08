#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <span>
#include <thread>

#include <omp.h>

#include "geom/basis_table.hpp"
#include "geom/curve_table.hpp"
#include "geom/inverse_span_lengths.hpp"
#include "geom/point.hpp"
#include "geom/span_lookup.hpp"

// The OpenMP versions of the table fills, shared by the benchmarks. They call the same per-item functions as
// the kernels. Include only when compiling with OpenMP.

namespace bench
{
	// The CPUs a container's quota allows (docker --cpus), rounded down, at least 1; 0 if there is no quota.
	// cgroup v2 has "quota period" (or "max period") in cpu.max, cgroup v1 two separate files.
	inline int cgroupCpuLimit()
	{
		long quota{};
		long period{};
		std::ifstream v2("/sys/fs/cgroup/cpu.max");
		std::ifstream v1Quota("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
		std::ifstream v1Period("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
		bool const found = static_cast<bool>(v2 >> quota >> period) || static_cast<bool>((v1Quota >> quota) && (v1Period >> period));
		return (found && quota > 0 && period > 0) ? std::max(1, static_cast<int>(quota / period)) : 0;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Threads above a container's CPU quota get throttled for milliseconds, so they must not be measured:
	// on a 128-thread machine with a quota of 13.6 CPUs an empty OpenMP region could take over 10 ms.
	inline int usableThreads()
	{
		int const hardware = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
		int const limit = cgroupCpuLimit();
		return limit > 0 ? std::min(hardware, limit) : hardware;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void fillInverseSpanLengthsThreaded(int degree, std::span<T const> knots, std::span<T> inverseSpanLengths, int threads)
	{
		auto const p = static_cast<std::size_t>(degree);
#pragma omp parallel for num_threads(threads) schedule(static)
		for (std::size_t row = 0; row < inverseSpanLengths.size(); ++row)
		{
			inverseSpanLengths[row] = geom::detail::inverseSpanLength<T>(row + p, knots);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void fillBasisThreaded(int degree, std::span<T const> knots, std::span<T> coefficients, int threads)
	{
		std::size_t const numSpans = knots.size() - 2 * static_cast<std::size_t>(degree) - 1;
#pragma omp parallel for num_threads(threads) schedule(static)
		for (std::size_t row = 0; row < numSpans; ++row)
		{
			geom::detail::buildBasisRow<T>(row, degree, knots, coefficients);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void fillCurveThreaded(int degree, std::span<T const> basis, std::span<geom::Point4<T> const> points, std::span<geom::Point4<T>> coefficients, int threads)
	{
		auto const p = static_cast<std::size_t>(degree);
		std::size_t const numSpans = basis.size() / ((p + 1) * (p + 1));
#pragma omp parallel for num_threads(threads) schedule(static)
		for (std::size_t row = 0; row < numSpans; ++row)
		{
			for (std::size_t d = 0; d <= p; ++d)
			{
				geom::detail::buildCurveCoefficient<T>(row, d, degree, basis, points, coefficients);
			}
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One thread's chunk [first, last) of the cells: one findSpan for its first cell, then the CPU's sweep.
	template <std::floating_point T>
	void fillLookupChunk(int degree, std::span<T const> knots, geom::SpanLookup<T> const& lookup, std::span<std::uint32_t> cells, std::size_t first, std::size_t last)
	{
		std::size_t const n = knots.size() - static_cast<std::size_t>(degree) - 2;
		std::size_t span = geom::detail::firstSpanOfCell<T>(first, degree, knots, lookup.start(), lookup.cellsPerUnit());
		for (std::size_t cell = first; cell < last; ++cell)
		{
			T const leftEdge = geom::detail::cellLeftEdge<T>(cell, lookup.start(), lookup.cellsPerUnit());
			while (span < n && leftEdge >= knots[span + 1])
			{
				++span;
			}
			cells[cell] = static_cast<std::uint32_t>(span);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Splits the cells evenly over the threads; a thread with no cells does nothing.
	template <std::floating_point T>
	void fillLookupThreaded(int degree, std::span<T const> knots, geom::SpanLookup<T> const& lookup, std::span<std::uint32_t> cells, int threads)
	{
#pragma omp parallel num_threads(threads)
		{
			auto const thread = static_cast<std::size_t>(omp_get_thread_num());
			auto const team = static_cast<std::size_t>(omp_get_num_threads());
			std::size_t const first = cells.size() * thread / team;
			std::size_t const last = cells.size() * (thread + 1) / team;
			if (first < last)
			{
				fillLookupChunk<T>(degree, knots, lookup, cells, first, last);
			}
		}
	}
}  // namespace bench
