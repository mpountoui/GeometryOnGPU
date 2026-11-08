#include <concepts>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "gpu_test_helpers.hpp"
#include "test_curves.hpp"
#include "geom/span_lookup.hpp"
#include "gpu/device_buffer.hpp"
#include "gpu/span_lookup_gpu.hpp"

using geom::SpanLookup;
using gpu::DeviceBuffer;

namespace
{
	// The domain end b = u_{n+1} of a clamped knot vector.
	template <std::floating_point T>
	T domainEnd(int degree, std::vector<T> const& knots)
	{
		return knots[knots.size() - 1 - static_cast<std::size_t>(degree)];
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Identical to the CPU table, not just close: the cell edges are one division and one addition, with
	// nothing for FMA to fuse, and a span index has no rounding.
	template <std::floating_point T>
	void checkTable(SpanLookup<T> const& cpu, gpu::DeviceSpanLookup<T> const& lookup)
	{
		CHECK(lookup.degree == cpu.degree());
		CHECK(lookup.start == cpu.start());
		CHECK(lookup.cellsPerUnit == cpu.cellsPerUnit());
		std::span<std::uint32_t const> const cells = cpu.firstSpans();
		CHECK(lookup.firstSpans.download() == std::vector<std::uint32_t>(cells.begin(), cells.end()));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Builds the table on the GPU from dirty memory, with 1, 2 and 4 cells per span. 0xFFFFFFFF is no valid
	// span, so a cell the kernel forgets to write fails the comparison.
	template <std::floating_point T>
	void checkAgainstCpu(int degree, std::vector<T> const& knots)
	{
		DeviceBuffer<T> const deviceKnots{std::span<T const>(knots)};
		for (std::size_t const cellsPerSpan : {1, 2, 4})
		{
			SpanLookup<T> const cpu(degree, knots, cellsPerSpan);
			test::dirtyDeviceMemory(cpu.firstSpans().size() * sizeof(std::uint32_t));
			checkTable(cpu, gpu::buildSpanLookupOnDevice<T>(degree, deviceKnots, knots[static_cast<std::size_t>(degree)], domainEnd(degree, knots), cellsPerSpan));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// An empty span, a p-fold knot, a cluster of tiny spans in one cell, uniform knots, and a domain
	// that does not start at 0.
	template <std::floating_point T>
	void testSpecialKnots()
	{
		checkAgainstCpu<T>(2, {0, 0, 0, T(0.1), T(0.15), T(0.6), 1, 1, 1});
		checkAgainstCpu<T>(2, {0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5});
		checkAgainstCpu<T>(3, {0, 0, 0, 0, T(0.5), T(0.5), T(0.5), 1, 1, 1, 1});
		checkAgainstCpu<T>(3, {0, 0, 0, 0, T(0.5), T(0.5001), T(0.5002), T(0.5003), T(0.5004), 1, 1, 1, 1});
		checkAgainstCpu<T>(3, test::uniformKnots<T>(3, 500));
		checkAgainstCpu<T>(2, {1000, 1000, 1000, T(1000.3), T(1001.7), 1002, 1002, 1002});
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// 30 random knot vectors per degree 1..9, with repeated knots (fixed seed).
	template <std::floating_point T>
	void testRandomKnots()
	{
		std::mt19937 random(12345);
		for (int degree = 1; degree <= geom::kMaxDegree; ++degree)
		{
			for (int trial = 0; trial < 30; ++trial)
			{
				std::vector<double> const knots = test::randomKnots(degree, random);
				checkAgainstCpu<T>(degree, std::vector<T>(knots.begin(), knots.end()));
			}
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// 100,001 spans, so 400,004 cells with 4 per span: many blocks, the last one only partly used.
	template <std::floating_point T>
	void testManySpans()
	{
		checkAgainstCpu<T>(3, test::uniformKnots<T>(3, 100'000));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One allocated table filled twice from different knots of the same length and domain: the second
	// fill overwrites the first.
	template <std::floating_point T>
	void testRefill()
	{
		std::vector<T> const first{0, 0, 0, 1, 2, 3, 4, 5, 5, 5};
		std::vector<T> const second{0, 0, 0, 1, 1, 3, 4, 5, 5, 5};
		DeviceBuffer<T> knots{std::span<T const>(first)};
		gpu::DeviceSpanLookup<T> lookup = gpu::allocateSpanLookupOnDevice<T>(2, first.size(), T{0}, T{5});
		gpu::fillSpanLookupOnDevice<T>(knots, lookup);
		checkTable(SpanLookup<T>(2, first), lookup);
		knots.upload(second);
		gpu::fillSpanLookupOnDevice<T>(knots, lookup);
		checkTable(SpanLookup<T>(2, second), lookup);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A table allocated for 10 knots cannot be filled from 11.
	template <std::floating_point T>
	void testFillSizeMismatchThrows()
	{
		std::vector<T> const knots{0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6};
		DeviceBuffer<T> const deviceKnots{std::span<T const>(knots)};
		gpu::DeviceSpanLookup<T> lookup = gpu::allocateSpanLookupOnDevice<T>(2, knots.size() - 1, T{0}, T{6});
		bool thrown = false;
		try
		{
			gpu::fillSpanLookupOnDevice<T>(deviceKnots, lookup);
		}
		catch (std::invalid_argument const&)
		{
			thrown = true;
		}
		CHECK(thrown);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void runAllTests()
	{
		testSpecialKnots<T>();
		testRandomKnots<T>();
		testManySpans<T>();
		testRefill<T>();
		testFillSizeMismatchThrows<T>();
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main()
{
	runAllTests<float>();
	runAllTests<double>();
	return test::report("span_lookup_gpu");
}
