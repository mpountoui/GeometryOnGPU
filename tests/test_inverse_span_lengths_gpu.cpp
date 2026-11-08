#include <concepts>
#include <span>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "gpu_test_helpers.hpp"
#include "test_curves.hpp"
#include "geom/inverse_span_lengths.hpp"
#include "gpu/device_buffer.hpp"
#include "gpu/inverse_span_lengths_gpu.hpp"

using gpu::DeviceBuffer;

namespace
{
	// Builds the array on the GPU from dirty memory. Each value is one subtraction and one correctly
	// rounded division on both sides, with nothing for FMA to fuse, so it must be identical to the CPU's.
	template <std::floating_point T>
	void checkAgainstCpu(int degree, std::vector<T> const& knots)
	{
		std::vector<T> const cpu = geom::buildInverseSpanLengths<T>(degree, knots);
		DeviceBuffer<T> const deviceKnots{std::span<T const>(knots)};
		test::dirtyDeviceMemory(cpu.size() * sizeof(T));
		CHECK(gpu::buildInverseSpanLengthsOnDevice<T>(degree, deviceKnots).download() == cpu);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void testUniform()
	{
		for (int const p : {1, 2, 3, 5, 9})
		{
			checkAgainstCpu<T>(p, test::uniformKnots<T>(p, 20));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Repeated interior knots: empty spans must come back as zeros.
	template <std::floating_point T>
	void testRepeatedKnots()
	{
		checkAgainstCpu<T>(2, {0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5});
		checkAgainstCpu<T>(3, {0, 0, 0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4});
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// 100,001 spans: 391 blocks of 256 threads, the last one only partly used.
	template <std::floating_point T>
	void testManySpans()
	{
		checkAgainstCpu<T>(3, test::uniformKnots<T>(3, 100'000));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One buffer filled twice from different knots of the same length (span 3 is empty only the second time).
	template <std::floating_point T>
	void testRefill()
	{
		std::vector<T> const first{0, 0, 0, 1, 2, 3, 4, 5, 5, 5};
		std::vector<T> const second{0, 0, 0, 1, 1, 3, 4, 5, 5, 5};
		DeviceBuffer<T> knots{std::span<T const>(first)};
		DeviceBuffer<T> inverseSpanLengths(first.size() - 5);
		gpu::fillInverseSpanLengthsOnDevice<T>(2, knots, inverseSpanLengths);
		CHECK(inverseSpanLengths.download() == geom::buildInverseSpanLengths<T>(2, first));
		knots.upload(second);
		gpu::fillInverseSpanLengthsOnDevice<T>(2, knots, inverseSpanLengths);
		CHECK(inverseSpanLengths.download() == geom::buildInverseSpanLengths<T>(2, second));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A buffer sized for 10 knots cannot be filled from 11.
	template <std::floating_point T>
	void testFillSizeMismatchThrows()
	{
		std::vector<T> const knots{0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6};
		DeviceBuffer<T> const deviceKnots{std::span<T const>(knots)};
		DeviceBuffer<T> inverseSpanLengths(knots.size() - 6);
		bool thrown = false;
		try
		{
			gpu::fillInverseSpanLengthsOnDevice<T>(2, deviceKnots, inverseSpanLengths);
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
		testUniform<T>();
		testRepeatedKnots<T>();
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
	return test::report("inverse_span_lengths_gpu");
}
