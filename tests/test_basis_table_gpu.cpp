#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "test_curves.hpp"
#include "geom/basis_table.hpp"
#include "gpu/basis_table_gpu.hpp"
#include "gpu/cuda_check.hpp"
#include "gpu/device_buffer.hpp"

using geom::BasisTable;
using gpu::DeviceBuffer;

namespace
{
	// Allowed difference in units of epsilon times the row's largest coefficient. The GPU result is
	// not bit-identical: nvcc fuses a * b + c into one FMA instruction (one rounding instead of two),
	// g++ on x86-64 does not by default, and the differences add up over the p levels. Measured
	// (2026-10-26, RTX 2000 Ada): at most 2; with --fmad=false the tables are bit-identical.
	constexpr int kUlps = 8;
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Sets every byte of a freed device block of this size to 0xFF (NaN as float and double). The next
	// cudaMalloc of the same size usually gets this block back, so a value the kernel forgets to write
	// shows up as NaN instead of a lucky 0.
	void dirtyDeviceMemory(std::size_t bytes)
	{
		DeviceBuffer<unsigned char> buffer(bytes);
		gpu::checkCuda(cudaMemset(buffer.data(), 0xFF, bytes));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Each GPU value within kUlps * epsilon * (largest |value| of the CPU row, at least 1).
	template <std::floating_point T>
	void checkRow(std::span<T const> cpu, std::span<T const> gpu)
	{
		T scale = T{1};
		for (T const c : cpu)
		{
			scale = std::max(scale, std::abs(c));
		}
		T const tolerance = static_cast<T>(kUlps) * std::numeric_limits<T>::epsilon() * scale;
		for (std::size_t i = 0; i < cpu.size(); ++i)
		{
			CHECK(test::near(gpu[i], cpu[i], tolerance));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void checkCoefficients(BasisTable<T> const& cpu, std::vector<T> const& gpu)
	{
		auto const p = static_cast<std::size_t>(cpu.degree());
		std::size_t const size = (p + 1) * (p + 1);
		CHECK(gpu.size() == cpu.coefficients().size());
		for (std::size_t row = 0; row < cpu.numSpans() && (row + 1) * size <= gpu.size(); ++row)
		{
			checkRow<T>(cpu.spanCoefficients(row + p), std::span<T const>(gpu).subspan(row * size, size));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Downloads the GPU table and compares it with the CPU one. 1 / h is one correctly rounded division
	// on both sides, so it must be identical.
	template <std::floating_point T>
	void checkTable(BasisTable<T> const& cpu, gpu::DeviceBasisTable<T> const& table)
	{
		CHECK(table.degree == cpu.degree());
		CHECK(table.numSpans == cpu.numSpans());
		CHECK(table.inverseSpanLengths.download() == std::vector<T>(cpu.inverseSpanLengths().begin(), cpu.inverseSpanLengths().end()));
		checkCoefficients(cpu, table.coefficients.download());
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Builds the table on the GPU from dirty memory and compares it with the CPU BasisTable.
	template <std::floating_point T>
	void checkAgainstCpu(int degree, std::vector<T> const& knots)
	{
		BasisTable<T> const cpu(degree, knots);
		DeviceBuffer<T> const deviceKnots{std::span<T const>(knots)};
		dirtyDeviceMemory(cpu.coefficients().size() * sizeof(T));
		checkTable(cpu, gpu::buildBasisTableOnDevice<T>(degree, deviceKnots));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Uniform knots with 20 interior knots, low to the highest degree.
	template <std::floating_point T>
	void testUniform()
	{
		for (int const p : {1, 2, 3, 5, 9})
		{
			checkAgainstCpu<T>(p, test::uniformKnots<T>(p, 20));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Repeated interior knots: empty spans must come back as zeros (0/0 = 0 factors elsewhere).
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
	
	// One allocated table filled twice from different knots of the same length: the second fill
	// overwrites every value of the first (span 3 is non-empty the first time, empty the second).
	template <std::floating_point T>
	void testRefill()
	{
		std::vector<T> const first{0, 0, 0, 1, 2, 3, 4, 5, 5, 5};
		std::vector<T> const second{0, 0, 0, 1, 1, 3, 4, 5, 5, 5};
		DeviceBuffer<T> knots{std::span<T const>(first)};
		gpu::DeviceBasisTable<T> table = gpu::allocateBasisTableOnDevice<T>(2, first.size());
		gpu::fillBasisTableOnDevice<T>(knots, table);
		checkTable(BasisTable<T>(2, first), table);
		knots.upload(second);
		gpu::fillBasisTableOnDevice<T>(knots, table);
		checkTable(BasisTable<T>(2, second), table);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A table allocated for 10 knots cannot be filled from 11.
	template <std::floating_point T>
	void testFillSizeMismatchThrows()
	{
		std::vector<T> const knots{0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6};
		DeviceBuffer<T> const deviceKnots{std::span<T const>(knots)};
		gpu::DeviceBasisTable<T> table = gpu::allocateBasisTableOnDevice<T>(2, knots.size() - 1);
		bool thrown = false;
		try
		{
			gpu::fillBasisTableOnDevice<T>(deviceKnots, table);
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
	return test::report("basis_table_gpu");
}
