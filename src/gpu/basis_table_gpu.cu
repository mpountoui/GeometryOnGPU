#include "gpu/basis_table_gpu.hpp"

#include <span>
#include <stdexcept>

#include "geom/basis_table.hpp"
#include "gpu/cuda_check.hpp"

namespace gpu
{
	namespace
	{
		// Threads per block: a multiple of the warp size (32); 256 is a common default.
		constexpr unsigned int kThreadsPerBlock = 256;
		
		// One thread per row of the table: thread `row` fills span row + p, exactly as one iteration of
		// the loop in the BasisTable constructor. The pointers are device memory.
		template <std::floating_point T>
		__global__ void basisTableKernel(int degree, T const* knots, std::size_t numKnots, std::size_t numSpans, T* coefficients, T* inverseSpanLengths)
		{
			std::size_t const row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
			if (row >= numSpans)
			{
				return;
			}
			auto const p = static_cast<std::size_t>(degree);
			std::size_t const tableSize = numSpans * (p + 1) * (p + 1);
			geom::detail::buildBasisRow<T>(row, degree, std::span<T const>(knots, numKnots), std::span<T>(coefficients, tableSize), std::span<T>(inverseSpanLengths, numSpans));
		}
	}  // namespace
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// numSpans = n + 1 - p with n = numKnots - p - 2.
	template <std::floating_point T>
	DeviceBasisTable<T> allocateBasisTableOnDevice(int degree, std::size_t numKnots)
	{
		auto const p = static_cast<std::size_t>(degree);
		std::size_t const numSpans = numKnots - 2 * p - 1;
		return {degree, numSpans, DeviceBuffer<T>(numSpans * (p + 1) * (p + 1)), DeviceBuffer<T>(numSpans)};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Enough blocks to give every row a thread: ceil(numSpans / kThreadsPerBlock).
	template <std::floating_point T>
	void fillBasisTableOnDevice(DeviceBuffer<T> const& knots, DeviceBasisTable<T>& table)
	{
		auto const p = static_cast<std::size_t>(table.degree);
		std::size_t const numSpans = table.numSpans;
		if (knots.size() != numSpans + 2 * p + 1 || table.coefficients.size() != numSpans * (p + 1) * (p + 1) || table.inverseSpanLengths.size() != numSpans)
		{
			throw std::invalid_argument("fillBasisTableOnDevice: table allocated for a different number of knots");
		}
		auto const blocks = static_cast<unsigned int>((numSpans + kThreadsPerBlock - 1) / kThreadsPerBlock);
		basisTableKernel<T><<<blocks, kThreadsPerBlock>>>(table.degree, knots.data(), knots.size(), numSpans, table.coefficients.data(), table.inverseSpanLengths.data());
		checkLaunch();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	DeviceBasisTable<T> buildBasisTableOnDevice(int degree, DeviceBuffer<T> const& knots)
	{
		DeviceBasisTable<T> table = allocateBasisTableOnDevice<T>(degree, knots.size());
		fillBasisTableOnDevice<T>(knots, table);
		return table;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Explicit instantiations: the templates are defined only here, so these are the types callers can use.
	template DeviceBasisTable<float>  allocateBasisTableOnDevice<float>( int degree, std::size_t numKnots);
	template DeviceBasisTable<double> allocateBasisTableOnDevice<double>(int degree, std::size_t numKnots);
	template void fillBasisTableOnDevice<float>( DeviceBuffer<float>  const& knots, DeviceBasisTable<float>&  table);
	template void fillBasisTableOnDevice<double>(DeviceBuffer<double> const& knots, DeviceBasisTable<double>& table);
	template DeviceBasisTable<float>  buildBasisTableOnDevice<float>( int degree, DeviceBuffer<float>  const& knots);
	template DeviceBasisTable<double> buildBasisTableOnDevice<double>(int degree, DeviceBuffer<double> const& knots);
}  // namespace gpu
