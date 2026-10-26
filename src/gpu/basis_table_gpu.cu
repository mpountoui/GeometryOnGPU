#include "gpu/basis_table_gpu.hpp"

#include <span>

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
	
	// Enough blocks to give every row a thread: ceil(numSpans / kThreadsPerBlock).
	template <std::floating_point T>
	DeviceBasisTable<T> buildBasisTableOnDevice(int degree, DeviceBuffer<T> const& knots)
	{
		auto const p = static_cast<std::size_t>(degree);
		std::size_t const numSpans = knots.size() - 2 * p - 1;
		DeviceBasisTable<T> table{degree, numSpans, DeviceBuffer<T>(numSpans * (p + 1) * (p + 1)), DeviceBuffer<T>(numSpans)};
		auto const blocks = static_cast<unsigned int>((numSpans + kThreadsPerBlock - 1) / kThreadsPerBlock);
		basisTableKernel<T><<<blocks, kThreadsPerBlock>>>(degree, knots.data(), knots.size(), numSpans, table.coefficients.data(), table.inverseSpanLengths.data());
		checkLaunch();
		return table;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Explicit instantiations: the template is defined only here, so these are the types callers can use.
	template DeviceBasisTable<float>  buildBasisTableOnDevice<float>( int degree, DeviceBuffer<float>  const& knots);
	template DeviceBasisTable<double> buildBasisTableOnDevice<double>(int degree, DeviceBuffer<double> const& knots);
}  // namespace gpu
