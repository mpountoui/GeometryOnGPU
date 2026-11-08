#include "gpu/inverse_span_lengths_gpu.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>

#include "geom/inverse_span_lengths.hpp"
#include "gpu/cuda_check.hpp"

namespace gpu
{
	namespace
	{
		// Threads per block, as in basis_table_gpu.cu.
		constexpr unsigned int kThreadsPerBlock = 256;
		
		// One thread per span: thread `row` computes the value of span row + p, exactly as one iteration
		// of the CPU loop. Neighbouring threads write neighbouring values. The pointers are device memory.
		template <std::floating_point T>
		__global__ void inverseSpanLengthsKernel(int degree, T const* knots, std::size_t numKnots, std::size_t numSpans, T* inverseSpanLengths)
		{
			std::size_t const row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
			if (row >= numSpans)
			{
				return;
			}
			inverseSpanLengths[row] = geom::detail::inverseSpanLength<T>(row + static_cast<std::size_t>(degree), std::span<T const>(knots, numKnots));
		}
	}  // namespace
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Enough blocks to give every span a thread: ceil(numSpans / kThreadsPerBlock).
	template <std::floating_point T>
	void fillInverseSpanLengthsOnDevice(int degree, DeviceBuffer<T> const& knots, DeviceBuffer<T>& inverseSpanLengths)
	{
		std::size_t const numSpans = inverseSpanLengths.size();
		if (knots.size() != numSpans + 2 * static_cast<std::size_t>(degree) + 1)
		{
			throw std::invalid_argument("fillInverseSpanLengthsOnDevice: buffer allocated for a different number of knots");
		}
		auto const blocks = static_cast<unsigned int>((numSpans + kThreadsPerBlock - 1) / kThreadsPerBlock);
		inverseSpanLengthsKernel<T><<<blocks, kThreadsPerBlock>>>(degree, knots.data(), knots.size(), numSpans, inverseSpanLengths.data());
		checkLaunch();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	DeviceBuffer<T> buildInverseSpanLengthsOnDevice(int degree, DeviceBuffer<T> const& knots)
	{
		DeviceBuffer<T> inverseSpanLengths(knots.size() - 2 * static_cast<std::size_t>(degree) - 1);
		fillInverseSpanLengthsOnDevice<T>(degree, knots, inverseSpanLengths);
		return inverseSpanLengths;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Explicit instantiations: the templates are defined only here, so these are the types callers can use.
	template void fillInverseSpanLengthsOnDevice<float>( int degree, DeviceBuffer<float>  const& knots, DeviceBuffer<float>&  inverseSpanLengths);
	template void fillInverseSpanLengthsOnDevice<double>(int degree, DeviceBuffer<double> const& knots, DeviceBuffer<double>& inverseSpanLengths);
	template DeviceBuffer<float>  buildInverseSpanLengthsOnDevice<float>( int degree, DeviceBuffer<float>  const& knots);
	template DeviceBuffer<double> buildInverseSpanLengthsOnDevice<double>(int degree, DeviceBuffer<double> const& knots);
}  // namespace gpu
