#include "gpu/span_lookup_gpu.hpp"

#include <span>
#include <stdexcept>

#include "geom/span_lookup.hpp"
#include "gpu/cuda_check.hpp"

namespace gpu
{
	namespace
	{
		// Threads per block, as in basis_table_gpu.cu.
		constexpr unsigned int kThreadsPerBlock = 256;
		
		// One thread per cell. The CPU sweep continues from the previous cell's span, which a thread cannot
		// see, so each thread does its own findSpan. The pointers are device memory.
		template <std::floating_point T>
		__global__ void spanLookupKernel(int degree, T const* knots, std::size_t numKnots, T start, T cellsPerUnit, std::size_t numCells, std::uint32_t* firstSpans)
		{
			std::size_t const cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
			if (cell >= numCells)
			{
				return;
			}
			firstSpans[cell] = geom::detail::firstSpanOfCell<T>(cell, degree, std::span<T const>(knots, numKnots), start, cellsPerUnit);
		}
	}  // namespace
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// numSpans = n + 1 - p with n = numKnots - p - 2.
	template <std::floating_point T>
	DeviceSpanLookup<T> allocateSpanLookupOnDevice(int degree, std::size_t numKnots, T start, T end, std::size_t cellsPerSpan)
	{
		std::size_t const numSpans = numKnots - 2 * static_cast<std::size_t>(degree) - 1;
		std::size_t const numCells = geom::detail::numLookupCells(numSpans, cellsPerSpan);
		return {degree, numKnots, start, geom::detail::lookupCellsPerUnit<T>(numCells, start, end), DeviceBuffer<std::uint32_t>(numCells)};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Enough blocks to give every cell a thread: ceil(numCells / kThreadsPerBlock).
	template <std::floating_point T>
	void fillSpanLookupOnDevice(DeviceBuffer<T> const& knots, DeviceSpanLookup<T>& lookup)
	{
		if (knots.size() != lookup.numKnots)
		{
			throw std::invalid_argument("fillSpanLookupOnDevice: table allocated for a different number of knots");
		}
		std::size_t const numCells = lookup.firstSpans.size();
		auto const blocks = static_cast<unsigned int>((numCells + kThreadsPerBlock - 1) / kThreadsPerBlock);
		spanLookupKernel<T><<<blocks, kThreadsPerBlock>>>(lookup.degree, knots.data(), knots.size(), lookup.start, lookup.cellsPerUnit, numCells, lookup.firstSpans.data());
		checkLaunch();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	DeviceSpanLookup<T> buildSpanLookupOnDevice(int degree, DeviceBuffer<T> const& knots, T start, T end, std::size_t cellsPerSpan)
	{
		DeviceSpanLookup<T> lookup = allocateSpanLookupOnDevice<T>(degree, knots.size(), start, end, cellsPerSpan);
		fillSpanLookupOnDevice<T>(knots, lookup);
		return lookup;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Explicit instantiations: the templates are defined only here, so these are the types callers can use.
	template DeviceSpanLookup<float>  allocateSpanLookupOnDevice<float>( int degree, std::size_t numKnots, float  start, float  end, std::size_t cellsPerSpan);
	template DeviceSpanLookup<double> allocateSpanLookupOnDevice<double>(int degree, std::size_t numKnots, double start, double end, std::size_t cellsPerSpan);
	template void fillSpanLookupOnDevice<float>( DeviceBuffer<float>  const& knots, DeviceSpanLookup<float>&  lookup);
	template void fillSpanLookupOnDevice<double>(DeviceBuffer<double> const& knots, DeviceSpanLookup<double>& lookup);
	template DeviceSpanLookup<float>  buildSpanLookupOnDevice<float>( int degree, DeviceBuffer<float>  const& knots, float  start, float  end, std::size_t cellsPerSpan);
	template DeviceSpanLookup<double> buildSpanLookupOnDevice<double>(int degree, DeviceBuffer<double> const& knots, double start, double end, std::size_t cellsPerSpan);
}  // namespace gpu
