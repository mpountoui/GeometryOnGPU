#include "gpu/curve_table_gpu.hpp"

#include <span>
#include <stdexcept>

#include "geom/curve_table.hpp"
#include "gpu/cuda_check.hpp"

namespace gpu
{
	namespace
	{
		// Threads per block, as in basis_table_gpu.cu. A block is (p + 1) x (256 / (p + 1)) threads.
		constexpr unsigned int kThreadsPerBlock = 256;

		// One thread per coefficient: threadIdx.x = d, threadIdx.y = row within the block, and the 1D grid
		// counts blocks of blockDim.y rows. Threads are numbered x-first, so neighbouring threads write
		// neighbouring A_j[d] (coalesced). The pointers are device memory.
		template <std::floating_point T>
		__global__ void curveTableKernel(int degree, std::size_t numSpans, T const* basisCoefficients, geom::Point4<T> const* weightedPoints, geom::Point4<T>* coefficients)
		{
			std::size_t const row = static_cast<std::size_t>(blockIdx.x) * blockDim.y + threadIdx.y;
			if (row >= numSpans)
			{
				return;
			}
			auto const p = static_cast<std::size_t>(degree);
			std::size_t const d = threadIdx.x;
			std::span<T const> const basis(basisCoefficients, numSpans * (p + 1) * (p + 1));
			std::span<geom::Point4<T> const> const points(weightedPoints, numSpans + p);
			geom::detail::buildCurveCoefficient<T>(row, d, degree, basis, points, std::span<geom::Point4<T>>(coefficients, numSpans * (p + 1)));
		}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

		template <std::floating_point T>
		void checkSizes(DeviceBasisTable<T> const& basis, DeviceBuffer<geom::Point4<T>> const& weightedPoints, DeviceCurveTable<T> const& table)
		{
			auto const p = static_cast<std::size_t>(table.degree);
			bool const sameCurve = basis.degree == table.degree && basis.numSpans == table.numSpans && weightedPoints.size() == table.numSpans + p;
			bool const allocated = table.coefficients.size() == table.numSpans * (p + 1) && table.inverseSpanLengths.size() == table.numSpans;
			if (!sameCurve || !allocated || basis.inverseSpanLengths.size() != table.numSpans)
			{
				throw std::invalid_argument("fillCurveTableOnDevice: basis table, control points and curve table do not match");
			}
		}
	}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	template <std::floating_point T>
	DeviceCurveTable<T> allocateCurveTableOnDevice(int degree, std::size_t numSpans)
	{
		auto const p = static_cast<std::size_t>(degree);
		return {degree, numSpans, DeviceBuffer<geom::Point4<T>>(numSpans * (p + 1)), DeviceBuffer<T>(numSpans)};
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	// Enough blocks to give every row a line of the block: ceil(numSpans / blockDim.y). The copy of 1 / h
	// goes into the same queue (stream) as the kernel, so it runs after it; device-to-device cudaMemcpy
	// does not make the CPU wait.
	template <std::floating_point T>
	void fillCurveTableOnDevice(DeviceBasisTable<T> const& basis, DeviceBuffer<geom::Point4<T>> const& weightedPoints, DeviceCurveTable<T>& table)
	{
		checkSizes<T>(basis, weightedPoints, table);
		auto const p = static_cast<unsigned int>(table.degree);
		dim3 const block(p + 1, kThreadsPerBlock / (p + 1));
		dim3 const grid(static_cast<unsigned int>((table.numSpans + block.y - 1) / block.y));
		curveTableKernel<T><<<grid, block>>>(table.degree, table.numSpans, basis.coefficients.data(), weightedPoints.data(), table.coefficients.data());
		checkLaunch();
		checkCuda(cudaMemcpy(table.inverseSpanLengths.data(), basis.inverseSpanLengths.data(), basis.inverseSpanLengths.sizeBytes(), cudaMemcpyDeviceToDevice));
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	template <std::floating_point T>
	DeviceCurveTable<T> buildCurveTableOnDevice(DeviceBasisTable<T> const& basis, DeviceBuffer<geom::Point4<T>> const& weightedPoints)
	{
		DeviceCurveTable<T> table = allocateCurveTableOnDevice<T>(basis.degree, basis.numSpans);
		fillCurveTableOnDevice<T>(basis, weightedPoints, table);
		return table;
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	// Explicit instantiations: the templates are defined only here, so these are the types callers can use.
	template DeviceCurveTable<float>  allocateCurveTableOnDevice<float>( int degree, std::size_t numSpans);
	template DeviceCurveTable<double> allocateCurveTableOnDevice<double>(int degree, std::size_t numSpans);
	template void fillCurveTableOnDevice<float>( DeviceBasisTable<float>  const& basis, DeviceBuffer<geom::Point4<float>>  const& weightedPoints, DeviceCurveTable<float>&  table);
	template void fillCurveTableOnDevice<double>(DeviceBasisTable<double> const& basis, DeviceBuffer<geom::Point4<double>> const& weightedPoints, DeviceCurveTable<double>& table);
	template DeviceCurveTable<float>  buildCurveTableOnDevice<float>( DeviceBasisTable<float>  const& basis, DeviceBuffer<geom::Point4<float>>  const& weightedPoints);
	template DeviceCurveTable<double> buildCurveTableOnDevice<double>(DeviceBasisTable<double> const& basis, DeviceBuffer<geom::Point4<double>> const& weightedPoints);
}  // namespace gpu
