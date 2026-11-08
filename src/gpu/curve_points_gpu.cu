#include "gpu/curve_points_gpu.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "geom/curve_table.hpp"
#include "geom/span_lookup.hpp"
#include "gpu/cuda_check.hpp"

namespace gpu
{
	namespace
	{
		// One thread, one point: the same two GEOM_HD functions the CPU query calls. The pointers are device memory.
		template <std::floating_point T>
		__global__ void curvePointKernel(int degree, T const* knots, std::size_t numKnots, geom::Point4<T> const* coefficients, T const* inverseSpanLengths, std::uint32_t const* firstSpans, std::size_t numCells, T start, T cellsPerUnit, T u, geom::Point3<T>* point)
		{
			auto const p = static_cast<std::size_t>(degree);
			std::size_t const numSpans = numKnots - 2 * p - 1;
			std::span<T const> const knotSpan(knots, numKnots);
			std::size_t const span = geom::findSpanWithLookup<T>(degree, knotSpan, std::span<std::uint32_t const>(firstSpans, numCells), start, cellsPerUnit, u);
			*point = geom::curvePointFromCurveTable<T>(degree, knotSpan, std::span<geom::Point4<T> const>(coefficients, numSpans * (p + 1)), std::span<T const>(inverseSpanLengths, numSpans), span, u);
		}
		
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
		
		template <std::floating_point T>
		void checkSizes(DeviceBuffer<T> const& knots, DeviceBuffer<T> const& inverseSpanLengths, DeviceCurveTable<T> const& table, DeviceSpanLookup<T> const& lookup, DeviceBuffer<geom::Point3<T>> const& result)
		{
			std::size_t const numSpans = knots.size() - 2 * static_cast<std::size_t>(table.degree) - 1;
			bool const sameCurve = table.numSpans == numSpans && inverseSpanLengths.size() == numSpans && lookup.numKnots == knots.size() && lookup.degree == table.degree;
			if (!sameCurve || result.size() == 0)
			{
				throw std::invalid_argument("evaluateCurvePointOnDevice: the arrays do not belong to the same curve, or result is empty");
			}
		}
	}  // namespace
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void evaluateCurvePointOnDevice(DeviceBuffer<T> const& knots, DeviceBuffer<T> const& inverseSpanLengths, DeviceCurveTable<T> const& table, DeviceSpanLookup<T> const& lookup, T u, DeviceBuffer<geom::Point3<T>>& result)
	{
		checkSizes<T>(knots, inverseSpanLengths, table, lookup, result);
		curvePointKernel<T><<<1, 1>>>(table.degree, knots.data(), knots.size(), table.coefficients.data(), inverseSpanLengths.data(), lookup.firstSpans.data(), lookup.firstSpans.size(), lookup.start, lookup.cellsPerUnit, u, result.data());
		checkLaunch();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Explicit instantiations: the templates are defined only here, so these are the types callers can use.
	template void evaluateCurvePointOnDevice<float>( DeviceBuffer<float>  const& knots, DeviceBuffer<float>  const& inverseSpanLengths, DeviceCurveTable<float>  const& table, DeviceSpanLookup<float>  const& lookup, float  u, DeviceBuffer<geom::Point3<float>>&  result);
	template void evaluateCurvePointOnDevice<double>(DeviceBuffer<double> const& knots, DeviceBuffer<double> const& inverseSpanLengths, DeviceCurveTable<double> const& table, DeviceSpanLookup<double> const& lookup, double u, DeviceBuffer<geom::Point3<double>>& result);
}  // namespace gpu
