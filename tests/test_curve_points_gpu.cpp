#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "check.hpp"
#include "test_curves.hpp"
#include "geom/curve_table.hpp"
#include "geom/inverse_span_lengths.hpp"
#include "geom/nurbs_curve.hpp"
#include "geom/point.hpp"
#include "geom/span_lookup.hpp"
#include "gpu/basis_table_gpu.hpp"
#include "gpu/curve_points_gpu.hpp"
#include "gpu/curve_table_gpu.hpp"
#include "gpu/device_buffer.hpp"
#include "gpu/inverse_span_lengths_gpu.hpp"
#include "gpu/span_lookup_gpu.hpp"

using geom::NurbsCurve;
using geom::Point3;
using geom::Point4;
using gpu::DeviceBuffer;

namespace
{
	// Allowed difference in units of epsilon times the point's largest |coordinate| (at least 1). The tables
	// differ from the CPU's by FMA rounding (see the table tests), and the evaluation adds its own.
	constexpr int kUlps = 128;
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// What the GPU queries read, built on the GPU and kept there.
	template <std::floating_point T>
	struct DeviceCurve
	{
		DeviceBuffer<T> knots{};
		DeviceBuffer<T> inverseSpanLengths{};
		gpu::DeviceCurveTable<T> table{};
		gpu::DeviceSpanLookup<T> lookup{};
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	DeviceCurve<T> buildOnDevice(NurbsCurve<T> const& curve)
	{
		auto const [start, end] = curve.domain();
		DeviceBuffer<T> knots{curve.knots()};
		DeviceBuffer<Point4<T>> const points{curve.weightedControlPoints()};
		DeviceBuffer<T> inverseSpanLengths = gpu::buildInverseSpanLengthsOnDevice<T>(curve.degree(), knots);
		gpu::DeviceBasisTable<T> const basis = gpu::buildBasisTableOnDevice<T>(curve.degree(), knots);
		gpu::DeviceCurveTable<T> table = gpu::buildCurveTableOnDevice<T>(basis, points);
		gpu::DeviceSpanLookup<T> lookup = gpu::buildSpanLookupOnDevice<T>(curve.degree(), knots, start, end);
		table.coefficients.download();
		return {std::move(knots), std::move(inverseSpanLengths), std::move(table), std::move(lookup)};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One point evaluated on the GPU.
	template <std::floating_point T>
	Point3<T> evaluate(DeviceCurve<T> const& device, T u, DeviceBuffer<Point3<T>>& result)
	{
		gpu::evaluateCurvePointOnDevice<T>(device.knots, device.inverseSpanLengths, device.table, device.lookup, u, result);
		return result.download().front();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void checkPoint(Point3<T> const& cpu, Point3<T> const& gpu)
	{
		T const scale = std::max({T{1}, std::abs(cpu.x), std::abs(cpu.y), std::abs(cpu.z)});
		CHECK(test::nearPoint(gpu, cpu, static_cast<T>(kUlps) * std::numeric_limits<T>::epsilon() * scale));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Each point from the GPU equals the CPU's lookup query: at evenly spaced parameters, the domain's ends
	// and every knot (the spans' boundaries).
	template <std::floating_point T>
	void checkAgainstCpu(NurbsCurve<T> const& curve)
	{
		geom::CurveTable<T> const table(curve);
		geom::SpanLookup<T> const lookup(curve);
		std::vector<T> const invH = geom::buildInverseSpanLengths<T>(curve.degree(), curve.knots());
		DeviceCurve<T> const device = buildOnDevice<T>(curve);
		DeviceBuffer<Point3<T>> result(1);
		auto const [a, b] = curve.domain();
		std::vector<T> params(curve.knots().begin() + curve.degree(), curve.knots().end() - curve.degree());
		for (int k = 0; k <= 50; ++k)
		{
			params.push_back(a + (b - a) * static_cast<T>(k) / T{50});
		}
		for (T const u : params)
		{
			checkPoint<T>(geom::curvePointFromTable(curve, table, invH, lookup, u), evaluate<T>(device, u, result));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Low to high degree on uniform knots, then knots with an empty span and a p-fold knot.
	template <std::floating_point T>
	void testCurves()
	{
		for (int const p : {1, 2, 3, 5, 9})
		{
			checkAgainstCpu<T>(test::rationalCurve(p, test::uniformKnots<T>(p, 20)));
		}
		checkAgainstCpu<T>(test::rationalCurve<T>(2, {0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5}));
		checkAgainstCpu<T>(test::rationalCurve<T>(3, {0, 0, 0, 0, T(0.5), T(0.5), T(0.5), 1, 1, 1, 1}));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Arrays of two different curves are refused, and so is an empty result buffer.
	template <std::floating_point T>
	void testMismatchThrows()
	{
		DeviceCurve<T> const device = buildOnDevice<T>(test::rationalCurve(2, test::uniformKnots<T>(2, 6)));
		DeviceCurve<T> const other = buildOnDevice<T>(test::rationalCurve(2, test::uniformKnots<T>(2, 7)));
		DeviceBuffer<Point3<T>> result(1);
		DeviceBuffer<Point3<T>> empty{};
		bool mismatch = false;
		bool noResult = false;
		try
		{
			gpu::evaluateCurvePointOnDevice<T>(device.knots, device.inverseSpanLengths, other.table, device.lookup, T{0}, result);
		}
		catch (std::invalid_argument const&)
		{
			mismatch = true;
		}
		try
		{
			gpu::evaluateCurvePointOnDevice<T>(device.knots, device.inverseSpanLengths, device.table, device.lookup, T{0}, empty);
		}
		catch (std::invalid_argument const&)
		{
			noResult = true;
		}
		CHECK(mismatch);
		CHECK(noResult);
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main()
{
	testCurves<float>();
	testCurves<double>();
	testMismatchThrows<float>();
	testMismatchThrows<double>();
	return test::report("curve_points_gpu");
}
