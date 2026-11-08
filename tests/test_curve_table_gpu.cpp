#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "gpu_test_helpers.hpp"
#include "test_curves.hpp"
#include "geom/basis_table.hpp"
#include "geom/curve_table.hpp"
#include "geom/nurbs_curve.hpp"
#include "geom/point.hpp"
#include "gpu/basis_table_gpu.hpp"
#include "gpu/curve_table_gpu.hpp"
#include "gpu/device_buffer.hpp"

using geom::BasisTable;
using geom::CurveTable;
using geom::NurbsCurve;
using geom::Point3;
using geom::Point4;
using gpu::DeviceBuffer;

namespace
{
	// Allowed difference in units of epsilon times the row's scale (see rowScale). Not bit-identical:
	// the GPU basis table is already a few ulps off (FMA, see test_basis_table_gpu.cpp), and nvcc also
	// fuses the sum c * Pw + sum into FMA. Measured (2026-11-01, RTX A4000): at most 5.4 (float) and 5.9
	// (double); with --fmad=false the tables are bit-identical.
	constexpr int kUlps = 16;
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The size of the largest term c(r, d) * Pw in any sum of the row (at least 1): the sums cancel, so
	// their rounding errors scale with the terms, not with the result A_j[d].
	template <std::floating_point T>
	T rowScale(std::span<T const> basisRow, std::span<Point4<T> const> points)
	{
		T largestC{};
		for (T const c : basisRow)
		{
			largestC = std::max(largestC, std::abs(c));
		}
		T largestP{};
		for (Point4<T> const& pw : points)
		{
			largestP = std::max({largestP, std::abs(pw.x), std::abs(pw.y), std::abs(pw.z), std::abs(pw.w)});
		}
		return std::max(T{1}, largestC * largestP);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The x, y, z, w of each point in turn, to compare points as plain values.
	template <std::floating_point T>
	std::vector<T> components(std::span<Point4<T> const> points)
	{
		std::vector<T> values{};
		for (Point4<T> const& pw : points)
		{
			values.insert(values.end(), {pw.x, pw.y, pw.z, pw.w});
		}
		return values;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Every row of the GPU coefficients against the CPU CurveTable, built from basis and weightedPoints.
	template <std::floating_point T>
	void checkCoefficients(BasisTable<T> const& basis, std::span<Point4<T> const> weightedPoints, CurveTable<T> const& cpu, std::vector<Point4<T>> const& gpu)
	{
		auto const p = static_cast<std::size_t>(cpu.degree());
		CHECK(gpu.size() == cpu.coefficients().size());
		for (std::size_t row = 0; row < cpu.numSpans() && (row + 1) * (p + 1) <= gpu.size(); ++row)
		{
			T const scale = rowScale<T>(basis.spanCoefficients(row + p), weightedPoints.subspan(row, p + 1));
			std::vector<T> const cpuRow = components<T>(cpu.spanCoefficients(row + p));
			std::vector<T> const gpuRow = components<T>(std::span<Point4<T> const>(gpu).subspan(row * (p + 1), p + 1));
			test::checkWithin<T>(cpuRow, gpuRow, static_cast<T>(kUlps) * std::numeric_limits<T>::epsilon() * scale);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Downloads the GPU table and compares it with the CPU one.
	template <std::floating_point T>
	void checkTable(BasisTable<T> const& basis, std::span<Point4<T> const> weightedPoints, gpu::DeviceCurveTable<T> const& table)
	{
		CurveTable<T> const cpu(basis, weightedPoints);
		CHECK(table.degree == cpu.degree());
		CHECK(table.numSpans == cpu.numSpans());
		checkCoefficients<T>(basis, weightedPoints, cpu, table.coefficients.download());
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Builds both tables on the GPU, each from dirty memory, and compares the curve table with the CPU's.
	template <std::floating_point T>
	void checkAgainstCpu(NurbsCurve<T> const& curve)
	{
		BasisTable<T> const basis(curve);
		DeviceBuffer<T> const knots{curve.knots()};
		DeviceBuffer<Point4<T>> const points{curve.weightedControlPoints()};
		test::dirtyDeviceMemory(basis.coefficients().size() * sizeof(T));
		gpu::DeviceBasisTable<T> const deviceBasis = gpu::buildBasisTableOnDevice<T>(curve.degree(), knots);
		test::dirtyDeviceMemory(basis.numSpans() * (static_cast<std::size_t>(curve.degree()) + 1) * sizeof(Point4<T>));
		checkTable<T>(basis, curve.weightedControlPoints(), gpu::buildCurveTableOnDevice<T>(deviceBasis, points));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Uniform knots with 20 interior knots, low to the highest degree. p = 2 and 5 give blocks whose width
	// p + 1 does not divide the warp size (3 x 85 and 6 x 42 threads).
	template <std::floating_point T>
	void testUniform()
	{
		for (int const p : {1, 2, 3, 5, 9})
		{
			checkAgainstCpu<T>(test::rationalCurve<T>(p, test::uniformKnots<T>(p, 20)));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Repeated interior knots: the kernel has no branch for empty spans, their zero basis rows give zeros.
	template <std::floating_point T>
	void testRepeatedKnots()
	{
		checkAgainstCpu<T>(test::rationalCurve<T>(2, {0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5}));
		checkAgainstCpu<T>(test::rationalCurve<T>(3, {0, 0, 0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4}));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// 100,001 spans: p = 3 is 1563 blocks of 64 rows, p = 9 is 4001 blocks of 25 rows; the last block is
	// only partly used.
	template <std::floating_point T>
	void testManySpans()
	{
		checkAgainstCpu<T>(test::rationalCurve<T>(3, test::uniformKnots<T>(3, 100'000)));
		checkAgainstCpu<T>(test::rationalCurve<T>(9, test::uniformKnots<T>(9, 100'000)));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The rationalCurve's points and weights in reverse order: another curve on the same knots.
	template <std::floating_point T>
	NurbsCurve<T> reversedCurve(int degree, std::vector<T> const& knots)
	{
		std::size_t const numPoints = knots.size() - static_cast<std::size_t>(degree) - 1;
		std::vector<Point3<T>> points = test::spiralPoints<T>(numPoints);
		std::vector<T> weights = test::wavyWeights<T>(numPoints);
		std::reverse(points.begin(), points.end());
		std::reverse(weights.begin(), weights.end());
		return NurbsCurve<T>(degree, knots, points, weights);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One device basis table serves two curves with the same knots, and is unchanged afterwards.
	template <std::floating_point T>
	void testTwoCurvesOneBasisTable()
	{
		std::vector<T> const knots = test::uniformKnots<T>(3, 20);
		NurbsCurve<T> const first = test::rationalCurve<T>(3, knots);
		NurbsCurve<T> const second = reversedCurve<T>(3, knots);
		DeviceBuffer<T> const deviceKnots{std::span<T const>(knots)};
		gpu::DeviceBasisTable<T> const deviceBasis = gpu::buildBasisTableOnDevice<T>(3, deviceKnots);
		std::vector<T> const basisBefore = deviceBasis.coefficients.download();
		DeviceBuffer<Point4<T>> const firstPoints{first.weightedControlPoints()};
		DeviceBuffer<Point4<T>> const secondPoints{second.weightedControlPoints()};
		checkTable<T>(BasisTable<T>(first), first.weightedControlPoints(), gpu::buildCurveTableOnDevice<T>(deviceBasis, firstPoints));
		checkTable<T>(BasisTable<T>(second), second.weightedControlPoints(), gpu::buildCurveTableOnDevice<T>(deviceBasis, secondPoints));
		CHECK(deviceBasis.coefficients.download() == basisBefore);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	bool fillThrows(gpu::DeviceBasisTable<T> const& basis, DeviceBuffer<Point4<T>> const& points, gpu::DeviceCurveTable<T>& table)
	{
		try
		{
			gpu::fillCurveTableOnDevice<T>(basis, points, table);
		}
		catch (std::invalid_argument const&)
		{
			return true;
		}
		return false;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One control point too few, and a curve table allocated for another degree.
	template <std::floating_point T>
	void testFillMismatchThrows()
	{
		NurbsCurve<T> const curve = test::rationalCurve<T>(2, test::uniformKnots<T>(2, 10));
		DeviceBuffer<T> const knots{curve.knots()};
		gpu::DeviceBasisTable<T> const basis = gpu::buildBasisTableOnDevice<T>(2, knots);
		DeviceBuffer<Point4<T>> const tooFew{curve.weightedControlPoints().first(curve.weightedControlPoints().size() - 1)};
		DeviceBuffer<Point4<T>> const points{curve.weightedControlPoints()};
		gpu::DeviceCurveTable<T> table = gpu::allocateCurveTableOnDevice<T>(2, basis.numSpans);
		gpu::DeviceCurveTable<T> otherDegree = gpu::allocateCurveTableOnDevice<T>(3, basis.numSpans);
		CHECK(fillThrows<T>(basis, tooFew, table));
		CHECK(fillThrows<T>(basis, points, otherDegree));
		CHECK(!fillThrows<T>(basis, points, table));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void runAllTests()
	{
		testUniform<T>();
		testRepeatedKnots<T>();
		testManySpans<T>();
		testTwoCurvesOneBasisTable<T>();
		testFillMismatchThrows<T>();
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main()
{
	runAllTests<float>();
	runAllTests<double>();
	return test::report("curve_table_gpu");
}
