#include <cstddef>
#include <numbers>
#include <vector>

#include "check.hpp"
#include "test_curves.hpp"
#include "geom/basis_table.hpp"
#include "geom/curve_table.hpp"
#include "geom/nurbs_curve.hpp"

using geom::BasisTable;
using geom::CurveTable;
using geom::NurbsCurve;
using geom::Point3;
using geom::Point4;

namespace
{
	using Points = std::vector<Point3<double>>;
	using Values = std::vector<double>;
	using test::nearPoint;
	using test::rationalCurve;
	
	constexpr double kTol = 1e-12;
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The curve table gives curve(u), and the basis table's points, at 1001 parameters (ends included).
	template <typename T>
	void checkAgainstCurve(NurbsCurve<T> const& curve, T tolerance)
	{
		CurveTable const table(curve);
		BasisTable const basis(curve);
		auto const [a, b] = curve.domain();
		for (int k = 0; k <= 1000; ++k)
		{
			T const u = (k == 1000) ? b : a + (b - a) * static_cast<T>(k) / T{1000};
			Point3<T> const c = geom::curvePointFromTable(curve, table, u);
			CHECK(nearPoint(c, curve(u), tolerance));
			CHECK(nearPoint(c, geom::curvePointFromTable(curve, basis, u), tolerance));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Quadratic Bezier: Cw(t) = (1-t)^2 P0 + 2t(1-t) P1 + t^2 P2, so A_0 = P0, A_1 = 2 (P1 - P0),
	// A_2 = P0 - 2 P1 + P2 (in weighted points; here all weights are 1).
	void testBezierPowerForm()
	{
		Points const points{{0, 0, 0}, {1, 2, 0}, {3, 1, 0}};
		CurveTable const table(NurbsCurve<double>(2, {0, 0, 0, 1, 1, 1}, points));
		std::span<Point4<double> const> const a = table.spanCoefficients(2);
		CHECK(a.size() == 3);
		CHECK(a[0] == (Point4<double>{0, 0, 0, 1}));
		CHECK(a[1] == (Point4<double>{2, 4, 0, 0}));
		CHECK(a[2] == (Point4<double>{1, -3, 0, 0}));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Ex2.3 knots (p = 2): 6 rows of 3 points; the empty span j = 6 (double knot at 4) stays 0.
	void testLayout()
	{
		NurbsCurve<double> const curve = rationalCurve<double>(2, {0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5});
		CurveTable const table(curve);
		CHECK(table.degree() == 2);
		CHECK(table.numSpans() == 6);
		CHECK(table.coefficients().size() == 6 * 3);
		CHECK(table.inverseSpanLengths()[6 - 2] == 0.0);
		for (Point4<double> const& a : table.spanCoefficients(6))
		{
			CHECK(a == Point4<double>{});
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Ex4.1 (The NURBS Book, pp.122-124): C(1) = (7/5, 6/5); a quarter circle stays on radius 1.
	void testBookExamples()
	{
		Points const points{{0, 0, 0}, {1, 1, 0}, {3, 2, 0}, {4, 1, 0}, {5, -1, 0}};
		NurbsCurve<double> const ex41(2, {0, 0, 0, 1, 2, 3, 3, 3}, points, Values{1, 4, 1, 1, 1});
		CHECK(nearPoint(geom::curvePointFromTable(ex41, CurveTable(ex41), 1.0), Point3<double>{7.0 / 5, 6.0 / 5, 0}, kTol));
		Points const quarter{{1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
		NurbsCurve<double> const circle(2, {0, 0, 0, 1, 1, 1}, quarter, Values{1, std::numbers::sqrt2 / 2, 1});
		CurveTable const table(circle);
		for (int k = 0; k <= 100; ++k)
		{
			Point3<double> const c = geom::curvePointFromTable(circle, table, k / 100.0);
			CHECK(test::near(c.x * c.x + c.y * c.y, 1.0, kTol));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Rational curves for p = 1..9 on uniform knots, and knots with double and p-fold interior knots.
	void testMatchesCurvePoint()
	{
		for (int p = 1; p <= geom::kMaxDegree; ++p)
		{
			checkAgainstCurve(rationalCurve(p, test::uniformKnots<double>(p, 12)), 1e-10);
		}
		checkAgainstCurve(rationalCurve<double>(2, {0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5}), kTol);
		checkAgainstCurve(rationalCurve<double>(3, {0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1}), kTol);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Built from an existing BasisTable gives the same table as built from the curve.
	void testFromBasisTable()
	{
		NurbsCurve<double> const curve = rationalCurve(3, test::uniformKnots<double>(3, 5));
		CurveTable const direct(curve);
		CurveTable const fromBasis(BasisTable(curve), curve.weightedControlPoints());
		std::span<Point4<double> const> const a = direct.coefficients();
		std::span<Point4<double> const> const b = fromBasis.coefficients();
		CHECK(std::vector<Point4<double>>(a.begin(), a.end()) == std::vector<Point4<double>>(b.begin(), b.end()));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// float, p = 3: within float rounding of curve(u) in float.
	void testFloat()
	{
		checkAgainstCurve(rationalCurve(3, test::uniformKnots<float>(3, 3)), 1e-5f);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void runAllTests()
	{
		testBezierPowerForm();
		testLayout();
		testBookExamples();
		testMatchesCurvePoint();
		testFromBasisTable();
		testFloat();
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main()
{
	runAllTests();
	return test::report("curve_table");
}
