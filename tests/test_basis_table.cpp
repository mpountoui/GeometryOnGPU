#include <algorithm>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>

#include "check.hpp"
#include "test_curves.hpp"
#include "geom/basis_polynomials.hpp"
#include "geom/basis_table.hpp"
#include "geom/inverse_span_lengths.hpp"
#include "geom/nurbs_curve.hpp"

using geom::BasisTable;
using geom::NurbsCurve;
using geom::Point3;

namespace
{
	using Points = std::vector<Point3<double>>;
	using Values = std::vector<double>;
	using test::nearPoint;
	using test::rationalCurve;
	
	constexpr double kTol = 1e-12;
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// curvePointFromTable equals curve(u) at 1001 parameters across the domain, both ends included.
	template <typename T>
	void checkAgainstCurve(NurbsCurve<T> const& curve, T tolerance)
	{
		BasisTable const table(curve);
		std::vector<T> const inverseSpanLengths = geom::buildInverseSpanLengths<T>(curve.degree(), curve.knots());
		auto const [a, b] = curve.domain();
		for (int k = 0; k <= 1000; ++k)
		{
			T const u = (k == 1000) ? b : a + (b - a) * static_cast<T>(k) / T{1000};
			CHECK(nearPoint(geom::curvePointFromTable(curve, table, inverseSpanLengths, u), curve(u), tolerance));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Ex2.3 knots (p = 2): 6 rows for j = 2..7; the empty span j = 6 (double knot at 4) stays 0.
	void testLayout()
	{
		Values const knots{0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5};
		BasisTable<double> const table(2, knots);
		CHECK(table.numSpans() == 6);
		CHECK(table.coefficients().size() == 6 * 9);
		for (double const c : table.spanCoefficients(6))
		{
			CHECK(c == 0.0);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Every non-empty row equals basisPolynomials for its span.
	void testRowsMatchBasisPolynomials()
	{
		Values const knots{0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5};
		BasisTable<double> const table(2, knots);
		std::vector<double> expected(9);
		for (std::size_t span = 2; span <= 7; ++span)
		{
			if (knots[span] < knots[span + 1])
			{
				geom::basisPolynomials<double>(span, 2, knots, expected);
				std::span<double const> const row = table.spanCoefficients(span);
				CHECK(std::vector<double>(row.begin(), row.end()) == expected);
			}
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Ex4.1 (The NURBS Book, pp.122-124): C(1) = (7/5, 6/5); and a quarter circle stays on radius 1.
	void testBookExamples()
	{
		Points const points{{0, 0, 0}, {1, 1, 0}, {3, 2, 0}, {4, 1, 0}, {5, -1, 0}};
		NurbsCurve<double> const ex41(2, {0, 0, 0, 1, 2, 3, 3, 3}, points, Values{1, 4, 1, 1, 1});
		CHECK(nearPoint(geom::curvePointFromTable(ex41, BasisTable(ex41), geom::buildInverseSpanLengths<double>(ex41.degree(), ex41.knots()), 1.0), Point3<double>{7.0 / 5, 6.0 / 5, 0}, kTol));
		Points const quarter{{1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
		NurbsCurve<double> const circle(2, {0, 0, 0, 1, 1, 1}, quarter, Values{1, std::numbers::sqrt2 / 2, 1});
		BasisTable const table(circle);
		Values const inverseSpanLengths = geom::buildInverseSpanLengths<double>(circle.degree(), circle.knots());
		for (int k = 0; k <= 100; ++k)
		{
			Point3<double> const c = geom::curvePointFromTable(circle, table, inverseSpanLengths, k / 100.0);
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
	
	// fillBasisTable into memory full of NaN writes every value, as the constructor (which calls it) has
	// them, also the zero row of the empty span at the double knot 4.
	void testFillIntoExistingMemory()
	{
		NurbsCurve<double> const curve = rationalCurve<double>(2, {0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5});
		BasisTable const table(curve);
		std::vector<double> coefficients(table.coefficients().size(), std::numeric_limits<double>::quiet_NaN());
		geom::fillBasisTable<double>(curve.degree(), curve.knots(), coefficients);
		CHECK(std::ranges::equal(coefficients, table.coefficients()));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// float, p = 3: within float rounding of curve(u) in float.
	void testFloat()
	{
		std::vector<float> const knots{0, 0, 0, 0, 0.25f, 0.5f, 0.75f, 1, 1, 1, 1};
		checkAgainstCurve(rationalCurve(3, knots), 1e-5f);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void runAllTests()
	{
		testLayout();
		testRowsMatchBasisPolynomials();
		testBookExamples();
		testMatchesCurvePoint();
		testFillIntoExistingMemory();
		testFloat();
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main()
{
	runAllTests();
	return test::report("basis_table");
}
