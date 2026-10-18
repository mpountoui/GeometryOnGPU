#include <numbers>
#include <vector>

#include "check.hpp"
#include "geom/curve_point.hpp"
#include "geom/nurbs_curve.hpp"

using geom::NurbsCurve;
using geom::Point3;

namespace
{
	using Points = std::vector<Point3<double>>;
	using Values = std::vector<double>;
	
	constexpr double kTol = 1e-12;
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	bool nearPoint(Point3<double> const& a, Point3<double> const& b)
	{
		return test::near(a.x, b.x, kTol) && test::near(a.y, b.y, kTol) && test::near(a.z, b.z, kTol);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Ex4.1 (The NURBS Book, pp.122-124): C(1) = (7/5, 6/5).
	void testEx41()
	{
		Points const points{{0, 0, 0}, {1, 1, 0}, {3, 2, 0}, {4, 1, 0}, {5, -1, 0}};
		NurbsCurve<double> const curve(2, {0, 0, 0, 1, 2, 3, 3, 3}, points, Values{1, 4, 1, 1, 1});
		CHECK(nearPoint(curve(1.0), {7.0 / 5, 6.0 / 5, 0}));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// P4.8: the curve starts at P_0 and ends at P_n (the rational cubic of Figure 4.1).
	void testEndPoints()
	{
		Points const points{{0, 0, 0}, {1, 2, 0}, {3, 3, 0}, {4, 0, 0}, {6, 0, 0}, {7, 2, 0}, {9, 2, 0}};
		NurbsCurve<double> const curve(3, {0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1}, points, Values{1, 1, 1, 3, 1, 1, 1});
		CHECK(nearPoint(curve(0.0), points.front()));
		CHECK(nearPoint(curve(1.0), points.back()));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Non-rational quadratic Bezier: C(1/2) = (P_0 + 2 P_1 + P_2) / 4 = (1, 1/2).
	void testNonRationalBezier()
	{
		Points const points{{0, 0, 0}, {1, 1, 0}, {2, 0, 0}};
		NurbsCurve<double> const curve(2, {0, 0, 0, 1, 1, 1}, points);
		CHECK(nearPoint(curve(0.5), {1, 0.5, 0}));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A quarter circle as a rational quadratic Bezier (book Chapter 7): every point has radius 1.
	void testQuarterCircle()
	{
		Points const points{{1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
		Values const weights{1, std::numbers::sqrt2 / 2, 1};
		NurbsCurve<double> const curve(2, {0, 0, 0, 1, 1, 1}, points, weights);
		for (int k = 0; k <= 100; ++k)
		{
			Point3<double> const c = curve(k / 100.0);
			CHECK(test::near(c.x * c.x + c.y * c.y, 1.0, kTol));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// P4.7: multiplying every weight by the same factor does not move the curve.
	void testWeightScaling()
	{
		Points const points{{0, 0, 0}, {1, 1, 0}, {3, 2, 0}, {4, 1, 0}, {5, -1, 0}};
		Values const knots{0, 0, 0, 1, 2, 3, 3, 3};
		NurbsCurve<double> const a(2, knots, points, Values{1, 4, 1, 1, 1});
		NurbsCurve<double> const b(2, knots, points, Values{2.5, 10, 2.5, 2.5, 2.5});
		for (double u : {0.0, 0.7, 1.5, 2.2, 3.0})
		{
			CHECK(nearPoint(a(u), b(u)));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testFloat()
	{
		std::vector<Point3<float>> const points{{0, 0, 0}, {1, 1, 0}, {2, 0, 0}};
		NurbsCurve<float> const curve(2, {0, 0, 0, 1, 1, 1}, points);
		Point3<float> const c = curve(0.5f);
		CHECK(test::near(c.x, 1.0f, 1e-6f) && test::near(c.y, 0.5f, 1e-6f));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The low-level function, called with the curve's raw data, gives the same point as curve(u).
	void testLowLevelMatchesOperator()
	{
		Points const points{{0, 0, 0}, {1, 1, 0}, {3, 2, 0}, {4, 1, 0}, {5, -1, 0}};
		NurbsCurve<double> const curve(2, {0, 0, 0, 1, 2, 3, 3, 3}, points, Values{1, 4, 1, 1, 1});
		for (double u : {0.0, 0.5, 1.0, 2.5, 3.0})
		{
			CHECK(nearPoint(geom::curvePoint(curve.degree(), curve.knots(), curve.weightedControlPoints(), u), curve(u)));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void runAllTests()
	{
		testLowLevelMatchesOperator();
		testEx41();
		testEndPoints();
		testNonRationalBezier();
		testQuarterCircle();
		testWeightScaling();
		testFloat();
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main()
{
	runAllTests();
	return test::report("curvePoint");
}
