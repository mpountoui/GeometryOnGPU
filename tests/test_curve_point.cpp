#include <numbers>
#include <vector>

#include "check.hpp"
#include "geom/curve_point.hpp"

using geom::curvePoint;
using geom::NurbsCurve;
using geom::Point3;

namespace
{
	using Points = std::vector<Point3<double>>;
	using Values = std::vector<double>;
	
	constexpr double kTol = 1e-12;
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	bool nearPoint(const Point3<double>& a, const Point3<double>& b)
	{
		return test::near(a.x, b.x, kTol) && test::near(a.y, b.y, kTol) && test::near(a.z, b.z, kTol);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Ex4.1 (The NURBS Book, pp.122-124): C(1) = (7/5, 6/5).
	void testEx41()
	{
		const Points points{{0, 0, 0}, {1, 1, 0}, {3, 2, 0}, {4, 1, 0}, {5, -1, 0}};
		const NurbsCurve<double> curve(2, {0, 0, 0, 1, 2, 3, 3, 3}, points, Values{1, 4, 1, 1, 1});
		CHECK(nearPoint(curvePoint(curve, 1.0), {7.0 / 5, 6.0 / 5, 0}));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// P4.8: the curve starts at P_0 and ends at P_n (the rational cubic of Figure 4.1).
	void testEndPoints()
	{
		const Points points{{0, 0, 0}, {1, 2, 0}, {3, 3, 0}, {4, 0, 0}, {6, 0, 0}, {7, 2, 0}, {9, 2, 0}};
		const NurbsCurve<double> curve(3, {0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1}, points, Values{1, 1, 1, 3, 1, 1, 1});
		CHECK(nearPoint(curvePoint(curve, 0.0), points.front()));
		CHECK(nearPoint(curvePoint(curve, 1.0), points.back()));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Non-rational quadratic Bezier: C(1/2) = (P_0 + 2 P_1 + P_2) / 4 = (1, 1/2).
	void testNonRationalBezier()
	{
		const Points points{{0, 0, 0}, {1, 1, 0}, {2, 0, 0}};
		const NurbsCurve<double> curve(2, {0, 0, 0, 1, 1, 1}, points);
		CHECK(nearPoint(curvePoint(curve, 0.5), {1, 0.5, 0}));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A quarter circle as a rational quadratic Bezier (book Chapter 7): every point has radius 1.
	void testQuarterCircle()
	{
		const Points points{{1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
		const Values weights{1, std::numbers::sqrt2 / 2, 1};
		const NurbsCurve<double> curve(2, {0, 0, 0, 1, 1, 1}, points, weights);
		for (int k = 0; k <= 100; ++k)
		{
			const Point3<double> c = curvePoint(curve, k / 100.0);
			CHECK(test::near(c.x * c.x + c.y * c.y, 1.0, kTol));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// P4.7: multiplying every weight by the same factor does not move the curve.
	void testWeightScaling()
	{
		const Points points{{0, 0, 0}, {1, 1, 0}, {3, 2, 0}, {4, 1, 0}, {5, -1, 0}};
		const Values knots{0, 0, 0, 1, 2, 3, 3, 3};
		const NurbsCurve<double> a(2, knots, points, Values{1, 4, 1, 1, 1});
		const NurbsCurve<double> b(2, knots, points, Values{2.5, 10, 2.5, 2.5, 2.5});
		for (double u : {0.0, 0.7, 1.5, 2.2, 3.0})
		{
			CHECK(nearPoint(curvePoint(a, u), curvePoint(b, u)));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testFloat()
	{
		const std::vector<Point3<float>> points{{0, 0, 0}, {1, 1, 0}, {2, 0, 0}};
		const NurbsCurve<float> curve(2, {0, 0, 0, 1, 1, 1}, points);
		const Point3<float> c = curvePoint(curve, 0.5f);
		CHECK(test::near(c.x, 1.0f, 1e-6f) && test::near(c.y, 0.5f, 1e-6f));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void runAllTests()
	{
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
