#include <cmath>
#include <print>
#include <stdexcept>
#include <utility>
#include <vector>

#include "geom/nurbs_curve.hpp"

using geom::NurbsCurve;
using geom::Point3;
using geom::Point4;

namespace
{
	int failures = 0;
	
#define CHECK(cond) \
	do \
	{ \
		if (!(cond)) \
		{ \
			std::println(stderr, "{}:{}: CHECK failed: {}", __FILE__, __LINE__, #cond); \
			++failures; \
		} \
	} while (false)
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <typename F>
	bool throwsInvalidArgument(F&& f)
	{
		try
		{
			f();
		}
		catch (const std::invalid_argument&)
		{
			return true;
		}
		return false;
	}
	
	using Points = std::vector<Point3<double>>;
	using Values = std::vector<double>;
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// True if constructing the curve from this data throws std::invalid_argument.
	bool isRejected(int degree, const Values& knots, const Points& points, const Values& weights)
	{
		return throwsInvalidArgument([&] { (void)NurbsCurve<double>(degree, knots, points, weights); });
	}
	
	// The cubic curve of Figure 4.1 (The NURBS Book, p.119): U = {0,0,0,0,1/4,1/2,3/4,1,1,1,1},
	// weights {1,1,1,3,1,1,1}. The book does not list the point coordinates; these are made up.
	const Values kKnots{0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1};
	const Values kWeights{1, 1, 1, 3, 1, 1, 1};
	const Points kPoints{{0, 0, 0}, {1, 2, 0}, {3, 3, 0}, {4, 0, 0}, {6, 0, 0}, {7, 2, 0}, {9, 2, 0}};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testFigure41Sizes()
	{
		const NurbsCurve<double> c(3, kKnots, kPoints, kWeights);
		CHECK(c.degree() == 3);
		CHECK(c.n() == 6);
		CHECK(c.m() == 10);
		CHECK(c.m() == c.n() + 3 + 1);
		CHECK(c.numControlPoints() == 7);
		CHECK(c.domain() == std::pair(0.0, 1.0));
		CHECK(c.isRational());
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testFigure41Points()
	{
		const NurbsCurve<double> c(3, kKnots, kPoints, kWeights);
		CHECK(c.weight(3) == 3.0);
		CHECK(c.weightedControlPoints()[3] == (Point4<double>{12, 0, 0, 3}));
		CHECK(c.controlPoint(3) == (Point3<double>{4, 0, 0}));
		CHECK(c.controlPoint(1) == (Point3<double>{1, 2, 0}));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testIndexOutOfRange()
	{
		const NurbsCurve<double> c(3, kKnots, kPoints, kWeights);
		bool outOfRange = false;
		try
		{
			(void)c.controlPoint(7);
		}
		catch (const std::out_of_range&)
		{
			outOfRange = true;
		}
		CHECK(outOfRange);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testNonRational()
	{
		const NurbsCurve<double> c(3, kKnots, kPoints);
		CHECK(!c.isRational());
		CHECK(c.weight(0) == 1.0);
		CHECK(c.weightedControlPoints()[2] == (Point4<double>{3, 3, 0, 1}));
		
		// Equal weights other than 1 also give a non-rational curve (P4.7).
		CHECK(!NurbsCurve<double>(3, kKnots, kPoints, Values(7, 2.0)).isRational());
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testBezierAndFloat()
	{
		// No interior knots: a quadratic Bezier curve (P4.13).
		const std::vector<Point3<float>> pts{{0, 0, 0}, {1, 1, 0}, {2, 0, 0}};
		const NurbsCurve<float> c(2, {0, 0, 0, 1, 1, 1}, pts);
		CHECK(c.n() == 2);
		CHECK(c.m() == 5);
		CHECK(c.domain() == std::pair(0.0f, 1.0f));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testInvalidDegreeAndCounts()
	{
		const Points two{{0, 0, 0}, {1, 0, 0}};
		CHECK(isRejected(0, {0, 0, 1, 1}, two, {1, 1}));
		CHECK(isRejected(3, {0, 0, 0, 1, 1, 1}, two, {1, 1}));
		CHECK(isRejected(3, {0, 0, 0, 0, 0.5, 1, 1, 1, 1}, kPoints, kWeights));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testInvalidKnotOrderAndClamping()
	{
		CHECK(isRejected(3, {0, 0, 0, 0, 0.5, 0.25, 0.75, 1, 1, 1, 1}, kPoints, kWeights));
		CHECK(isRejected(3, {0, 0, 0, 0.1, 0.25, 0.5, 0.75, 1, 1, 1, 1}, kPoints, kWeights));
		CHECK(isRejected(3, {0, 0, 0, 0, 0, 0.5, 0.75, 1, 1, 1, 1}, kPoints, kWeights));
		const Points three{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
		CHECK(isRejected(2, {0, 0, 0, 0, 0, 0}, three, {1, 1, 1}));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testInteriorKnotMultiplicity()
	{
		// Repeated p + 1 = 4 times: rejected. Repeated p = 3 times: allowed (C^0 there).
		const Points eight(8, Point3<double>{});
		CHECK(isRejected(3, {0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1}, eight, Values(8, 1.0)));
		CHECK(!isRejected(3, {0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1}, kPoints, kWeights));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testInvalidWeights()
	{
		CHECK(isRejected(3, kKnots, kPoints, {1, 1, 1}));
		CHECK(isRejected(3, kKnots, kPoints, {1, 1, 1, 0, 1, 1, 1}));
		CHECK(isRejected(3, kKnots, kPoints, {1, 1, 1, -2, 1, 1, 1}));
		CHECK(isRejected(3, kKnots, kPoints, {1, 1, 1, std::nan(""), 1, 1, 1}));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void runAllTests()
	{
		testFigure41Sizes();
		testFigure41Points();
		testIndexOutOfRange();
		testNonRational();
		testBezierAndFloat();
		testInvalidDegreeAndCounts();
		testInvalidKnotOrderAndClamping();
		testInteriorKnotMultiplicity();
		testInvalidWeights();
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main()
{
	runAllTests();
	
	if (failures > 0)
	{
		std::println(stderr, "{} check(s) failed", failures);
		return 1;
	}
	std::println("All NurbsCurve tests passed");
	return 0;
}
