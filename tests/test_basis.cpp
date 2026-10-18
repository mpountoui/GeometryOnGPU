#include <array>
#include <cstddef>
#include <vector>

#include "check.hpp"
#include "geom/basis.hpp"
#include "geom/nurbs_curve.hpp"

using geom::basisFunctions;
using geom::findSpan;

namespace
{
	using Values = std::vector<double>;
	
	// Ex2.3 (The NURBS Book, p.68): p = 2, U = {0,0,0,1,2,3,4,4,5,5,5}, so n = 7.
	const Values kEx23Knots{0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5};
	
	// Figure 4.1 (p.119): p = 3, U = {0,0,0,0,1/4,1/2,3/4,1,1,1,1}, so n = 6.
	const Values kFig41Knots{0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1};
	
	constexpr double kTol = 1e-12;
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The p + 1 nonvanishing basis values at u, computed with A2.1 and A2.2.
	Values basisAt(int degree, const Values& knots, double u)
	{
		Values values(static_cast<std::size_t>(degree) + 1);
		basisFunctions(findSpan(degree, knots, u), u, degree, knots, values);
		return values;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testFindSpanEx23()
	{
		CHECK(findSpan(2, kEx23Knots, 2.5) == 4);
		CHECK(findSpan(2, kEx23Knots, 0.0) == 2);    // u = a: the first span, index p
		CHECK(findSpan(2, kEx23Knots, 1.0) == 3);    // u on a knot belongs to the span on its right
		CHECK(findSpan(2, kEx23Knots, 3.999) == 5);
		CHECK(findSpan(2, kEx23Knots, 4.0) == 7);    // double knot at 4: span [u_6, u_7) is empty
		CHECK(findSpan(2, kEx23Knots, 5.0) == 7);    // u = b: the special case, returns n
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testBasisFunctionsEx23()
	{
		const Values n = basisAt(2, kEx23Knots, 2.5);
		CHECK(n.size() == 3);
		CHECK(test::near(n[0], 1.0 / 8, kTol));
		CHECK(test::near(n[1], 6.0 / 8, kTol));
		CHECK(test::near(n[2], 1.0 / 8, kTol));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// P2.3 and P2.4: every N_{i,p}(u) >= 0 and they sum to 1, at 101 parameters across the domain.
	void checkPartitionOfUnity(int degree, const Values& knots)
	{
		const double a = knots.front();
		const double b = knots.back();
		for (int k = 0; k <= 100; ++k)
		{
			const double u = a + (b - a) * k / 100.0;
			double sum = 0.0;
			for (double value : basisAt(degree, knots, u))
			{
				CHECK(value >= -kTol);
				sum += value;
			}
			CHECK(test::near(sum, 1.0, kTol));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testPartitionOfUnity()
	{
		checkPartitionOfUnity(2, kEx23Knots);
		checkPartitionOfUnity(3, kFig41Knots);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// P2.8: at u = a only the first function is 1, at u = b only the last one.
	void testEndValues()
	{
		const Values start = basisAt(3, kFig41Knots, 0.0);
		const Values end = basisAt(3, kFig41Knots, 1.0);
		CHECK(test::near(start[0], 1.0, kTol) && test::near(start[3], 0.0, kTol));
		CHECK(test::near(start[1], 0.0, kTol) && test::near(start[2], 0.0, kTol));
		CHECK(test::near(end[3], 1.0, kTol) && test::near(end[0], 0.0, kTol));
		CHECK(test::near(end[1], 0.0, kTol) && test::near(end[2], 0.0, kTol));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testFloat()
	{
		const std::vector<float> knots{0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5};
		std::array<float, 3> values{};
		const std::size_t span = findSpan(2, knots, 2.5f);
		basisFunctions(span, 2.5f, 2, knots, values);
		CHECK(span == 4);
		CHECK(test::near(values[1], 0.75f, 1e-6f));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The functions take the knots straight from a NurbsCurve.
	void testWithNurbsCurve()
	{
		const std::vector<geom::Point3<double>> points(7);
		const geom::NurbsCurve<double> curve(3, kFig41Knots, points);
		CHECK(findSpan(curve.degree(), curve.knots(), 0.3) == 4);    // [u_4, u_5) = [1/4, 1/2)
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void runAllTests()
	{
		testFindSpanEx23();
		testBasisFunctionsEx23();
		testPartitionOfUnity();
		testEndValues();
		testFloat();
		testWithNurbsCurve();
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main()
{
	runAllTests();
	return test::report("basis");
}
