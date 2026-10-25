#include <cstddef>
#include <span>
#include <vector>

#include "check.hpp"
#include "geom/basis.hpp"
#include "geom/basis_polynomials.hpp"

using geom::basisFunctions;
using geom::basisPolynomials;
using geom::evaluatePolynomial;
using geom::localParameter;

namespace
{
	// Ex2.3 (The NURBS Book, p.68): p = 2, a double knot at 4 (an empty span and 0/0 factors).
	std::vector<double> const kEx23Knots{0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5};
	
	// Figure 4.1 (p.119): p = 3, uniform interior knots.
	std::vector<double> const kFig41Knots{0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1};
	
	// p = 3 with an interior knot of multiplicity p (the curve is only C^0 there).
	std::vector<double> const kTripleKnots{0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1};
	
	// p = 5, non-uniform.
	std::vector<double> const kDegree5Knots{0, 0, 0, 0, 0, 0, 0.1, 0.15, 0.4, 0.7, 0.71, 1, 1, 1, 1, 1, 1};
	
	// p = 9, the highest supported degree (kMaxDegree), with three interior knots.
	std::vector<double> const kDegree9Knots{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.5, 0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
	
	constexpr double kTol = 1e-12;
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The (degree + 1)^2 coefficients of the span, as basisPolynomials computes them.
	template <typename T>
	std::vector<T> tableOf(std::size_t span, int degree, std::vector<T> const& knots)
	{
		auto const stride = static_cast<std::size_t>(degree) + 1;
		std::vector<T> table(stride * stride);
		basisPolynomials<T>(span, degree, knots, table);
		return table;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Every function of the span's table, at u = u_span + h * s / 10 (s = 0..10, both ends
	// included), equals A2.2 at u.
	template <typename T>
	void checkSpan(std::size_t span, int degree, std::vector<T> const& knots, T tolerance)
	{
		auto const stride = static_cast<std::size_t>(degree) + 1;
		std::vector<T> const table = tableOf(span, degree, knots);
		std::vector<T> expected(stride);
		for (int s = 0; s <= 10; ++s)
		{
			T const u = knots[span] + (knots[span + 1] - knots[span]) * static_cast<T>(s) / T{10};
			basisFunctions(span, u, degree, knots, expected);
			T const t = localParameter(span, knots, u);
			for (std::size_t r = 0; r < stride; ++r)
			{
				T const value = evaluatePolynomial(std::span<T const>(table).subspan(r * stride, stride), t);
				CHECK(test::near(value, expected[r], tolerance));
			}
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// checkSpan on every non-empty span j = p..n.
	template <typename T>
	void checkAllSpans(int degree, std::vector<T> const& knots, T tolerance)
	{
		auto const p = static_cast<std::size_t>(degree);
		std::size_t const n = knots.size() - p - 2;
		for (std::size_t span = p; span <= n; ++span)
		{
			if (knots[span] < knots[span + 1])
			{
				checkSpan(span, degree, knots, tolerance);
			}
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// p = 1 on one span: N_{0,1} = 1 - t and N_{1,1} = t.
	void testLinearExact()
	{
		std::vector<double> const table = tableOf<double>(1, 1, {0, 0, 1, 1});
		CHECK(table == (std::vector<double>{1, -1, 0, 1}));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// p = 2 on one span is the Bernstein basis: (1 - t)^2, 2t(1 - t), t^2.
	void testBernsteinExact()
	{
		std::vector<double> const table = tableOf<double>(2, 2, {0, 0, 0, 1, 1, 1});
		CHECK(table == (std::vector<double>{1, -2, 1, 0, 2, -2, 0, 0, 1}));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testMatchesBasisFunctions()
	{
		checkAllSpans(1, std::vector<double>{0, 0, 0.3, 0.6, 1, 1}, kTol);
		checkAllSpans(2, kEx23Knots, kTol);
		checkAllSpans(3, kFig41Knots, kTol);
		checkAllSpans(3, kTripleKnots, kTol);
		checkAllSpans(5, kDegree5Knots, kTol);
		checkAllSpans(9, kDegree9Knots, kTol);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testFloat()
	{
		std::vector<float> const knots(kFig41Knots.begin(), kFig41Knots.end());
		checkAllSpans(3, knots, 1e-6f);
		std::vector<float> const degree9(kDegree9Knots.begin(), kDegree9Knots.end());
		checkAllSpans(9, degree9, 1e-5f);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Knots far from 0 (Fig. 4.1 shifted by 1000) in float: the table uses only knot differences,
	// so its values at exact t stay as accurate as near 0. Reference: A2.2 in double.
	void testFloatFarFromZero()
	{
		std::vector<double> shifted = kFig41Knots;
		for (double& knot : shifted)
		{
			knot += 1000.0;
		}
		std::vector<float> const knots(shifted.begin(), shifted.end());
		std::vector<float> const table = tableOf<float>(5, 3, knots);
		std::vector<double> expected(4);
		basisFunctions(5, 1000.625, 3, shifted, expected);    // t = 1/2 in [1000.5, 1000.75)
		for (std::size_t r = 0; r < 4; ++r)
		{
			float const value = evaluatePolynomial(std::span<float const>(table).subspan(r * 4, 4), 0.5f);
			CHECK(test::near(static_cast<double>(value), expected[r], 1e-6));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void runAllTests()
	{
		testLinearExact();
		testBernsteinExact();
		testMatchesBasisFunctions();
		testFloat();
		testFloatFarFromZero();
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main()
{
	runAllTests();
	return test::report("basis_polynomials");
}
