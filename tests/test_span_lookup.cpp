#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "check.hpp"
#include "test_curves.hpp"
#include "geom/basis.hpp"
#include "geom/curve_table.hpp"
#include "geom/nurbs_curve.hpp"
#include "geom/span_lookup.hpp"

using geom::findSpan;
using geom::SpanLookup;

namespace
{
	using Values = std::vector<double>;
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The query values for a knot vector, all inside the domain [a, b] (findSpan's precondition, asserted
	// in debug builds): every knot and its two floating-point neighbours, and count random values.
	template <typename T>
	std::vector<T> queryValues(std::vector<T> const& knots, std::mt19937& random, int count)
	{
		std::vector<T> values{};
		for (T const knot : knots)
		{
			values.insert(values.end(), {knot, std::nextafter(knot, -std::numeric_limits<T>::infinity()), std::nextafter(knot, std::numeric_limits<T>::infinity())});
		}
		std::erase_if(values, [&knots](T u) { return u < knots.front() || u > knots.back(); });
		std::uniform_real_distribution<double> uniform(static_cast<double>(knots.front()), static_cast<double>(knots.back()));
		for (int k = 0; k < count; ++k)
		{
			values.push_back(static_cast<T>(uniform(random)));
		}
		return values;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The lookup gives the same span as findSpan for every query value, with 1, 2 and 4 cells per span.
	// Outside the domain it clamps to p and n, as findSpan does in release builds.
	template <typename T>
	void checkMatchesFindSpan(int degree, std::vector<T> const& knots, std::mt19937& random)
	{
		std::vector<T> const values = queryValues(knots, random, 200);
		auto const p = static_cast<std::size_t>(degree);
		for (std::size_t const cellsPerSpan : {1, 2, 4})
		{
			SpanLookup<T> const lookup(degree, knots, cellsPerSpan);
			for (T const u : values)
			{
				CHECK(lookup.find(knots, u) == findSpan(degree, knots, u));
			}
			CHECK(lookup.find(knots, knots.front() - T{1}) == p);
			CHECK(lookup.find(knots, knots.back() + T{1}) == knots.size() - p - 2);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A random clamped knot vector of the degree on [0, 1]: up to 12 distinct interior knots, each
	// repeated 1..degree times.
	std::vector<double> randomKnots(int degree, std::mt19937& random)
	{
		std::uniform_int_distribution<int> distinct(0, 12);
		std::uniform_int_distribution<int> multiplicity(1, degree);
		std::uniform_real_distribution<double> position(0.0, 1.0);
		std::vector<double> interior{};
		int const count = distinct(random);
		for (int i = 0; i < count; ++i)
		{
			interior.insert(interior.end(), static_cast<std::size_t>(multiplicity(random)), position(random));
		}
		std::ranges::sort(interior);
		std::vector<double> knots(static_cast<std::size_t>(degree) + 1, 0.0);
		knots.insert(knots.end(), interior.begin(), interior.end());
		knots.insert(knots.end(), static_cast<std::size_t>(degree) + 1, 1.0);
		return knots;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The example of the doc: p = 2, knots {0,0,0, 0.1, 0.15, 0.6, 1,1,1}, one cell per span gives
	// 4 cells of width 0.25 with first spans {2, 4, 4, 5}.
	void testDocExample()
	{
		Values const knots{0, 0, 0, 0.1, 0.15, 0.6, 1, 1, 1};
		SpanLookup<double> const lookup(2, knots, 1);
		std::span<std::uint32_t const> const cells = lookup.firstSpans();
		CHECK(std::vector<std::uint32_t>(cells.begin(), cells.end()) == (std::vector<std::uint32_t>{2, 4, 4, 5}));
		CHECK(lookup.find(knots, 0.12) == 3);
		CHECK(lookup.find(knots, 0.55) == 4);
		CHECK(lookup.find(knots, 0.65) == 5);
		CHECK(lookup.find(knots, 1.0) == 5);
		CHECK(lookup.find(knots, 0.0) == 2);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Hand-picked knot vectors: an empty span, a p-fold knot, a cluster of tiny spans in one cell,
	// uniform knots, and knots that do not start at 0.
	void testSpecialKnots()
	{
		std::mt19937 random(1);
		checkMatchesFindSpan<double>(2, {0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5}, random);
		checkMatchesFindSpan<double>(3, {0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1}, random);
		checkMatchesFindSpan<double>(3, {0, 0, 0, 0, 0.5, 0.5001, 0.5002, 0.5003, 0.5004, 1, 1, 1, 1}, random);
		checkMatchesFindSpan(3, test::uniformKnots<double>(3, 500), random);
		checkMatchesFindSpan<double>(2, {1000, 1000, 1000, 1000.3, 1001.7, 1002, 1002, 1002}, random);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// 300 random knot vectors per degree 1..9, with repeated knots, in double and float.
	void testRandomKnots()
	{
		std::mt19937 random(12345);
		for (int degree = 1; degree <= geom::kMaxDegree; ++degree)
		{
			for (int trial = 0; trial < 300; ++trial)
			{
				std::vector<double> const knots = randomKnots(degree, random);
				checkMatchesFindSpan(degree, knots, random);
				checkMatchesFindSpan(degree, std::vector<float>(knots.begin(), knots.end()), random);
			}
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Curve points with the span from the lookup equal curve(u).
	void testCurvePoints()
	{
		geom::NurbsCurve<double> const curve = test::rationalCurve<double>(3, {0, 0, 0, 0, 0.1, 0.15, 0.6, 0.6, 0.9, 1, 1, 1, 1});
		geom::CurveTable const table(curve);
		SpanLookup const lookup(curve);
		for (int k = 0; k <= 1000; ++k)
		{
			double const u = k / 1000.0;
			CHECK(test::nearPoint(geom::curvePointFromTable(curve, table, lookup, u), curve(u), 1e-12));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void runAllTests()
	{
		testDocExample();
		testSpecialKnots();
		testRandomKnots();
		testCurvePoints();
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main()
{
	runAllTests();
	return test::report("span_lookup");
}
