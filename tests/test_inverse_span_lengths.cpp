#include <concepts>
#include <limits>
#include <span>
#include <vector>

#include "check.hpp"
#include "geom/inverse_span_lengths.hpp"

namespace
{
	// Span lengths 1, 2, 0 (repeated knot 3) and 4.
	template <std::floating_point T>
	void testKnownValues()
	{
		std::vector<T> const knots{0, 0, 0, 1, 3, 3, 7, 7, 7};
		CHECK(geom::buildInverseSpanLengths<T>(2, knots) == std::vector<T>({T{1}, T{0.5}, T{0}, T{0.25}}));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Every value is written, also the zero of an empty span, so NaN never survives.
	template <std::floating_point T>
	void testFillWritesEveryValue()
	{
		std::vector<T> const knots{0, 0, 0, 1, 3, 3, 7, 7, 7};
		std::vector<T> filled(4, std::numeric_limits<T>::quiet_NaN());
		geom::fillInverseSpanLengths<T>(2, knots, filled);
		CHECK(filled == geom::buildInverseSpanLengths<T>(2, knots));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void runAllTests()
	{
		testKnownValues<T>();
		testFillWritesEveryValue<T>();
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main()
{
	runAllTests<float>();
	runAllTests<double>();
	return test::report("inverse_span_lengths");
}
