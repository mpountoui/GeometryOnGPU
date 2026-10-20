#include <cmath>
#include <numbers>
#include <vector>

#include "check.hpp"
#include "geom/newton_raphson.hpp"

using geom::newtonRaphson;
using geom::NewtonOptions;
using geom::NewtonResult;
using geom::NewtonStatus;

namespace
{
	constexpr double kTol = 1e-12;
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// x^2 - 2 from 1: sqrt(2), with quadratic convergence (a handful of steps).
	void testSquareRootOfTwo()
	{
		NewtonResult<double> const r = newtonRaphson([](double x) { return x * x - 2.0; }, [](double x) { return 2.0 * x; }, 1.0);
		CHECK(r.status == NewtonStatus::Converged);
		CHECK(test::near(r.root, std::numbers::sqrt2, kTol));
		CHECK(r.iterations >= 1 && r.iterations <= 6);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Newton's own example x^3 - 2x - 5, and cos(x) = x.
	void testClassicExamples()
	{
		NewtonResult<double> const cubic = newtonRaphson([](double x) { return x * x * x - 2.0 * x - 5.0; }, [](double x) { return 3.0 * x * x - 2.0; }, 3.0);
		CHECK(cubic.status == NewtonStatus::Converged);
		CHECK(test::near(cubic.root, 2.0945514815423265, kTol));
		NewtonResult<double> const cosine = newtonRaphson([](double x) { return std::cos(x) - x; }, [](double x) { return -std::sin(x) - 1.0; }, 1.0);
		CHECK(cosine.status == NewtonStatus::Converged);
		CHECK(test::near(cosine.root, 0.7390851332151607, kTol));
		CHECK(test::near(cosine.value, 0.0, kTol));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A start value that already is the root takes no steps; a line takes exactly one.
	void testExactRootAndLine()
	{
		NewtonResult<double> const exact = newtonRaphson([](double x) { return x - 3.0; }, [](double) { return 1.0; }, 3.0);
		CHECK(exact.status == NewtonStatus::Converged && exact.iterations == 0 && exact.root == 3.0);
		NewtonResult<double> const line = newtonRaphson([](double x) { return 2.0 * x - 4.0; }, [](double) { return 2.0; }, 10.0);
		CHECK(line.status == NewtonStatus::Converged && line.iterations == 1 && line.root == 2.0);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// x^2 - 1 from 0: f'(0) = 0, the tangent is horizontal. x^2 + 1 has no real root.
	void testFailures()
	{
		NewtonResult<double> const flat = newtonRaphson([](double x) { return x * x - 1.0; }, [](double x) { return 2.0 * x; }, 0.0);
		CHECK(flat.status == NewtonStatus::ZeroDerivative);
		CHECK(flat.root == 0.0 && flat.iterations == 0);
		NewtonOptions<double> options{};
		options.maxIterations = 30;
		NewtonResult<double> const none = newtonRaphson([](double x) { return x * x + 1.0; }, [](double x) { return 2.0 * x; }, 0.5, options);
		CHECK(none.status == NewtonStatus::MaxIterations);
		CHECK(none.iterations == 30);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// NaN from the function: NotFinite instead of looping on garbage.
	void testNotFinite()
	{
		NewtonResult<double> const r = newtonRaphson([](double x) { return std::log(x); }, [](double x) { return 1.0 / x; }, -1.0);
		CHECK(r.status == NewtonStatus::NotFinite);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The callback sees every step, in order, starting at x0.
	void testStepCallback()
	{
		std::vector<double> xs{};
		auto const record = [&xs](int i, double x, double, double) { CHECK(i == static_cast<int>(xs.size())); xs.push_back(x); };
		NewtonResult<double> const r = newtonRaphson([](double x) { return x * x - 2.0; }, [](double x) { return 2.0 * x; }, 1.0, NewtonOptions<double>{}, record);
		CHECK(!xs.empty() && xs.front() == 1.0);
		CHECK(xs.size() == static_cast<std::size_t>(r.iterations) || xs.size() == static_cast<std::size_t>(r.iterations) + 1);
		CHECK(xs.size() >= 2 && xs[1] == 1.5);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testFloat()
	{
		NewtonResult<float> const r = newtonRaphson([](float x) { return x * x - 2.0f; }, [](float x) { return 2.0f * x; }, 1.0f);
		CHECK(r.status == NewtonStatus::Converged);
		CHECK(test::near(r.root, std::numbers::sqrt2_v<float>, 1e-6f));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void runAllTests()
	{
		testSquareRootOfTwo();
		testClassicExamples();
		testExactRootAndLine();
		testFailures();
		testNotFinite();
		testStepCallback();
		testFloat();
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main()
{
	runAllTests();
	return test::report("newtonRaphson");
}
