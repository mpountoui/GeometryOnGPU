#pragma once

#include <cmath>
#include <concepts>
#include <limits>

#include "geom/host_device.hpp"

namespace geom
{
	// Newton-Raphson root finding: x with f(x) = 0, from a start value x0, using
	// x_{i+1} = x_i - f(x_i) / f'(x_i). See docs/algorithms/newton-raphson.md.
	//
	// f and df are any callables T -> T (lambdas, functions, function objects); df is f'.
	// No allocation and no exceptions, so it can run inside a kernel when f and df can.
	
	enum class NewtonStatus
	{
		Converged,         // |f(x)| <= valueTolerance, or the last step was tiny
		MaxIterations,     // no convergence within maxIterations steps
		ZeroDerivative,    // f'(x) == 0: the tangent never crosses zero
		NotFinite          // f(x) or f'(x) became NaN or infinite
	};
	
	template <std::floating_point T>
	struct NewtonOptions
	{
		// |f(x)| at which x counts as a root. It bounds f, not x: the error in x is about |f| / |f'|,
		// so the default only accepts an (almost) exact zero and leaves stopping to the step test.
		T valueTolerance{std::numeric_limits<T>::epsilon()};
		T stepTolerance{T{100} * std::numeric_limits<T>::epsilon()};    // relative to max(1, |x|)
		int maxIterations{50};
	};
	
	template <std::floating_point T>
	struct NewtonResult
	{
		T root{};          // the last x
		T value{};         // f(root)
		int iterations{};  // Newton steps taken
		NewtonStatus status{NewtonStatus::MaxIterations};
	};
	
	// The default step callback: ignores every step.
	struct IgnoreSteps
	{
		template <typename T>
		GEOM_HD void operator()(int i, T x, T fx, T dfx) const;
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The root of f near x0.
	template <std::floating_point T, typename F, typename DF>
	GEOM_HD NewtonResult<T> newtonRaphson(F f, DF df, T x0, NewtonOptions<T> options = {});
	
	// Same, and calls onStep(i, x_i, f(x_i), f'(x_i)) before every step (for plots or logging).
	template <std::floating_point T, typename F, typename DF, typename OnStep>
	GEOM_HD NewtonResult<T> newtonRaphson(F f, DF df, T x0, NewtonOptions<T> options, OnStep onStep);
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	namespace detail
	{
		// Step i from x. Returns true and fills result when the iteration is finished;
		// otherwise moves x to the next iterate and returns false.
		template <std::floating_point T, typename F, typename DF, typename OnStep>
		GEOM_HD bool newtonIteration(F& f, DF& df, NewtonOptions<T> const& options, OnStep& onStep, int i, T& x, NewtonResult<T>& result);
		
		// True (and sets status) if the iteration must stop at a point with value fx and slope dfx.
		template <std::floating_point T>
		GEOM_HD bool stopsAt(T fx, T dfx, T valueTolerance, NewtonStatus& status);
		
		// True if the step from x to next is small: |next - x| <= tolerance * max(1, |next|).
		template <std::floating_point T>
		GEOM_HD bool isTinyStep(T x, T next, T tolerance);
	}  // namespace detail
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <typename T>
	GEOM_HD void IgnoreSteps::operator()(int, T, T, T) const
	{
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T, typename F, typename DF>
	GEOM_HD NewtonResult<T> newtonRaphson(F f, DF df, T x0, NewtonOptions<T> options)
	{
		return newtonRaphson(f, df, x0, options, IgnoreSteps{});
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T, typename F, typename DF, typename OnStep>
	GEOM_HD NewtonResult<T> newtonRaphson(F f, DF df, T x0, NewtonOptions<T> options, OnStep onStep)
	{
		T x = x0;
		NewtonResult<T> result{};
		for (int i = 0; i < options.maxIterations; ++i)
		{
			if (detail::newtonIteration(f, df, options, onStep, i, x, result))
			{
				return result;
			}
		}
		return {x, f(x), options.maxIterations, NewtonStatus::MaxIterations};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T, typename F, typename DF, typename OnStep>
	GEOM_HD bool detail::newtonIteration(F& f, DF& df, NewtonOptions<T> const& options, OnStep& onStep, int i, T& x, NewtonResult<T>& result)
	{
		T const fx = f(x);
		T const dfx = df(x);
		onStep(i, x, fx, dfx);
		NewtonStatus status{};
		if (stopsAt(fx, dfx, options.valueTolerance, status))
		{
			result = {x, fx, i, status};
			return true;
		}
		T const next = x - fx / dfx;
		if (isTinyStep(x, next, options.stepTolerance))
		{
			result = {next, f(next), i + 1, NewtonStatus::Converged};
			return true;
		}
		x = next;
		return false;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	GEOM_HD bool detail::stopsAt(T fx, T dfx, T valueTolerance, NewtonStatus& status)
	{
		if (!std::isfinite(fx) || !std::isfinite(dfx))
		{
			status = NewtonStatus::NotFinite;
			return true;
		}
		if (std::abs(fx) <= valueTolerance)
		{
			status = NewtonStatus::Converged;
			return true;
		}
		if (dfx == T{0})
		{
			status = NewtonStatus::ZeroDerivative;
			return true;
		}
		return false;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	GEOM_HD bool detail::isTinyStep(T x, T next, T tolerance)
	{
		T const scale = (std::abs(next) > T{1}) ? std::abs(next) : T{1};
		return std::abs(next - x) <= tolerance * scale;
	}
}  // namespace geom
