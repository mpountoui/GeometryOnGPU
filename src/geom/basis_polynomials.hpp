#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

#include "geom/host_device.hpp"

namespace geom
{
	// The B-spline basis functions of one knot span as polynomials, for precomputed evaluation.
	// See docs/algorithms/basis-polynomials.md.
	//
	// The polynomials are in the local parameter t = (u - u_span) / (u_{span+1} - u_span), which
	// runs over [0, 1] on the span, instead of in u (better conditioned, see the doc). The knots
	// must be a valid clamped knot vector, as for basisFunctions.
	//
	// No allocation and no exceptions: the helpers in detail are shared with the GPU kernel.
	
	// A linear polynomial beta + alpha * t (one factor of Eq. (2.5), written in t).
	template <std::floating_point T>
	struct LinearFactor
	{
		T beta{};
		T alpha{};
	};
	
	// The coefficients of the p + 1 basis functions that are nonzero on the span (as findSpan
	// returns it, so u_span < u_{span+1}). No parameter has type T, so callers name it:
	// basisPolynomials<double>(...).
	//
	// coefficients is a (p + 1) x (p + 1) table stored row by row in a flat array of (p + 1)^2
	// entries, stride = p + 1:
	//
	//   coefficients[r * stride + d] = coefficient of t^d in N_{span-p+r,p}
	//
	//              d = 0 (t^0)   d = 1 (t^1)   ...   d = p (t^p)
	//   row r = 0  [ c(0,0)        c(0,1)      ...     c(0,p) ]   <- N_{span-p,p}
	//   row r = 1  [ c(1,0)        c(1,1)      ...     c(1,p) ]   <- N_{span-p+1,p}
	//   ...
	//   row r = p  [ c(p,0)        c(p,1)      ...     c(p,p) ]   <- N_{span,p}
	//
	// So a row is one basis function and a column one power of t:
	//   N_{span-p+r,p}(t) = c(r,0) + c(r,1) t + ... + c(r,p) t^p,
	// evaluated with evaluatePolynomial on the row, coefficients.subspan(r * stride, stride).
	// Row r matches values[r] of basisFunctions on the same span.
	//
	// Example, p = 2 on the only span of the knots {0,0,0,1,1,1} (the Bernstein basis):
	//   [ 1 -2  1 ]   N_{0,2} = (1 - t)^2 = 1 - 2t + t^2
	//   [ 0  2 -2 ]   N_{1,2} = 2t(1 - t) = 2t - 2t^2
	//   [ 0  0  1 ]   N_{2,2} = t^2
	//
	// In float, accuracy matches basisFunctions up to p = 5 and drops for higher degrees (power
	// basis cancellation); see "Precision" in the doc.
	template <std::floating_point T>
	GEOM_HD void basisPolynomials(std::size_t span, int degree, std::type_identity_t<std::span<T const>> knots, std::type_identity_t<std::span<T>> coefficients);
	
	// t = (u - u_span) / (u_{span+1} - u_span), the local parameter of u in a non-empty span.
	template <std::floating_point T>
	GEOM_HD T localParameter(std::size_t span, std::type_identity_t<std::span<T const>> knots, T u);
	
	// sum_d coefficients[d] * t^d, with Horner's method.
	template <std::floating_point T>
	GEOM_HD T evaluatePolynomial(std::type_identity_t<std::span<T const>> coefficients, T t);
	
	namespace detail
	{
		// L = (u - u_i) / (u_{i+k} - u_i) of Eq. (2.5) in the t of the span; 0 if u_{i+k} == u_i.
		template <std::floating_point T>
		GEOM_HD LinearFactor<T> leftFactor(std::span<T const> knots, std::size_t span, std::size_t i, std::size_t k);
		
		// R = (u_{i+k+1} - u) / (u_{i+k+1} - u_{i+1}) in the t of the span; 0 if the denominator is 0.
		template <std::floating_point T>
		GEOM_HD LinearFactor<T> rightFactor(std::span<T const> knots, std::size_t span, std::size_t i, std::size_t k);
		
		// Entry (r, d) of level k, from level k - 1 stored in previous (same layout, row stride
		// stride): the coefficient of t^d in left * c_{r-1} + right * c_r, where c_{r-1} and c_r are
		// rows r - 1 and r of level k - 1. Reads the four entries (r-1, d-1), (r-1, d), (r, d-1) and
		// (r, d). Row k and column k of previous must be 0 (they stand for the missing neighbour
		// function and the missing power t^k).
		template <std::floating_point T>
		GEOM_HD T levelCoefficient(std::span<T const> previous, std::size_t stride, std::size_t r, std::size_t d, LinearFactor<T> left, LinearFactor<T> right);
		
		// Replaces level k - 1 in coefficients by level k, in place (r and d run downwards).
		template <std::floating_point T>
		GEOM_HD void raiseLevelInPlace(std::size_t span, std::size_t k, std::span<T const> knots, std::span<T> coefficients, std::size_t stride);
	}  // namespace detail
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The table is built degree by degree in place (see raiseLevelInPlace). At level k only the
	// top-left (k + 1) x (k + 1) block is in use: row r holds N_{span-k+r,k}, columns 0..k its
	// coefficients. Everything outside the block is 0 and acts as the zero function (row k + 1) and
	// zero coefficient (column k + 1) the recurrence needs at the edges. For the example above:
	//
	//   level 0          level 1                 level 2
	//   [ 1  0  0 ]      [ 1 -1  0 ]  1 - t      [ 1 -2  1 ]
	//   [ 0  0  0 ]      [ 0  1  0 ]  t          [ 0  2 -2 ]
	//   [ 0  0  0 ]      [ 0  0  0 ]             [ 0  0  1 ]
	//
	// Level 0 is N_{span,0} = 1 in entry (0, 0); after level p the table is the result.
	template <std::floating_point T>
	GEOM_HD void basisPolynomials(std::size_t span, int degree, std::type_identity_t<std::span<T const>> knots, std::type_identity_t<std::span<T>> coefficients)
	{
		auto const p = static_cast<std::size_t>(degree);
		std::size_t const stride = p + 1;
		assert(coefficients.size() == stride * stride);
		assert(knots[span] < knots[span + 1]);
		for (T& c : coefficients)
		{
			c = T{0};
		}
		coefficients[0] = T{1};
		for (std::size_t k = 1; k <= p; ++k)
		{
			detail::raiseLevelInPlace<T>(span, k, knots, coefficients, stride);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	GEOM_HD T localParameter(std::size_t span, std::type_identity_t<std::span<T const>> knots, T u)
	{
		return (u - knots[span]) / (knots[span + 1] - knots[span]);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	GEOM_HD T evaluatePolynomial(std::type_identity_t<std::span<T const>> coefficients, T t)
	{
		T value = T{0};
		for (std::size_t d = coefficients.size(); d > 0; --d)
		{
			value = value * t + coefficients[d - 1];
		}
		return value;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// With u = u_span + h t: (u_span - u_i + h t) / dL.
	template <std::floating_point T>
	GEOM_HD LinearFactor<T> detail::leftFactor(std::span<T const> knots, std::size_t span, std::size_t i, std::size_t k)
	{
		T const denominator = knots[i + k] - knots[i];
		if (denominator == T{0})
		{
			return {};
		}
		T const h = knots[span + 1] - knots[span];
		return {(knots[span] - knots[i]) / denominator, h / denominator};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// With u = u_span + h t: (u_{i+k+1} - u_span - h t) / dR.
	template <std::floating_point T>
	GEOM_HD LinearFactor<T> detail::rightFactor(std::span<T const> knots, std::size_t span, std::size_t i, std::size_t k)
	{
		T const denominator = knots[i + k + 1] - knots[i + 1];
		if (denominator == T{0})
		{
			return {};
		}
		T const h = knots[span + 1] - knots[span];
		return {(knots[i + k + 1] - knots[span]) / denominator, -h / denominator};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// (beta + alpha t) * c has the coefficients beta * c[d] + alpha * c[d-1].
	template <std::floating_point T>
	GEOM_HD T detail::levelCoefficient(std::span<T const> previous, std::size_t stride, std::size_t r, std::size_t d, LinearFactor<T> left, LinearFactor<T> right)
	{
		std::size_t const index = r * stride + d;
		T value = right.beta * previous[index];
		if (d > 0)
		{
			value += right.alpha * previous[index - 1];
		}
		if (r > 0)
		{
			value += left.beta * previous[index - stride];
			if (d > 0)
			{
				value += left.alpha * previous[index - stride - 1];
			}
		}
		return value;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One step of Eq. (2.5): turns the k functions of degree k - 1 (rows 0..k-1, columns 0..k-1
	// of the table) into the k + 1 functions of degree k (rows 0..k, columns 0..k).
	//
	// The same row number means a different function on each level: on level k, row r is N_{i,k}
	// with i = span - k + r (the functions nonzero on the span). Eq. (2.5) builds it from two
	// functions of level k - 1:
	//   N_{i,k} = L * N_{i,k-1} + R * N_{i+1,k-1},
	// and on level k - 1 (rows numbered from span - k + 1) these two are rows r - 1 and r.
	// Example: level 2, row 1 (N_{1,2}) is made from level 1, rows 0 (N_{1,1}) and 1 (N_{2,1}).
	// Multiplying a row by a line beta + alpha t moves part of it one column right (t^(d-1) -> t^d),
	// so entry (r, d) of the new level reads the old entries (r - 1, d - 1), (r - 1, d), (r, d - 1)
	// and (r, d) (see levelCoefficient).
	//
	// Why in place works: all four have row <= r and column <= d. With r and d both running
	// downwards (from k to 0), every entry is overwritten only after the last entry that reads it.
	// Example for k = 2: the order is (2,2), (2,1), (2,0), (1,2), ... and (1,1) reads (1,1), (1,0),
	// (0,1), (0,0), none of which has been written yet in this level.
	//
	// The loops count from k + 1 down to 1 and use r - 1 and d - 1, because the indices are
	// unsigned: "r >= 0" would always be true and the loop would never end.
	template <std::floating_point T>
	GEOM_HD void detail::raiseLevelInPlace(std::size_t span, std::size_t k, std::span<T const> knots, std::span<T> coefficients, std::size_t stride)
	{
		for (std::size_t r = k + 1; r > 0; --r)
		{
			// The function of this row and its two factors L and R, as lines in t.
			std::size_t const i = span - k + (r - 1);
			LinearFactor<T> const left = leftFactor(knots, span, i, k);
			LinearFactor<T> const right = rightFactor(knots, span, i, k);
			// Its k + 1 coefficients (powers t^k .. t^0), highest first.
			for (std::size_t d = k + 1; d > 0; --d)
			{
				coefficients[(r - 1) * stride + (d - 1)] = levelCoefficient<T>(coefficients, stride, r - 1, d - 1, left, right);
			}
		}
	}
}  // namespace geom
