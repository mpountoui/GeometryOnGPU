#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

#include "geom/host_device.hpp"

namespace geom
{
	// B-spline basis functions, The NURBS Book Section 2.5 (pp.67-70). See
	// docs/algorithms/basis-functions.md.
	//
	// The knot vector must be a valid clamped one (as NurbsCurve guarantees), so the book's
	// indices apply: m = knots.size() - 1 and n = m - p - 1.
	//
	// The span parameters use std::type_identity_t so that T is deduced from u alone. That lets
	// callers pass a std::vector or a span directly; without it, template deduction would fail.
	//
	// No allocation and no exceptions: this code is meant to run inside GPU kernels later.
	
	// Highest supported degree. Evaluation keeps the p + 1 basis values in a fixed-size local
	// array (registers on the GPU), so the degree needs an upper bound. NurbsCurve enforces it.
	inline constexpr int kMaxDegree = 9;
	
	// A2.1 FindSpan (p.68): the span index i with u in [u_i, u_{i+1}). For u == u_m it returns n,
	// the last non-empty span. Precondition: u in [u_p, u_m]; outside it, u is clamped to the
	// first or last span.
	template <std::floating_point T>
	GEOM_HD std::size_t findSpan(int degree, std::type_identity_t<std::span<T const>> knots, T u);
	
	// A2.2 BasisFuns (p.70): the p + 1 nonvanishing basis functions at u in the given span.
	// values must hold degree + 1 entries; values[r] = N_{span-p+r, p}(u).
	template <std::floating_point T>
	GEOM_HD void basisFunctions(std::size_t span, T u, int degree, std::type_identity_t<std::span<T const>> knots, std::type_identity_t<std::span<T>> values);
	
	namespace detail
	{
		// The binary search of A2.1 on the span indices [low, high).
		template <std::floating_point T>
		GEOM_HD std::size_t binarySearchSpan(std::span<T const> knots, T u, std::size_t low, std::size_t high);
	}  // namespace detail
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	GEOM_HD std::size_t findSpan(int degree, std::type_identity_t<std::span<T const>> knots, T u)
	{
		auto const p = static_cast<std::size_t>(degree);
		std::size_t const n = knots.size() - p - 2;
		assert(u >= knots[p] && u <= knots[n + 1]);
		if (u >= knots[n + 1])
		{
			return n;
		}
		if (u <= knots[p])
		{
			return p;
		}
		return detail::binarySearchSpan(knots, u, p, n + 1);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// left[j - r] = u - U[span + 1 + r - j] and right[r + 1] = U[span + 1 + r] - u are the book's
	// scratch arrays, recomputed from the knots instead of stored (see the doc).
	template <std::floating_point T>
	GEOM_HD void basisFunctions(std::size_t span, T u, int degree, std::type_identity_t<std::span<T const>> knots, std::type_identity_t<std::span<T>> values)
	{
		auto const p = static_cast<std::size_t>(degree);
		assert(values.size() == p + 1);
		values[0] = T{1};
		for (std::size_t j = 1; j <= p; ++j)
		{
			T saved = T{0};
			for (std::size_t r = 0; r < j; ++r)
			{
				T const right = knots[span + 1 + r] - u;
				T const left = u - knots[span + 1 + r - j];
				T const temp = values[r] / (right + left);
				values[r] = saved + right * temp;
				saved = left * temp;
			}
			values[j] = saved;
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	GEOM_HD std::size_t detail::binarySearchSpan(std::span<T const> knots, T u, std::size_t low, std::size_t high)
	{
		std::size_t mid = (low + high) / 2;
		while (u < knots[mid] || u >= knots[mid + 1])
		{
			if (u < knots[mid])
			{
				high = mid;
			}
			else
			{
				low = mid;
			}
			mid = (low + high) / 2;
		}
		return mid;
	}
}  // namespace geom
