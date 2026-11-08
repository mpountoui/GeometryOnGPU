#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

#include "geom/host_device.hpp"

namespace geom
{
	// The values 1 / h_j = 1 / (u_{j+1} - u_j) of every span j = p..n, in rows j - p, 0 for empty spans.
	// A query needs them to get its local parameter, t = (u - u_j) * inverseSpanLengths[j - p]. They depend
	// only on the knots, so one array serves every table built from those knots.
	
	// Fills numSpans = knots.size() - 2p - 1 values into memory the caller owns, without allocating.
	// Writes every value, so the memory need not be zeroed.
	template <std::floating_point T>
	void fillInverseSpanLengths(int degree, std::span<T const> knots, std::span<T> inverseSpanLengths);
	
	// Allocates the array, then fillInverseSpanLengths.
	template <std::floating_point T>
	std::vector<T> buildInverseSpanLengths(int degree, std::span<T const> knots);
	
	namespace detail
	{
		// 1 / h_j of one span, or 0 if the span is empty. Shared by the CPU loop and the GPU kernel.
		template <std::floating_point T>
		GEOM_HD T inverseSpanLength(std::size_t span, std::span<T const> knots);
	}  // namespace detail
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void fillInverseSpanLengths(int degree, std::span<T const> knots, std::span<T> inverseSpanLengths)
	{
		auto const p = static_cast<std::size_t>(degree);
		assert(knots.size() == inverseSpanLengths.size() + 2 * p + 1);
		for (std::size_t row = 0; row < inverseSpanLengths.size(); ++row)
		{
			inverseSpanLengths[row] = detail::inverseSpanLength<T>(row + p, knots);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::vector<T> buildInverseSpanLengths(int degree, std::span<T const> knots)
	{
		std::vector<T> inverseSpanLengths(knots.size() - 2 * static_cast<std::size_t>(degree) - 1);
		fillInverseSpanLengths<T>(degree, knots, inverseSpanLengths);
		return inverseSpanLengths;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	GEOM_HD T detail::inverseSpanLength(std::size_t span, std::span<T const> knots)
	{
		if (!(knots[span] < knots[span + 1]))
		{
			return T{0};
		}
		return T{1} / (knots[span + 1] - knots[span]);
	}
}  // namespace geom
