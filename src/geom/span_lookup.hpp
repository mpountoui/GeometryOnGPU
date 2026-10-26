#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include "geom/basis.hpp"
#include "geom/host_device.hpp"
#include "geom/nurbs_curve.hpp"

namespace geom
{
	// A precomputed table that finds the span of u in O(1) instead of findSpan's binary search, with
	// exactly the same result. See docs/algorithms/span-lookup.md.
	//
	// The domain [a, b] is split into equal cells; firstSpans()[cell] is the span that contains the
	// cell's left edge. A query jumps to its cell, starts at that span and walks a step or two over
	// the knots (forward, or back after rounding at a cell edge).
	//
	// The table does not store the knots: query it with the knots it was built from.
	template <std::floating_point T>
	class SpanLookup
	{
	public:
		// For a valid clamped knot vector (as NurbsCurve guarantees). cellsPerSpan cells for every
		// span index p..n; more cells mean fewer walk steps where many knots are close together.
		SpanLookup(int degree, std::span<T const> knots, std::size_t cellsPerSpan = 2);
		
		explicit SpanLookup(NurbsCurve<T> const& curve, std::size_t cellsPerSpan = 2);
		
	public:
		int degree() const noexcept;
		
		// a = u_p, the start of the domain.
		T start() const noexcept;
		
		// cells / (b - a): cell = floor((u - a) * cellsPerUnit()).
		T cellsPerUnit() const noexcept;
		
		// One span index per cell.
		std::span<std::uint32_t const> firstSpans() const noexcept;
		
		// The span of u, equal to findSpan(degree, knots, u). knots must be the ones the table was built from.
		std::size_t find(std::span<T const> knots, T u) const;
		
	private:
		// One pass over cells and knots together: the span of each cell's left edge, as findSpan would
		// give it, in O(cells + knots) instead of one binary search per cell.
		void fillCells(std::span<T const> knots);
		
	private:
		int m_degree{};
		T m_start{};
		T m_cellsPerUnit{};
		std::vector<std::uint32_t> m_firstSpans{};
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The span of u from a SpanLookup's data (firstSpans, start, cellsPerUnit) and the knots it was
	// built from. Returns the same as findSpan(degree, knots, u) for every u.
	template <std::floating_point T>
	GEOM_HD std::size_t findSpanWithLookup(int degree, std::type_identity_t<std::span<T const>> knots, std::span<std::uint32_t const> firstSpans, T start, T cellsPerUnit, T u);
	
	namespace detail
	{
		// floor((u - start) * cellsPerUnit), clamped to [0, numCells - 1] (also for NaN).
		template <std::floating_point T>
		GEOM_HD std::size_t cellIndex(T u, T start, T cellsPerUnit, std::size_t numCells);
		
		// From a start span near u, walks forward while u >= u_{span+1} and back while u < u_span.
		// Precondition: a < u < b.
		template <std::floating_point T>
		GEOM_HD std::size_t walkToSpan(std::span<T const> knots, std::size_t span, T u);
	}  // namespace detail
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// numSpans = n + 1 - p; every cell gets the span of its left edge (fillCells).
	template <std::floating_point T>
	SpanLookup<T>::SpanLookup(int degree, std::span<T const> knots, std::size_t cellsPerSpan)
		: m_degree(degree),
		m_start(knots[static_cast<std::size_t>(degree)]),
		m_cellsPerUnit{},
		m_firstSpans(std::max<std::size_t>(1, cellsPerSpan * (knots.size() - 2 * static_cast<std::size_t>(degree) - 1)))
	{
		assert(degree >= 1 && cellsPerSpan >= 1);
		T const end = knots[knots.size() - 1 - static_cast<std::size_t>(degree)];
		m_cellsPerUnit = static_cast<T>(m_firstSpans.size()) / (end - m_start);
		fillCells(knots);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	SpanLookup<T>::SpanLookup(NurbsCurve<T> const& curve, std::size_t cellsPerSpan)
		: SpanLookup(curve.degree(), curve.knots(), cellsPerSpan)
	{
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	int SpanLookup<T>::degree() const noexcept
	{
		return m_degree;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	T SpanLookup<T>::start() const noexcept
	{
		return m_start;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	T SpanLookup<T>::cellsPerUnit() const noexcept
	{
		return m_cellsPerUnit;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::span<std::uint32_t const> SpanLookup<T>::firstSpans() const noexcept
	{
		return m_firstSpans;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::size_t SpanLookup<T>::find(std::span<T const> knots, T u) const
	{
		return findSpanWithLookup<T>(m_degree, knots, m_firstSpans, m_start, m_cellsPerUnit, u);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The left edges increase with the cell, so the span only moves forward. It stops at n, which
	// also covers a left edge that rounding pushed to b.
	template <std::floating_point T>
	void SpanLookup<T>::fillCells(std::span<T const> knots)
	{
		auto const p = static_cast<std::size_t>(m_degree);
		std::size_t const n = knots.size() - p - 2;
		std::size_t span = p;
		for (std::size_t cell = 0; cell < m_firstSpans.size(); ++cell)
		{
			T const leftEdge = m_start + static_cast<T>(cell) / m_cellsPerUnit;
			while (span < n && leftEdge >= knots[span + 1])
			{
				++span;
			}
			m_firstSpans[cell] = static_cast<std::uint32_t>(span);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The two special cases first, as in findSpan; then a < u < b, so the walks stay in p..n and stop on
	// the non-empty span with u_span <= u < u_{span+1} (see "Correctness" in the doc).
	template <std::floating_point T>
	GEOM_HD std::size_t findSpanWithLookup(int degree, std::type_identity_t<std::span<T const>> knots, std::span<std::uint32_t const> firstSpans, T start, T cellsPerUnit, T u)
	{
		auto const p = static_cast<std::size_t>(degree);
		std::size_t const n = knots.size() - p - 2;
		if (u >= knots[n + 1])
		{
			return n;
		}
		if (u <= knots[p])
		{
			return p;
		}
		std::size_t const first = firstSpans[detail::cellIndex(u, start, cellsPerUnit, firstSpans.size())];
		return detail::walkToSpan<T>(knots, first, u);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Clamps before the conversion: converting a value outside the range of std::size_t is undefined.
	template <std::floating_point T>
	GEOM_HD std::size_t detail::cellIndex(T u, T start, T cellsPerUnit, std::size_t numCells)
	{
		T const x = (u - start) * cellsPerUnit;
		if (!(x > T{0}))
		{
			return 0;
		}
		if (x >= static_cast<T>(numCells))
		{
			return numCells - 1;
		}
		return static_cast<std::size_t>(x);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	GEOM_HD std::size_t detail::walkToSpan(std::span<T const> knots, std::size_t span, T u)
	{
		while (u >= knots[span + 1])
		{
			++span;
		}
		while (u < knots[span])
		{
			--span;
		}
		return span;
	}
}  // namespace geom
