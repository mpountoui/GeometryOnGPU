#pragma once

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

#include "geom/basis.hpp"
#include "geom/basis_polynomials.hpp"
#include "geom/curve_point.hpp"
#include "geom/host_device.hpp"
#include "geom/nurbs_curve.hpp"
#include "geom/point.hpp"

namespace geom
{
	// The basis polynomials of every span of a curve, computed once, so that a point C(u) needs no
	// A2.2: find the span, evaluate its p + 1 polynomials at t, take the weighted sum. See
	// "Evaluation from the basis table" in docs/algorithms/basis-polynomials.md.
	//
	// Layout: one row per span j = p..n (row j - p), so findSpan gives the row directly.
	//   coefficients():       rows of (p + 1)^2 values, each the table of basisPolynomials for span j
	//                         (row r = N_{j-p+r,p}, column d = coefficient of t^d)
	//   inverseSpanLengths(): 1 / (u_{j+1} - u_j), so t = (u - u_j) * inverseSpanLengths()[j - p]
	// Empty spans (repeated knots) keep zeros: findSpan never returns them.
	//
	// The table does not store the knots: evaluation also needs the curve's knots, so build and use
	// it with the same curve. Host-side owner (std::vector); curvePointFromTable runs on raw spans.
	template <std::floating_point T>
	class BasisTable
	{
	public:
		// For a valid clamped knot vector (as NurbsCurve guarantees) and degree <= kMaxDegree.
		BasisTable(int degree, std::span<T const> knots);
		
		explicit BasisTable(NurbsCurve<T> const& curve);
		
	public:
		int degree() const noexcept;
		
		// Rows in the table: n + 1 - p, one per span index j = p..n.
		std::size_t numSpans() const noexcept;
		
		std::span<T const> coefficients() const noexcept;
		std::span<T const> inverseSpanLengths() const noexcept;
		
		// The (p + 1)^2 coefficients of span j (p <= j <= n).
		std::span<T const> spanCoefficients(std::size_t span) const;
		
	private:
		int m_degree{};
		std::size_t m_numSpans{};
		std::vector<T> m_coefficients{};
		std::vector<T> m_inverseSpanLengths{};
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// C(u) from the table of the same curve: findSpan, t, the p + 1 basis polynomials (Horner),
	// then the weighted sum and projection as in curvePoint. Preconditions as for curvePoint.
	template <std::floating_point T>
	GEOM_HD Point3<T> curvePointFromTable(int degree, std::type_identity_t<std::span<T const>> knots, std::type_identity_t<std::span<Point4<T> const>> weightedPoints, std::type_identity_t<std::span<T const>> coefficients, std::type_identity_t<std::span<T const>> inverseSpanLengths, T u);
	
	// Same, for a curve and its table.
	template <std::floating_point T>
	Point3<T> curvePointFromTable(NurbsCurve<T> const& curve, BasisTable<T> const& table, T u);
	
	namespace detail
	{
		// Fills row `row` (span j = row + p) of a basis table: its (p + 1)^2 coefficients in
		// coefficients and 1 / h_j in inverseSpanLengths (both the whole table), or zeros if the span is
		// empty. Writes every value of the row, so the table need not start zeroed (cudaMalloc memory
		// does not). Shared by BasisTable (one call per row) and the GPU kernel (one thread per row).
		template <std::floating_point T>
		GEOM_HD void buildBasisRow(std::size_t row, int degree, std::span<T const> knots, std::span<T> coefficients, std::span<T> inverseSpanLengths);
	}  // namespace detail
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// numSpans = n + 1 - p with n = knots.size() - p - 2.
	template <std::floating_point T>
	BasisTable<T>::BasisTable(int degree, std::span<T const> knots)
		: m_degree(degree),
		m_numSpans(knots.size() - 2 * static_cast<std::size_t>(degree) - 1),
		m_coefficients(m_numSpans * (static_cast<std::size_t>(degree) + 1) * (static_cast<std::size_t>(degree) + 1)),
		m_inverseSpanLengths(m_numSpans)
	{
		assert(degree >= 1 && degree <= kMaxDegree);
		for (std::size_t row = 0; row < m_numSpans; ++row)
		{
			detail::buildBasisRow<T>(row, degree, knots, m_coefficients, m_inverseSpanLengths);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	BasisTable<T>::BasisTable(NurbsCurve<T> const& curve)
		: BasisTable(curve.degree(), curve.knots())
	{
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	int BasisTable<T>::degree() const noexcept
	{
		return m_degree;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::size_t BasisTable<T>::numSpans() const noexcept
	{
		return m_numSpans;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::span<T const> BasisTable<T>::coefficients() const noexcept
	{
		return m_coefficients;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::span<T const> BasisTable<T>::inverseSpanLengths() const noexcept
	{
		return m_inverseSpanLengths;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::span<T const> BasisTable<T>::spanCoefficients(std::size_t span) const
	{
		auto const p = static_cast<std::size_t>(m_degree);
		std::size_t const size = (p + 1) * (p + 1);
		assert(span >= p && span - p < m_numSpans);
		return std::span<T const>(m_coefficients).subspan((span - p) * size, size);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	GEOM_HD Point3<T> curvePointFromTable(int degree, std::type_identity_t<std::span<T const>> knots, std::type_identity_t<std::span<Point4<T> const>> weightedPoints, std::type_identity_t<std::span<T const>> coefficients, std::type_identity_t<std::span<T const>> inverseSpanLengths, T u)
	{
		auto const p = static_cast<std::size_t>(degree);
		std::size_t const stride = p + 1;
		std::size_t const span = findSpan(degree, knots, u);
		T const t = (u - knots[span]) * inverseSpanLengths[span - p];
		std::span<T const> const table = coefficients.subspan((span - p) * stride * stride, stride * stride);
		std::array<T, kMaxDegree + 1> basis{};
		for (std::size_t r = 0; r < stride; ++r)
		{
			basis[r] = evaluatePolynomial<T>(table.subspan(r * stride, stride), t);
		}
		return Get3DPoint(detail::weightedSum<T>(std::span<T const>(basis.data(), stride), weightedPoints, span - p));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	Point3<T> curvePointFromTable(NurbsCurve<T> const& curve, BasisTable<T> const& table, T u)
	{
		return curvePointFromTable<T>(curve.degree(), curve.knots(), curve.weightedControlPoints(), table.coefficients(), table.inverseSpanLengths(), u);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	GEOM_HD void detail::buildBasisRow(std::size_t row, int degree, std::span<T const> knots, std::span<T> coefficients, std::span<T> inverseSpanLengths)
	{
		auto const p = static_cast<std::size_t>(degree);
		std::size_t const size = (p + 1) * (p + 1);
		std::size_t const span = row + p;
		std::span<T> const rowCoefficients = coefficients.subspan(row * size, size);
		if (!(knots[span] < knots[span + 1]))
		{
			for (T& c : rowCoefficients)
			{
				c = T{0};
			}
			inverseSpanLengths[row] = T{0};
			return;
		}
		inverseSpanLengths[row] = T{1} / (knots[span + 1] - knots[span]);
		basisPolynomials<T>(span, degree, knots, rowCoefficients);
	}
}  // namespace geom
