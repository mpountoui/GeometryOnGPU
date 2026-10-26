#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

#include "geom/basis.hpp"
#include "geom/basis_table.hpp"
#include "geom/host_device.hpp"
#include "geom/nurbs_curve.hpp"
#include "geom/point.hpp"
#include "geom/span_lookup.hpp"

namespace geom
{
	// The curve itself as one polynomial per span, computed once: the basis table multiplied by the
	// control points. A point C(u) then needs no basis functions at all: find the span, one Horner
	// evaluation of a 4D polynomial in t, divide by w. See "Evaluation from a curve table" in
	// docs/algorithms/basis-polynomials.md.
	//
	// On span j, with c(r, d) the span's basis coefficients (BasisTable):
	//   Cw(t) = sum_r N_{j-p+r,p}(t) Pw[j-p+r] = sum_d A_j[d] t^d,   A_j[d] = sum_r c(r, d) Pw[j-p+r]
	//
	// Layout: one row per span j = p..n (row j - p), as in BasisTable.
	//   coefficients():       rows of p + 1 weighted points A_j[0..p] (A_j[d] multiplies t^d)
	//   inverseSpanLengths(): 1 / (u_{j+1} - u_j), copied from the BasisTable
	// Empty spans (repeated knots) keep zeros: findSpan never returns them.
	//
	// Unlike a BasisTable it depends on the control points and weights: rebuild it when they change.
	// The knots are not stored: evaluate with the knots of the same curve.
	template <std::floating_point T>
	class CurveTable
	{
	public:
		// From the basis table of a curve and that curve's weighted control points.
		CurveTable(BasisTable<T> const& basis, std::span<Point4<T> const> weightedPoints);
		
		// Builds the curve's BasisTable first.
		explicit CurveTable(NurbsCurve<T> const& curve);
		
	public:
		int degree() const noexcept;
		
		// Rows in the table: n + 1 - p, one per span index j = p..n.
		std::size_t numSpans() const noexcept;
		
		std::span<Point4<T> const> coefficients() const noexcept;
		std::span<T const> inverseSpanLengths() const noexcept;
		
		// A_j[0..p] of span j (p <= j <= n).
		std::span<Point4<T> const> spanCoefficients(std::size_t span) const;
		
	private:
		// A_j[d] = sum_r c(r, d) Pw[j-p+r] for the span of this row (zeros for an empty span).
		void buildRow(std::size_t row, BasisTable<T> const& basis, std::span<Point4<T> const> weightedPoints);
		
	private:
		int m_degree{};
		std::size_t m_numSpans{};
		std::vector<Point4<T>> m_coefficients{};
		std::vector<T> m_inverseSpanLengths{};
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// C(u) from the curve table of the same curve: findSpan, t, Horner on A_j, then C = Cw / w.
	// Preconditions as for curvePoint.
	template <std::floating_point T>
	GEOM_HD Point3<T> curvePointFromCurveTable(int degree, std::type_identity_t<std::span<T const>> knots, std::type_identity_t<std::span<Point4<T> const>> coefficients, std::type_identity_t<std::span<T const>> inverseSpanLengths, T u);
	
	// Same, for a curve and its curve table.
	template <std::floating_point T>
	Point3<T> curvePointFromTable(NurbsCurve<T> const& curve, CurveTable<T> const& table, T u);
	
	// C(u) from the curve table when the span of u is already known (from findSpan or a SpanLookup):
	// t, Horner on A_span, C = Cw / w.
	template <std::floating_point T>
	GEOM_HD Point3<T> curvePointFromCurveTable(int degree, std::type_identity_t<std::span<T const>> knots, std::type_identity_t<std::span<Point4<T> const>> coefficients, std::type_identity_t<std::span<T const>> inverseSpanLengths, std::size_t span, T u);
	
	// Same as curvePointFromTable(curve, table, u), with the span from the curve's SpanLookup
	// instead of the binary search.
	template <std::floating_point T>
	Point3<T> curvePointFromTable(NurbsCurve<T> const& curve, CurveTable<T> const& table, SpanLookup<T> const& lookup, T u);
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A curve has n + 1 = numSpans + p control points.
	template <std::floating_point T>
	CurveTable<T>::CurveTable(BasisTable<T> const& basis, std::span<Point4<T> const> weightedPoints)
		: m_degree(basis.degree()),
		m_numSpans(basis.numSpans()),
		m_coefficients(basis.numSpans() * (static_cast<std::size_t>(basis.degree()) + 1)),
		m_inverseSpanLengths(basis.inverseSpanLengths().begin(), basis.inverseSpanLengths().end())
	{
		assert(weightedPoints.size() == m_numSpans + static_cast<std::size_t>(m_degree));
		for (std::size_t row = 0; row < m_numSpans; ++row)
		{
			buildRow(row, basis, weightedPoints);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	CurveTable<T>::CurveTable(NurbsCurve<T> const& curve)
		: CurveTable(BasisTable<T>(curve), curve.weightedControlPoints())
	{
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	int CurveTable<T>::degree() const noexcept
	{
		return m_degree;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::size_t CurveTable<T>::numSpans() const noexcept
	{
		return m_numSpans;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::span<Point4<T> const> CurveTable<T>::coefficients() const noexcept
	{
		return m_coefficients;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::span<T const> CurveTable<T>::inverseSpanLengths() const noexcept
	{
		return m_inverseSpanLengths;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::span<Point4<T> const> CurveTable<T>::spanCoefficients(std::size_t span) const
	{
		auto const p = static_cast<std::size_t>(m_degree);
		assert(span >= p && span - p < m_numSpans);
		return std::span<Point4<T> const>(m_coefficients).subspan((span - p) * (p + 1), p + 1);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Row r of the span's basis table belongs to the control point Pw[row + r] (j - p + r = row + r).
	template <std::floating_point T>
	void CurveTable<T>::buildRow(std::size_t row, BasisTable<T> const& basis, std::span<Point4<T> const> weightedPoints)
	{
		auto const p = static_cast<std::size_t>(m_degree);
		std::span<T const> const c = basis.spanCoefficients(row + p);
		for (std::size_t d = 0; d <= p; ++d)
		{
			Point4<T> sum{};
			for (std::size_t r = 0; r <= p; ++r)
			{
				T const weight = c[r * (p + 1) + d];
				Point4<T> const& pw = weightedPoints[row + r];
				sum = {sum.x + weight * pw.x, sum.y + weight * pw.y, sum.z + weight * pw.z, sum.w + weight * pw.w};
			}
			m_coefficients[row * (p + 1) + d] = sum;
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	GEOM_HD Point3<T> curvePointFromCurveTable(int degree, std::type_identity_t<std::span<T const>> knots, std::type_identity_t<std::span<Point4<T> const>> coefficients, std::type_identity_t<std::span<T const>> inverseSpanLengths, T u)
	{
		return curvePointFromCurveTable<T>(degree, knots, coefficients, inverseSpanLengths, findSpan(degree, knots, u), u);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	Point3<T> curvePointFromTable(NurbsCurve<T> const& curve, CurveTable<T> const& table, T u)
	{
		return curvePointFromCurveTable<T>(curve.degree(), curve.knots(), table.coefficients(), table.inverseSpanLengths(), u);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Horner on 4D points: Cw = (...(A_p t + A_{p-1}) t + ...) t + A_0.
	template <std::floating_point T>
	GEOM_HD Point3<T> curvePointFromCurveTable(int degree, std::type_identity_t<std::span<T const>> knots, std::type_identity_t<std::span<Point4<T> const>> coefficients, std::type_identity_t<std::span<T const>> inverseSpanLengths, std::size_t span, T u)
	{
		auto const p = static_cast<std::size_t>(degree);
		T const t = (u - knots[span]) * inverseSpanLengths[span - p];
		std::span<Point4<T> const> const a = coefficients.subspan((span - p) * (p + 1), p + 1);
		Point4<T> cw = a[p];
		for (std::size_t d = p; d > 0; --d)
		{
			Point4<T> const& next = a[d - 1];
			cw = {cw.x * t + next.x, cw.y * t + next.y, cw.z * t + next.z, cw.w * t + next.w};
		}
		return Get3DPoint(cw);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	Point3<T> curvePointFromTable(NurbsCurve<T> const& curve, CurveTable<T> const& table, SpanLookup<T> const& lookup, T u)
	{
		std::size_t const span = lookup.find(curve.knots(), u);
		return curvePointFromCurveTable<T>(curve.degree(), curve.knots(), table.coefficients(), table.inverseSpanLengths(), span, u);
	}
}  // namespace geom
