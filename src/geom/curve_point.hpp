#pragma once

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

#include "geom/basis.hpp"
#include "geom/point.hpp"

namespace geom
{
	// Point on a NURBS curve, The NURBS Book A4.1 CurvePoint (p.124). See
	// docs/algorithms/curve-point.md.
	//
	// No allocation and no exceptions: the basis values live in a local array of kMaxDegree + 1.
	// It does not depend on NurbsCurve (which includes this header): use curve(u) for a curve.
	
	// C(u) from the degree, the knots and the weighted control points Pw.
	// Preconditions: degree <= kMaxDegree, and u in the domain (see findSpan).
	template <std::floating_point T>
	GEOM_HD Point3<T> curvePoint(int degree, std::type_identity_t<std::span<T const>> knots, std::type_identity_t<std::span<Point4<T> const>> weightedPoints, T u);
	
	namespace detail
	{
		// Cw = sum_j basis[j] * Pw[first + j], the 4D point before the projection C = Cw / w.
		template <std::floating_point T>
		GEOM_HD Point4<T> weightedSum(std::span<T const> basis, std::span<Point4<T> const> weightedPoints, std::size_t first);
	}  // namespace detail
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	GEOM_HD Point3<T> curvePoint(int degree, std::type_identity_t<std::span<T const>> knots, std::type_identity_t<std::span<Point4<T> const>> weightedPoints, T u)
	{
		assert(degree >= 1 && degree <= kMaxDegree);
		assert(weightedPoints.size() + static_cast<std::size_t>(degree) + 1 == knots.size());
		auto const p = static_cast<std::size_t>(degree);
		std::size_t const span = findSpan(degree, knots, u);
		std::array<T, kMaxDegree + 1> storage{};
		std::span<T> const basis(storage.data(), p + 1);
		basisFunctions(span, u, degree, knots, basis);
		return Get3DPoint(detail::weightedSum<T>(basis, weightedPoints, span - p));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	GEOM_HD Point4<T> detail::weightedSum(std::span<T const> basis, std::span<Point4<T> const> weightedPoints, std::size_t first)
	{
		Point4<T> sum{};
		for (std::size_t j = 0; j < basis.size(); ++j)
		{
			Point4<T> const& pw = weightedPoints[first + j];
			sum.x += basis[j] * pw.x;
			sum.y += basis[j] * pw.y;
			sum.z += basis[j] * pw.z;
			sum.w += basis[j] * pw.w;
		}
		return sum;
	}
}  // namespace geom
