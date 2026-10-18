#pragma once

#include <concepts>

#include "geom/host_device.hpp"

namespace geom
{
	// Euclidean point P = (x, y, z); the book's POINT (Section 13.3, p.597).
	template <std::floating_point T>
	struct Point3
	{
		T x{}, y{}, z{};
		
		friend bool operator==(Point3 const&, Point3 const&) = default;
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Weighted (homogeneous) control point Pw = (w*x, w*y, w*z, w); the book's CPOINT (p.597).
	// Aligned to its full size so that Point4<float> has the layout of CUDA's float4 and can be
	// loaded on the GPU in a single 16-byte transaction.
	template <std::floating_point T>
	struct alignas(4 * sizeof(T)) Point4
	{
		T x{}, y{}, z{}, w{};
		
		friend bool operator==(Point4 const&, Point4 const&) = default;
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Projection C = Cw / w (the book's notation, p.124): the Euclidean point of a weighted point.
	template <std::floating_point T>
	GEOM_HD Point3<T> Get3DPoint(Point4<T> const& pw)
	{
		return {pw.x / pw.w, pw.y / pw.w, pw.z / pw.w};
	}
}  // namespace geom
