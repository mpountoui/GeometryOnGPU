#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "check.hpp"
#include "geom/nurbs_curve.hpp"
#include "geom/point.hpp"

// Test curves and point comparison shared by the table tests.

namespace test
{
	// |a - b| <= tolerance in every coordinate.
	template <typename T>
	bool nearPoint(geom::Point3<T> const& a, geom::Point3<T> const& b, T tolerance);
	
	// Points on a widening spiral that rises in z, for any number of control points.
	template <typename T>
	std::vector<geom::Point3<T>> spiralPoints(std::size_t numPoints);
	
	// Weights between 0.5 and 1.5, so the curve is rational.
	template <typename T>
	std::vector<T> wavyWeights(std::size_t numPoints);
	
	// A rational spiral curve on the given knots (knots.size() - degree - 1 control points).
	template <typename T>
	geom::NurbsCurve<T> rationalCurve(int degree, std::vector<T> const& knots);
	
	// Clamped knots of the degree with numInterior evenly spaced interior knots in (0, 1).
	template <typename T>
	std::vector<T> uniformKnots(int degree, int numInterior);
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <typename T>
	bool nearPoint(geom::Point3<T> const& a, geom::Point3<T> const& b, T tolerance)
	{
		return near(a.x, b.x, tolerance) && near(a.y, b.y, tolerance) && near(a.z, b.z, tolerance);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <typename T>
	std::vector<geom::Point3<T>> spiralPoints(std::size_t numPoints)
	{
		std::vector<geom::Point3<T>> points{};
		for (std::size_t i = 0; i < numPoints; ++i)
		{
			double const a = 0.4 * static_cast<double>(i);
			points.push_back({static_cast<T>(std::cos(a) * (1.0 + 0.1 * a)), static_cast<T>(std::sin(a) * (1.0 + 0.1 * a)), static_cast<T>(0.05 * a)});
		}
		return points;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <typename T>
	std::vector<T> wavyWeights(std::size_t numPoints)
	{
		std::vector<T> weights{};
		for (std::size_t i = 0; i < numPoints; ++i)
		{
			weights.push_back(static_cast<T>(1.0 + 0.5 * std::sin(0.7 * static_cast<double>(i))));
		}
		return weights;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <typename T>
	geom::NurbsCurve<T> rationalCurve(int degree, std::vector<T> const& knots)
	{
		std::size_t const numPoints = knots.size() - static_cast<std::size_t>(degree) - 1;
		return geom::NurbsCurve<T>(degree, knots, spiralPoints<T>(numPoints), wavyWeights<T>(numPoints));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <typename T>
	std::vector<T> uniformKnots(int degree, int numInterior)
	{
		std::vector<T> knots(static_cast<std::size_t>(degree) + 1, T{0});
		for (int k = 1; k <= numInterior; ++k)
		{
			knots.push_back(static_cast<T>(k) / static_cast<T>(numInterior + 1));
		}
		knots.insert(knots.end(), static_cast<std::size_t>(degree) + 1, T{1});
		return knots;
	}
}  // namespace test
