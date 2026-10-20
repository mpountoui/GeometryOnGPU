#pragma once

#include <algorithm>
#include <concepts>
#include <utility>
#include <vector>

#include "geom/newton_raphson.hpp"
#include "geom/point.hpp"
#include "viz/svg_document.hpp"

namespace viz
{
	// Draws Newton-Raphson for f(x) = 0 as an SVG plot: f on [a, b] (widened to include every
	// iterate), the x-axis, and for every step the tangent at (x_i, f(x_i)) down to the x-axis at
	// x_{i+1}, then a dashed line up to (x_{i+1}, f(x_{i+1})). Iterates are red dots, the final root
	// green. See docs/algorithms/newton-raphson.md. Host-only.
	
	// Runs newtonRaphson(f, df, x0, options), draws it and returns its result.
	// Switches svg to separate x and y scales (setKeepAspectRatio(false)).
	template <std::floating_point T, typename F, typename DF>
	geom::NewtonResult<T> addNewtonPlot(SvgDocument& svg, F f, DF df, T x0, T a, T b, geom::NewtonOptions<T> options = {});
	
	// The x-axis from xMin to xMax, as a thin grey line.
	void addXAxis(SvgDocument& svg, double xMin, double xMax);
	
	// The Newton path through the points (x_i, f(x_i)) in order: tangents, dashed verticals,
	// red dots, and a green dot on the last point.
	void addNewtonPath(SvgDocument& svg, std::vector<geom::Point3<double>> const& path);
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	namespace detail
	{
		// (x, f(x)) at samples + 1 evenly spaced x in [a, b].
		template <std::floating_point T, typename F>
		std::vector<geom::Point3<double>> sampleFunction(F& f, T a, T b, int samples);
		
		// [min, max] of a, b and the x of every path point.
		std::pair<double, double> plotRange(std::vector<geom::Point3<double>> const& path, double a, double b);
	}  // namespace detail
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T, typename F, typename DF>
	geom::NewtonResult<T> addNewtonPlot(SvgDocument& svg, F f, DF df, T x0, T a, T b, geom::NewtonOptions<T> options)
	{
		std::vector<geom::Point3<double>> path{};
		auto const record = [&path](int, T x, T fx, T) { path.push_back({static_cast<double>(x), static_cast<double>(fx), 0.0}); };
		geom::NewtonResult<T> const result = geom::newtonRaphson(f, df, x0, options, record);
		if (path.empty() || path.back().x != static_cast<double>(result.root))
		{
			path.push_back({static_cast<double>(result.root), static_cast<double>(result.value), 0.0});
		}
		auto const [xMin, xMax] = detail::plotRange(path, static_cast<double>(a), static_cast<double>(b));
		svg.setKeepAspectRatio(false);
		addXAxis(svg, xMin, xMax);
		svg.addPolyline(detail::sampleFunction(f, static_cast<T>(xMin), static_cast<T>(xMax), 400), "#1f6feb", 2.0, false);
		addNewtonPath(svg, path);
		return result;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T, typename F>
	std::vector<geom::Point3<double>> detail::sampleFunction(F& f, T a, T b, int samples)
	{
		std::vector<geom::Point3<double>> points{};
		points.reserve(static_cast<std::size_t>(samples) + 1);
		for (int k = 0; k <= samples; ++k)
		{
			T const x = (k == samples) ? b : a + (b - a) * static_cast<T>(k) / static_cast<T>(samples);
			points.push_back({static_cast<double>(x), static_cast<double>(f(x)), 0.0});
		}
		return points;
	}
}  // namespace viz
