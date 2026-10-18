#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

#include "geom/nurbs_curve.hpp"
#include "geom/point.hpp"

namespace viz
{
	// Writes 2D drawings as SVG images, for looking at curves. Only x and y of each point are
	// drawn (the projection onto the xy plane). The drawing is scaled to fit the image, keeps its
	// aspect ratio and has y pointing up, like the book's figures.
	//
	// Host-only (std::string, files); never include this from .cu files. SvgDocument is compiled
	// in svg_document.cpp (the viz library); the templated helpers stay here.
	
	// One shape to draw, in world coordinates.
	struct SvgShape
	{
		enum class Kind
		{
			Polyline,
			Dots
		};
		
		Kind kind{};
		std::vector<geom::Point3<double>> points{};
		std::string color{};
		double size{};    // stroke width for a polyline, radius for dots
		bool dashed{};
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	class SvgDocument
	{
	public:
		SvgDocument(int width, int height);
		
	public:
		void addPolyline(std::vector<geom::Point3<double>> points, std::string color, double strokeWidth, bool dashed);
		void addDots(std::vector<geom::Point3<double>> points, std::string color, double radius);
		std::string toString() const;
		
		// Throws std::runtime_error if the file cannot be written.
		void save(std::filesystem::path const& path) const;
		
	private:
		struct Viewport
		{
			double minX{};
			double minY{};
			double scale{};
			double offsetX{};
			double offsetY{};
			double height{};
		};
		
		Viewport fitViewport() const;
		
	private:
		static constexpr double kMargin = 20.0;
		
		int m_width{};
		int m_height{};
		std::vector<SvgShape> m_shapes{};
		
	private:
		static std::pair<double, double> toPixel(Viewport const& viewport, geom::Point3<double> const& p);
		static void writeShape(std::ostream& out, SvgShape const& shape, Viewport const& viewport);
		static void writePolyline(std::ostream& out, SvgShape const& shape, Viewport const& viewport);
		static void writeDots(std::ostream& out, SvgShape const& shape, Viewport const& viewport);
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// C(u) at samples + 1 evenly spaced parameters across the curve's domain.
	template <std::floating_point T>
	std::vector<geom::Point3<double>> sampleCurve(geom::NurbsCurve<T> const& curve, int samples);
	
	// The Euclidean control points P_0 .. P_n, in order.
	template <std::floating_point T>
	std::vector<geom::Point3<double>> controlPolygon(geom::NurbsCurve<T> const& curve);
	
	// The curve as a solid line in `color`. With withControlPolygon, also its dashed control
	// polygon and its control points as red dots.
	template <std::floating_point T>
	void addNurbsCurve(SvgDocument& svg, geom::NurbsCurve<T> const& curve, std::string const& color, bool withControlPolygon);
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	namespace detail
	{
		template <std::floating_point T>
		geom::Point3<double> toDouble(geom::Point3<T> const& p);
	}  // namespace detail
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::vector<geom::Point3<double>> sampleCurve(geom::NurbsCurve<T> const& curve, int samples)
	{
		auto const [a, b] = curve.domain();
		std::vector<geom::Point3<double>> points{};
		points.reserve(static_cast<std::size_t>(samples) + 1);
		for (int k = 0; k <= samples; ++k)
		{
			// The last sample is exactly b, so rounding can never push u past the domain.
			T const u = (k == samples) ? b : a + (b - a) * static_cast<T>(k) / static_cast<T>(samples);
			points.push_back(detail::toDouble(curve(u)));
		}
		return points;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::vector<geom::Point3<double>> controlPolygon(geom::NurbsCurve<T> const& curve)
	{
		std::vector<geom::Point3<double>> points{};
		points.reserve(curve.numControlPoints());
		for (std::size_t i = 0; i < curve.numControlPoints(); ++i)
		{
			points.push_back(detail::toDouble(curve.controlPoint(i)));
		}
		return points;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void addNurbsCurve(SvgDocument& svg, geom::NurbsCurve<T> const& curve, std::string const& color, bool withControlPolygon)
	{
		std::vector<geom::Point3<double>> const polygon = controlPolygon(curve);
		if (withControlPolygon)
		{
			svg.addPolyline(polygon, "#9a9a9a", 1.0, true);
		}
		// About 20 samples per control point (at least 200), so long curves stay smooth.
		int const samples = std::max(200, 20 * static_cast<int>(curve.numControlPoints()));
		svg.addPolyline(sampleCurve(curve, samples), color, 2.0, false);
		if (withControlPolygon)
		{
			svg.addDots(polygon, "#d1242f", 3.5);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	geom::Point3<double> detail::toDouble(geom::Point3<T> const& p)
	{
		return {static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z)};
	}
}  // namespace viz
