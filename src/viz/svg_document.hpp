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
		void save(const std::filesystem::path& path) const;
		
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
		static std::pair<double, double> toPixel(const Viewport& viewport, const geom::Point3<double>& p);
		static void writeShape(std::ostream& out, const SvgShape& shape, const Viewport& viewport);
		static void writePolyline(std::ostream& out, const SvgShape& shape, const Viewport& viewport);
		static void writeDots(std::ostream& out, const SvgShape& shape, const Viewport& viewport);
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// C(u) at samples + 1 evenly spaced parameters across the curve's domain.
	template <std::floating_point T>
	std::vector<geom::Point3<double>> sampleCurve(const geom::NurbsCurve<T>& curve, int samples);
	
	// The Euclidean control points P_0 .. P_n, in order.
	template <std::floating_point T>
	std::vector<geom::Point3<double>> controlPolygon(const geom::NurbsCurve<T>& curve);
	
	// The curve as a solid line in `color`. With withControlPolygon, also its dashed control
	// polygon and its control points as red dots.
	template <std::floating_point T>
	void addNurbsCurve(SvgDocument& svg, const geom::NurbsCurve<T>& curve, const std::string& color, bool withControlPolygon);
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	namespace detail
	{
		template <std::floating_point T>
		geom::Point3<double> toDouble(const geom::Point3<T>& p);
	}  // namespace detail
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::vector<geom::Point3<double>> sampleCurve(const geom::NurbsCurve<T>& curve, int samples)
	{
		const auto [a, b] = curve.domain();
		std::vector<geom::Point3<double>> points{};
		points.reserve(static_cast<std::size_t>(samples) + 1);
		for (int k = 0; k <= samples; ++k)
		{
			// The last sample is exactly b, so rounding can never push u past the domain.
			const T u = (k == samples) ? b : a + (b - a) * static_cast<T>(k) / static_cast<T>(samples);
			points.push_back(detail::toDouble(curve(u)));
		}
		return points;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::vector<geom::Point3<double>> controlPolygon(const geom::NurbsCurve<T>& curve)
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
	void addNurbsCurve(SvgDocument& svg, const geom::NurbsCurve<T>& curve, const std::string& color, bool withControlPolygon)
	{
		const std::vector<geom::Point3<double>> polygon = controlPolygon(curve);
		if (withControlPolygon)
		{
			svg.addPolyline(polygon, "#9a9a9a", 1.0, true);
		}
		// About 20 samples per control point (at least 200), so long curves stay smooth.
		const int samples = std::max(200, 20 * static_cast<int>(curve.numControlPoints()));
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
	geom::Point3<double> detail::toDouble(const geom::Point3<T>& p)
	{
		return {static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z)};
	}
}  // namespace viz
