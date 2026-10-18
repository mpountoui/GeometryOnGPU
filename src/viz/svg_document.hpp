#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "geom/curve_point.hpp"
#include "geom/nurbs_curve.hpp"
#include "geom/point.hpp"

namespace viz
{
	// Writes 2D drawings as SVG images, for looking at curves. Only x and y of each point are
	// drawn (the projection onto the xy plane). The drawing is scaled to fit the image, keeps its
	// aspect ratio and has y pointing up, like the book's figures.
	//
	// Host-only (std::string, files); never include this from .cu files.
	
	// One shape to draw, in world coordinates.
	struct SvgShape
	{
		enum class Kind
		{
			Polyline,
			Dots
		};
		
		Kind kind;
		std::vector<geom::Point3<double>> points;
		std::string color;
		double size;    // stroke width for a polyline, radius for dots
		bool dashed;
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
			double minX;
			double minY;
			double scale;
			double offsetX;
			double offsetY;
			double height;
		};
		
		Viewport fitViewport() const;
		static std::pair<double, double> toPixel(const Viewport& viewport, const geom::Point3<double>& p);
		static void writeShape(std::ostream& out, const SvgShape& shape, const Viewport& viewport);
		static void writePolyline(std::ostream& out, const SvgShape& shape, const Viewport& viewport);
		static void writeDots(std::ostream& out, const SvgShape& shape, const Viewport& viewport);
		
	private:
		static constexpr double kMargin = 20.0;
		
		int m_width;
		int m_height;
		std::vector<SvgShape> m_shapes;
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
		// The smallest box around every point of every shape; {0, 0, 1, 1} if there are none.
		struct Box
		{
			double minX;
			double minY;
			double maxX;
			double maxY;
		};
		
		inline Box boundingBox(const std::vector<SvgShape>& shapes);
		
		template <std::floating_point T>
		geom::Point3<double> toDouble(const geom::Point3<T>& p);
	}  // namespace detail
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	inline SvgDocument::SvgDocument(int width, int height)
		: m_width(width),
		m_height(height)
	{
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	inline void SvgDocument::addPolyline(std::vector<geom::Point3<double>> points, std::string color, double strokeWidth, bool dashed)
	{
		m_shapes.push_back({SvgShape::Kind::Polyline, std::move(points), std::move(color), strokeWidth, dashed});
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	inline void SvgDocument::addDots(std::vector<geom::Point3<double>> points, std::string color, double radius)
	{
		m_shapes.push_back({SvgShape::Kind::Dots, std::move(points), std::move(color), radius, false});
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	inline std::string SvgDocument::toString() const
	{
		const Viewport viewport = fitViewport();
		std::ostringstream out;
		out << std::fixed << std::setprecision(2);
		out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << m_width << "\" height=\"" << m_height << "\" viewBox=\"0 0 " << m_width << ' ' << m_height << "\">\n";
		out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
		for (const SvgShape& shape : m_shapes)
		{
			writeShape(out, shape, viewport);
		}
		out << "</svg>\n";
		return out.str();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	inline void SvgDocument::save(const std::filesystem::path& path) const
	{
		std::ofstream file(path);
		if (!file)
		{
			throw std::runtime_error("SvgDocument: cannot open " + path.string());
		}
		file << toString();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Uniform scale so the bounding box fits inside the margins, centred in the image.
	inline SvgDocument::Viewport SvgDocument::fitViewport() const
	{
		const detail::Box box = detail::boundingBox(m_shapes);
		const double spanX = std::max(box.maxX - box.minX, 1e-12);
		const double spanY = std::max(box.maxY - box.minY, 1e-12);
		const double scale = std::min((m_width - 2 * kMargin) / spanX, (m_height - 2 * kMargin) / spanY);
		const double offsetX = (m_width - spanX * scale) / 2;
		const double offsetY = (m_height - spanY * scale) / 2;
		return {box.minX, box.minY, scale, offsetX, offsetY, static_cast<double>(m_height)};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// SVG's y axis points down, so y is flipped to make it point up.
	inline std::pair<double, double> SvgDocument::toPixel(const Viewport& viewport, const geom::Point3<double>& p)
	{
		const double x = viewport.offsetX + (p.x - viewport.minX) * viewport.scale;
		const double y = viewport.height - (viewport.offsetY + (p.y - viewport.minY) * viewport.scale);
		return {x, y};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	inline void SvgDocument::writeShape(std::ostream& out, const SvgShape& shape, const Viewport& viewport)
	{
		if (shape.kind == SvgShape::Kind::Polyline)
		{
			writePolyline(out, shape, viewport);
		}
		else
		{
			writeDots(out, shape, viewport);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	inline void SvgDocument::writePolyline(std::ostream& out, const SvgShape& shape, const Viewport& viewport)
	{
		out << "<polyline fill=\"none\" stroke=\"" << shape.color << "\" stroke-width=\"" << shape.size << '"';
		if (shape.dashed)
		{
			out << " stroke-dasharray=\"6 4\"";
		}
		out << " points=\"";
		for (const geom::Point3<double>& p : shape.points)
		{
			const auto [x, y] = toPixel(viewport, p);
			out << x << ',' << y << ' ';
		}
		out << "\"/>\n";
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	inline void SvgDocument::writeDots(std::ostream& out, const SvgShape& shape, const Viewport& viewport)
	{
		for (const geom::Point3<double>& p : shape.points)
		{
			const auto [x, y] = toPixel(viewport, p);
			out << "<circle cx=\"" << x << "\" cy=\"" << y << "\" r=\"" << shape.size << "\" fill=\"" << shape.color << "\"/>\n";
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::vector<geom::Point3<double>> sampleCurve(const geom::NurbsCurve<T>& curve, int samples)
	{
		const auto [a, b] = curve.domain();
		std::vector<geom::Point3<double>> points;
		points.reserve(static_cast<std::size_t>(samples) + 1);
		for (int k = 0; k <= samples; ++k)
		{
			// The last sample is exactly b, so rounding can never push u past the domain.
			const T u = (k == samples) ? b : a + (b - a) * static_cast<T>(k) / static_cast<T>(samples);
			points.push_back(detail::toDouble(geom::curvePoint(curve, u)));
		}
		return points;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::vector<geom::Point3<double>> controlPolygon(const geom::NurbsCurve<T>& curve)
	{
		std::vector<geom::Point3<double>> points;
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
		svg.addPolyline(sampleCurve(curve, 200), color, 2.0, false);
		if (withControlPolygon)
		{
			svg.addDots(polygon, "#d1242f", 3.5);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	inline detail::Box detail::boundingBox(const std::vector<SvgShape>& shapes)
	{
		constexpr double inf = std::numeric_limits<double>::infinity();
		Box box{inf, inf, -inf, -inf};
		for (const SvgShape& shape : shapes)
		{
			for (const geom::Point3<double>& p : shape.points)
			{
				box = {std::min(box.minX, p.x), std::min(box.minY, p.y), std::max(box.maxX, p.x), std::max(box.maxY, p.y)};
			}
		}
		if (box.minX > box.maxX)
		{
			return {0.0, 0.0, 1.0, 1.0};
		}
		return box;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	geom::Point3<double> detail::toDouble(const geom::Point3<T>& p)
	{
		return {static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z)};
	}
}  // namespace viz
