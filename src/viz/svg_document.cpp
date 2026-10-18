#include "viz/svg_document.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace viz
{
	namespace
	{
		// The smallest box around every point of every shape; {0, 0, 1, 1} if there are none.
		struct Box
		{
			double minX{};
			double minY{};
			double maxX{};
			double maxY{};
		};
		
		Box boundingBox(const std::vector<SvgShape>& shapes);
	}  // namespace
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	SvgDocument::SvgDocument(int width, int height)
		: m_width(width),
		m_height(height)
	{
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void SvgDocument::addPolyline(std::vector<geom::Point3<double>> points, std::string color, double strokeWidth, bool dashed)
	{
		m_shapes.push_back({SvgShape::Kind::Polyline, std::move(points), std::move(color), strokeWidth, dashed});
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void SvgDocument::addDots(std::vector<geom::Point3<double>> points, std::string color, double radius)
	{
		m_shapes.push_back({SvgShape::Kind::Dots, std::move(points), std::move(color), radius, false});
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	std::string SvgDocument::toString() const
	{
		const Viewport viewport = fitViewport();
		std::ostringstream out{};
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
	
	void SvgDocument::save(const std::filesystem::path& path) const
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
	SvgDocument::Viewport SvgDocument::fitViewport() const
	{
		const Box box = boundingBox(m_shapes);
		const double spanX = std::max(box.maxX - box.minX, 1e-12);
		const double spanY = std::max(box.maxY - box.minY, 1e-12);
		const double scale = std::min((m_width - 2 * kMargin) / spanX, (m_height - 2 * kMargin) / spanY);
		const double offsetX = (m_width - spanX * scale) / 2;
		const double offsetY = (m_height - spanY * scale) / 2;
		return {box.minX, box.minY, scale, offsetX, offsetY, static_cast<double>(m_height)};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// SVG's y axis points down, so y is flipped to make it point up.
	std::pair<double, double> SvgDocument::toPixel(const Viewport& viewport, const geom::Point3<double>& p)
	{
		const double x = viewport.offsetX + (p.x - viewport.minX) * viewport.scale;
		const double y = viewport.height - (viewport.offsetY + (p.y - viewport.minY) * viewport.scale);
		return {x, y};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void SvgDocument::writeShape(std::ostream& out, const SvgShape& shape, const Viewport& viewport)
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
	
	void SvgDocument::writePolyline(std::ostream& out, const SvgShape& shape, const Viewport& viewport)
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
	
	void SvgDocument::writeDots(std::ostream& out, const SvgShape& shape, const Viewport& viewport)
	{
		for (const geom::Point3<double>& p : shape.points)
		{
			const auto [x, y] = toPixel(viewport, p);
			out << "<circle cx=\"" << x << "\" cy=\"" << y << "\" r=\"" << shape.size << "\" fill=\"" << shape.color << "\"/>\n";
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	namespace
	{
		Box boundingBox(const std::vector<SvgShape>& shapes)
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
	}  // namespace
}  // namespace viz
