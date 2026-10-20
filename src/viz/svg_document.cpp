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
		
		Box boundingBox(std::vector<SvgShape> const& shapes);
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
	
	void SvgDocument::setKeepAspectRatio(bool keep)
	{
		m_keepAspectRatio = keep;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	std::string SvgDocument::toString() const
	{
		Viewport const viewport = fitViewport();
		std::ostringstream out{};
		out << std::fixed << std::setprecision(2);
		out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << m_width << "\" height=\"" << m_height << "\" viewBox=\"0 0 " << m_width << ' ' << m_height << "\">\n";
		out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
		for (SvgShape const& shape : m_shapes)
		{
			writeShape(out, shape, viewport);
		}
		out << "</svg>\n";
		return out.str();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void SvgDocument::save(std::filesystem::path const& path) const
	{
		std::ofstream file(path);
		if (!file)
		{
			throw std::runtime_error("SvgDocument: cannot open " + path.string());
		}
		file << toString();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Scales so the bounding box fits inside the margins (the same scale for x and y if the aspect
	// ratio is kept), centred in the image.
	SvgDocument::Viewport SvgDocument::fitViewport() const
	{
		Box const box = boundingBox(m_shapes);
		double const spanX = std::max(box.maxX - box.minX, 1e-12);
		double const spanY = std::max(box.maxY - box.minY, 1e-12);
		double scaleX = (m_width - 2 * kMargin) / spanX;
		double scaleY = (m_height - 2 * kMargin) / spanY;
		if (m_keepAspectRatio)
		{
			scaleX = std::min(scaleX, scaleY);
			scaleY = scaleX;
		}
		double const offsetX = (m_width - spanX * scaleX) / 2;
		double const offsetY = (m_height - spanY * scaleY) / 2;
		return {box.minX, box.minY, scaleX, scaleY, offsetX, offsetY, static_cast<double>(m_height)};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// SVG's y axis points down, so y is flipped to make it point up.
	std::pair<double, double> SvgDocument::toPixel(Viewport const& viewport, geom::Point3<double> const& p)
	{
		double const x = viewport.offsetX + (p.x - viewport.minX) * viewport.scaleX;
		double const y = viewport.height - (viewport.offsetY + (p.y - viewport.minY) * viewport.scaleY);
		return {x, y};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void SvgDocument::writeShape(std::ostream& out, SvgShape const& shape, Viewport const& viewport)
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
	
	void SvgDocument::writePolyline(std::ostream& out, SvgShape const& shape, Viewport const& viewport)
	{
		out << "<polyline fill=\"none\" stroke=\"" << shape.color << "\" stroke-width=\"" << shape.size << '"';
		if (shape.dashed)
		{
			out << " stroke-dasharray=\"6 4\"";
		}
		out << " points=\"";
		for (geom::Point3<double> const& p : shape.points)
		{
			auto const [x, y] = toPixel(viewport, p);
			out << x << ',' << y << ' ';
		}
		out << "\"/>\n";
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void SvgDocument::writeDots(std::ostream& out, SvgShape const& shape, Viewport const& viewport)
	{
		for (geom::Point3<double> const& p : shape.points)
		{
			auto const [x, y] = toPixel(viewport, p);
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
		Box boundingBox(std::vector<SvgShape> const& shapes)
		{
			constexpr double inf = std::numeric_limits<double>::infinity();
			Box box{inf, inf, -inf, -inf};
			for (SvgShape const& shape : shapes)
			{
				for (geom::Point3<double> const& p : shape.points)
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
