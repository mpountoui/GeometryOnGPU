#include "viz/newton_plot.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace viz
{
	void addXAxis(SvgDocument& svg, double xMin, double xMax)
	{
		svg.addPolyline({{xMin, 0.0, 0.0}, {xMax, 0.0, 0.0}}, "#9a9a9a", 1.0, false);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void addNewtonPath(SvgDocument& svg, std::vector<geom::Point3<double>> const& path)
	{
		for (std::size_t i = 0; i + 1 < path.size(); ++i)
		{
			geom::Point3<double> const crossing{path[i + 1].x, 0.0, 0.0};
			svg.addPolyline({path[i], crossing}, "#e36209", 1.5, false);
			svg.addPolyline({crossing, path[i + 1]}, "#9a9a9a", 1.0, true);
		}
		if (path.empty())
		{
			return;
		}
		svg.addDots(std::vector<geom::Point3<double>>(path.begin(), path.end() - 1), "#d1242f", 4.0);
		svg.addDots({path.back()}, "#2da44e", 5.0);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	std::pair<double, double> detail::plotRange(std::vector<geom::Point3<double>> const& path, double a, double b)
	{
		double xMin = std::min(a, b);
		double xMax = std::max(a, b);
		for (geom::Point3<double> const& p : path)
		{
			xMin = std::min(xMin, p.x);
			xMax = std::max(xMax, p.x);
		}
		return {xMin, xMax};
	}
}  // namespace viz
