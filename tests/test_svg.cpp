#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "geom/nurbs_curve.hpp"
#include "viz/svg_document.hpp"

using geom::Point3;
using viz::SvgDocument;

namespace
{
	using Points = std::vector<Point3<double>>;
	using Pixels = std::vector<std::pair<double, double>>;
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	std::size_t countOccurrences(std::string const& text, std::string const& needle)
	{
		std::size_t count = 0;
		for (std::size_t pos = text.find(needle); pos != std::string::npos; pos = text.find(needle, pos + 1))
		{
			++count;
		}
		return count;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The pixel coordinates in the points="..." attribute of the first polyline.
	Pixels firstPolylinePixels(std::string const& svg)
	{
		std::size_t const start = svg.find("points=\"") + 8;
		std::istringstream in(svg.substr(start, svg.find('"', start) - start));
		Pixels pixels{};
		double x = 0;
		double y = 0;
		char comma = 0;
		while (in >> x >> comma >> y)
		{
			pixels.push_back({x, y});
		}
		return pixels;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testEmptyDocument()
	{
		std::string const svg = SvgDocument(400, 300).toString();
		CHECK(svg.starts_with("<svg"));
		CHECK(svg.ends_with("</svg>\n"));
		CHECK(countOccurrences(svg, "<polyline") == 0);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testShapesAreWritten()
	{
		SvgDocument doc(400, 300);
		doc.addPolyline({{0, 0, 0}, {1, 1, 0}, {2, 0, 0}}, "black", 1.0, true);
		doc.addDots({{0, 0, 0}, {1, 1, 0}, {2, 0, 0}}, "red", 3.0);
		std::string const svg = doc.toString();
		CHECK(countOccurrences(svg, "<polyline") == 1);
		CHECK(countOccurrences(svg, "<circle") == 3);
		CHECK(countOccurrences(svg, "stroke-dasharray") == 1);
		CHECK(firstPolylinePixels(svg).size() == 3);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// World y points up: a point with larger y gets a smaller pixel y. Everything stays inside.
	void testYAxisPointsUpAndFits()
	{
		SvgDocument doc(400, 300);
		doc.addPolyline({{0, 0, 0}, {1, 1, 0}}, "black", 1.0, false);
		Pixels const pixels = firstPolylinePixels(doc.toString());
		CHECK(pixels.size() == 2);
		CHECK(pixels[1].second < pixels[0].second);
		for (auto const& [x, y] : pixels)
		{
			CHECK(x >= 0 && x <= 400 && y >= 0 && y <= 300);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The aspect ratio is kept: a unit square is drawn as a square.
	void testAspectRatio()
	{
		SvgDocument doc(400, 300);
		doc.addPolyline({{0, 0, 0}, {1, 0, 0}, {1, 1, 0}}, "black", 1.0, false);
		Pixels const pixels = firstPolylinePixels(doc.toString());
		double const width = pixels[1].first - pixels[0].first;
		double const height = pixels[1].second - pixels[2].second;
		CHECK(test::near(width, height, 0.02));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A curve with its control polygon: dashed polygon, sampled curve, one dot per control point.
	void testNurbsCurve()
	{
		Points const points{{1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
		geom::NurbsCurve<double> const curve(2, {0, 0, 0, 1, 1, 1}, points);
		SvgDocument doc(400, 300);
		viz::addNurbsCurve(doc, curve, "blue", true);
		std::string const svg = doc.toString();
		CHECK(countOccurrences(svg, "<polyline") == 2);
		CHECK(countOccurrences(svg, "<circle") == 3);
		CHECK(viz::sampleCurve(curve, 10).size() == 11);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void testSave()
	{
		std::filesystem::path const path = std::filesystem::temp_directory_path() / "geometry_on_gpu_test.svg";
		SvgDocument(200, 100).save(path);
		std::ifstream file(path);
		std::string firstLine{};
		std::getline(file, firstLine);
		CHECK(firstLine.starts_with("<svg"));
		std::filesystem::remove(path);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void runAllTests()
	{
		testEmptyDocument();
		testShapesAreWritten();
		testYAxisPointsUpAndFits();
		testAspectRatio();
		testNurbsCurve();
		testSave();
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main()
{
	runAllTests();
	return test::report("svg");
}
