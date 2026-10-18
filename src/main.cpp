#include <exception>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <string_view>
#include <vector>

#include "geom/nurbs_curve.hpp"
#include "viz/svg_document.hpp"

#ifdef GEOM_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace
{
#ifdef GEOM_HAS_CUDA
	// Returns false (after printing the error) if a CUDA call failed.
	bool checkCuda(cudaError_t err)
	{
		if (err == cudaSuccess)
		{
			return true;
		}
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << '\n';
		return false;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	bool printCudaDevice(int index)
	{
		cudaDeviceProp prop{};
		if (!checkCuda(cudaGetDeviceProperties(&prop, index)))
		{
			return false;
		}
		std::cout << "CUDA device " << index << ": " << prop.name << " (sm_" << prop.major << prop.minor << ")\n";
		return true;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	bool printCudaDevices()
	{
		int deviceCount = 0;
		if (!checkCuda(cudaGetDeviceCount(&deviceCount)))
		{
			return false;
		}
		for (int i = 0; i < deviceCount; ++i)
		{
			if (!printCudaDevice(i))
			{
				return false;
			}
		}
		if (deviceCount == 0)
		{
			std::cout << "No CUDA devices found\n";
		}
		return true;
	}
#else
	bool printCudaDevices()
	{
		std::cout << "Built without CUDA\n";
		return true;
	}
#endif
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	using Curve  = geom::NurbsCurve<double>;
	using Points = std::vector<geom::Point3<double>>;
	using Values = std::vector<double>;
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The rational cubic of Figure 4.1 (The NURBS Book, p.119), with made-up control points.
	Curve figure41Curve()
	{
		const Points points{{0, 0, 0}, {1, 2, 0}, {3, 3, 0}, {4, 0, 0}, {6, 0, 0}, {7, 2, 0}, {9, 2, 0}};
		return Curve(3, {0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1}, points, Values{1, 1, 1, 3, 1, 1, 1});
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The quadratic of Ex4.1 (p.122) with weight w_1 changed, as in Figure 4.4.
	Curve example41Curve(double w1)
	{
		const Points points{{0, 0, 0}, {1, 1, 0}, {3, 2, 0}, {4, 1, 0}, {5, -1, 0}};
		return Curve(2, {0, 0, 0, 1, 2, 3, 3, 3}, points, Values{1, w1, 1, 1, 1});
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A quarter of the unit circle as a rational quadratic Bezier (book Chapter 7).
	Curve quarterCircle()
	{
		const Points points{{1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
		return Curve(2, {0, 0, 0, 1, 1, 1}, points, Values{1, std::numbers::sqrt2 / 2, 1});
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void writeSingleCurve(const std::filesystem::path& path, const Curve& curve)
	{
		viz::SvgDocument svg(800, 600);
		viz::addNurbsCurve(svg, curve, "#1f6feb", true);
		svg.save(path);
		std::cout << "wrote " << path.string() << '\n';
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Figure 4.4: one control polygon, w_1 = 0.3 (green), 1 (blue) and 4 (purple).
	void writeWeightComparison(const std::filesystem::path& path)
	{
		viz::SvgDocument svg(800, 600);
		viz::addNurbsCurve(svg, example41Curve(0.3), "#2da44e", true);
		viz::addNurbsCurve(svg, example41Curve(1.0), "#1f6feb", false);
		viz::addNurbsCurve(svg, example41Curve(4.0), "#8250df", false);
		svg.save(path);
		std::cout << "wrote " << path.string() << '\n';
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void writeDemoSvgs(const std::filesystem::path& directory)
	{
		std::filesystem::create_directories(directory);
		writeSingleCurve(directory / "figure41.svg", figure41Curve());
		writeSingleCurve(directory / "quarter_circle.svg", quarterCircle());
		writeWeightComparison(directory / "weights.svg");
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// --svg [directory]: writes the demo SVGs, by default to an "svg" folder next to the executable.
	int runSvgDemo(int argc, char* argv[])
	{
		const std::filesystem::path directory = (argc >= 3) ? std::filesystem::path(argv[2]) : std::filesystem::path(argv[0]).parent_path() / "svg";
		try
		{
			writeDemoSvgs(directory);
			return 0;
		}
		catch (const std::exception& error)
		{
			std::cerr << "error: " << error.what() << '\n';
			return 1;
		}
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main(int argc, char* argv[])
{
	std::cout << "GeometryOnGPU\n";
	if (argc >= 2 && std::string_view(argv[1]) == "--svg")
	{
		return runSvgDemo(argc, argv);
	}
	return printCudaDevices() ? 0 : 1;
}
