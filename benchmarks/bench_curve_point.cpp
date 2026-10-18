#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <print>
#include <span>
#include <string_view>
#include <system_error>
#include <vector>

#include "geom/nurbs_curve.hpp"
#include "viz/svg_document.hpp"

// Benchmark: evaluate N points C(u) on a NURBS curve on the CPU (one thread) and time it.
//
//   bench_curve_point [N] [degree] [--svg]      defaults: N = 1000000, degree = 3
//
// --svg also saves a picture of the benchmark curve (xy projection) to svg/benchmark_curve.svg
// next to the executable.
//
// Only the evaluation loop is timed: the curve, the parameters and the output array are set up
// before. One warm-up run, then the best and the median of kRuns runs. The checksum is printed
// so the compiler cannot skip the work, and it lets float and double results be compared.

namespace
{
	using Clock = std::chrono::steady_clock;
	
	constexpr std::size_t kNumControlPoints = 100;
	constexpr int kRuns = 5;
	
	struct Timing
	{
		double bestSeconds{};
		double medianSeconds{};
		double checksum{};
	};
	
	struct Options
	{
		std::size_t count{1'000'000};
		std::size_t degree{3};
		bool writeSvg{false};
		bool valid{true};
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Uniform clamped knots: degree + 1 zeros, evenly spaced interior knots, degree + 1 ones.
	template <std::floating_point T>
	std::vector<T> uniformClampedKnots(int degree, std::size_t numPoints)
	{
		const auto p = static_cast<std::size_t>(degree);
		const std::size_t numInterior = numPoints - p - 1;
		std::vector<T> knots(p + 1, T{0});
		for (std::size_t k = 1; k <= numInterior; ++k)
		{
			knots.push_back(static_cast<T>(k) / static_cast<T>(numInterior + 1));
		}
		knots.insert(knots.end(), p + 1, T{1});
		return knots;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A rational 3D spiral: control points on a widening spiral in xy that rises in z (about six
	// turns), weights between 0.5 and 1.5.
	template <std::floating_point T>
	geom::NurbsCurve<T> makeBenchmarkCurve(int degree)
	{
		std::vector<geom::Point3<T>> points{};
		std::vector<T> weights{};
		for (std::size_t i = 0; i < kNumControlPoints; ++i)
		{
			const double t = static_cast<double>(i);
			const double radius = 1.0 + 0.08 * t;
			points.push_back({static_cast<T>(radius * std::cos(0.4 * t)), static_cast<T>(radius * std::sin(0.4 * t)), static_cast<T>(0.05 * t)});
			weights.push_back(static_cast<T>(1.0 + 0.5 * std::sin(0.7 * t)));
		}
		return geom::NurbsCurve<T>(degree, uniformClampedKnots<T>(degree, kNumControlPoints), points, weights);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// N parameters evenly spaced over [a, b]; the last one is exactly b.
	template <std::floating_point T>
	std::vector<T> evenlySpacedParameters(const geom::NurbsCurve<T>& curve, std::size_t count)
	{
		const auto [a, b] = curve.domain();
		std::vector<T> params(count, b);
		for (std::size_t k = 0; k + 1 < count; ++k)
		{
			params[k] = a + (b - a) * static_cast<T>(k) / static_cast<T>(count - 1);
		}
		return params;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The timed loop: out[k] = C(params[k]). Returns the elapsed time in seconds.
	template <std::floating_point T>
	double evaluateAll(const geom::NurbsCurve<T>& curve, std::span<const T> params, std::span<geom::Point3<T>> out)
	{
		const Clock::time_point start = Clock::now();
		for (std::size_t k = 0; k < params.size(); ++k)
		{
			out[k] = curve(params[k]);
		}
		const Clock::time_point stop = Clock::now();
		return std::chrono::duration<double>(stop - start).count();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	double checksum(std::span<const geom::Point3<T>> points)
	{
		double sum = 0.0;
		for (const geom::Point3<T>& p : points)
		{
			sum += static_cast<double>(p.x) + static_cast<double>(p.y) + static_cast<double>(p.z);
		}
		return sum;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One warm-up run, then kRuns timed runs.
	template <std::floating_point T>
	Timing timeEvaluation(int degree, std::size_t count)
	{
		const geom::NurbsCurve<T> curve = makeBenchmarkCurve<T>(degree);
		const std::vector<T> params = evenlySpacedParameters(curve, count);
		std::vector<geom::Point3<T>> out(count);
		evaluateAll<T>(curve, params, out);
		std::vector<double> seconds{};
		for (int run = 0; run < kRuns; ++run)
		{
			seconds.push_back(evaluateAll<T>(curve, params, out));
		}
		std::ranges::sort(seconds);
		return {seconds.front(), seconds[seconds.size() / 2], checksum<T>(out)};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printRow(std::string_view type, const Timing& timing, std::size_t count)
	{
		const double n = static_cast<double>(count);
		const double nsPerPoint = timing.bestSeconds * 1e9 / n;
		const double millionPerSecond = n / timing.bestSeconds / 1e6;
		std::println("{:<8} {:>10.3f} {:>11.3f} {:>13.2f} {:>11.1f}   {:.6e}", type, timing.bestSeconds * 1e3, timing.medianSeconds * 1e3, nsPerPoint, millionPerSecond, timing.checksum);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printHeader(std::size_t count, int degree)
	{
#ifdef NDEBUG
		const std::string_view build = "Release";
#else
		const std::string_view build = "Debug (asserts on): timings are NOT representative";
#endif
		std::println("NURBS curve point evaluation, CPU, 1 thread ({} build)", build);
		std::println("curve: degree {}, {} control points, rational 3D spiral; N = {} points; best and median of {} runs", degree, kNumControlPoints, count, kRuns);
		std::println("{:<8} {:>10} {:>11} {:>13} {:>11}   {}", "type", "best [ms]", "median [ms]", "per point [ns]", "Mpoints/s", "checksum");
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Saves the benchmark curve (double) with its control polygon as an SVG next to the executable.
	void writeCurveSvg(const char* executable, int degree)
	{
		const std::filesystem::path directory = std::filesystem::path(executable).parent_path() / "svg";
		std::filesystem::create_directories(directory);
		const std::filesystem::path path = directory / "benchmark_curve.svg";
		viz::SvgDocument svg(800, 800);
		viz::addNurbsCurve(svg, makeBenchmarkCurve<double>(degree), "#1f6feb", true);
		svg.save(path);
		std::println("wrote {}", path.string());
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A positive integer, or 0 if the text is not one.
	std::size_t parseNumber(std::string_view text)
	{
		std::size_t value = 0;
		const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
		if (error != std::errc{} || end != text.data() + text.size())
		{
			return 0;
		}
		return value;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// [N] [degree] in that order, plus an optional --svg anywhere.
	Options parseOptions(int argc, char* argv[])
	{
		Options options{};
		std::vector<std::string_view> numbers{};
		for (int i = 1; i < argc; ++i)
		{
			const std::string_view arg = argv[i];
			if (arg == "--svg")
			{
				options.writeSvg = true;
			}
			else
			{
				numbers.push_back(arg);
			}
		}
		options.count = numbers.size() > 0 ? parseNumber(numbers[0]) : options.count;
		options.degree = numbers.size() > 1 ? parseNumber(numbers[1]) : options.degree;
		options.valid = numbers.size() <= 2 && options.count >= 2 && options.degree >= 1 && options.degree <= static_cast<std::size_t>(geom::kMaxDegree);
		return options;
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main(int argc, char* argv[])
{
	const Options options = parseOptions(argc, argv);
	if (!options.valid)
	{
		std::println(stderr, "usage: bench_curve_point [N >= 2] [1 <= degree <= {}] [--svg]", geom::kMaxDegree);
		return 1;
	}
	const int degree = static_cast<int>(options.degree);
	if (options.writeSvg)
	{
		writeCurveSvg(argv[0], degree);
	}
	printHeader(options.count, degree);
	printRow("float", timeEvaluation<float>(degree, options.count), options.count);
	printRow("double", timeEvaluation<double>(degree, options.count), options.count);
	return 0;
}
