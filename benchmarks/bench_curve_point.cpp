#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "geom/basis_table.hpp"
#include "geom/nurbs_curve.hpp"
#include "viz/svg_document.hpp"

// Benchmark: evaluate N points C(u) on a NURBS curve on the CPU (one thread) and time it, with two
// methods:
//   direct  everything per point: findSpan (A2.1), basisFunctions (A2.2), weighted sum (A4.1);
//   table   the basis polynomials of every span are computed once (BasisTable); per point only
//           findSpan, Horner on the span's p + 1 polynomials, and the same weighted sum.
// The table's one-time build is timed separately, and the largest distance between the two
// methods' points is printed as a check that they agree.
//
//   bench_curve_point [N] [degree] [--knots K] [--svg]      defaults: N = 1000000, degree = 3
//
// --knots K sets the number of knots (uniform and clamped, so K - degree - 1 control points;
// K >= 2 * (degree + 1)). Without it the curve has 100 control points.
// --svg also saves a picture of the benchmark curve (xy projection) to svg/benchmark_curve.svg
// next to the executable.
//
// Only the evaluation loops are timed: the curve, the parameters, the table and the output arrays
// are set up before. One warm-up run, then the best and the median of kRuns runs. The checksum is printed
// so the compiler cannot skip the work, and it lets float and double results be compared.

namespace
{
	using Clock = std::chrono::steady_clock;
	
	constexpr std::size_t kDefaultControlPoints = 100;
	constexpr int kRuns = 5;
	
	// Written by timeTableBuild so the compiler cannot drop the table builds it times.
	double volatile checksumSink = 0.0;
	
	struct Timing
	{
		double bestSeconds{};
		double medianSeconds{};
		double checksum{};
	};
	
	// Both methods on one curve and one type.
	struct Comparison
	{
		Timing direct{};
		Timing table{};
		double buildSeconds{};      // best of kRuns builds of the BasisTable
		double maxDifference{};     // largest |C_table(u) - C_direct(u)| over all u
	};
	
	struct Options
	{
		std::size_t count{1'000'000};
		std::size_t degree{3};
		std::size_t numKnots{};
		bool knotsGiven{false};
		bool writeSvg{false};
		bool valid{true};
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Uniform clamped knots: degree + 1 zeros, evenly spaced interior knots, degree + 1 ones.
	template <std::floating_point T>
	std::vector<T> uniformClampedKnots(int degree, std::size_t numPoints)
	{
		auto const p = static_cast<std::size_t>(degree);
		std::size_t const numInterior = numPoints - p - 1;
		std::vector<T> knots(p + 1, T{0});
		for (std::size_t k = 1; k <= numInterior; ++k)
		{
			knots.push_back(static_cast<T>(k) / static_cast<T>(numInterior + 1));
		}
		knots.insert(knots.end(), p + 1, T{1});
		return knots;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A rational 3D spiral with numKnots uniform clamped knots: control points on a widening spiral
	// in xy that rises in z (about six turns per 100 points), weights between 0.5 and 1.5.
	// Precondition: numKnots >= 2 * (degree + 1).
	template <std::floating_point T>
	geom::NurbsCurve<T> makeBenchmarkCurve(int degree, std::size_t numKnots)
	{
		std::size_t const numControlPoints = numKnots - static_cast<std::size_t>(degree) - 1;
		std::vector<geom::Point3<T>> points{};
		std::vector<T> weights{};
		for (std::size_t i = 0; i < numControlPoints; ++i)
		{
			double const t = static_cast<double>(i);
			double const radius = 1.0 + 0.08 * t;
			points.push_back({static_cast<T>(radius * std::cos(0.4 * t)), static_cast<T>(radius * std::sin(0.4 * t)), static_cast<T>(0.05 * t)});
			weights.push_back(static_cast<T>(1.0 + 0.5 * std::sin(0.7 * t)));
		}
		return geom::NurbsCurve<T>(degree, uniformClampedKnots<T>(degree, numControlPoints), points, weights);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// N parameters evenly spaced over [a, b]; the last one is exactly b.
	template <std::floating_point T>
	std::vector<T> evenlySpacedParameters(geom::NurbsCurve<T> const& curve, std::size_t count)
	{
		auto const [a, b] = curve.domain();
		std::vector<T> params(count, b);
		for (std::size_t k = 0; k + 1 < count; ++k)
		{
			params[k] = a + (b - a) * static_cast<T>(k) / static_cast<T>(count - 1);
		}
		return params;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The timed loop: out[k] = evaluate(params[k]). Returns the elapsed time in seconds.
	template <std::floating_point T, typename Evaluate>
	double evaluateAll(Evaluate const& evaluate, std::span<T const> params, std::span<geom::Point3<T>> out)
	{
		Clock::time_point const start = Clock::now();
		for (std::size_t k = 0; k < params.size(); ++k)
		{
			out[k] = evaluate(params[k]);
		}
		Clock::time_point const stop = Clock::now();
		return std::chrono::duration<double>(stop - start).count();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	double checksum(std::span<geom::Point3<T> const> points)
	{
		double sum = 0.0;
		for (geom::Point3<T> const& p : points)
		{
			sum += static_cast<double>(p.x) + static_cast<double>(p.y) + static_cast<double>(p.z);
		}
		return sum;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One warm-up run, then kRuns timed runs.
	template <std::floating_point T, typename Evaluate>
	Timing timeMethod(Evaluate const& evaluate, std::span<T const> params, std::span<geom::Point3<T>> out)
	{
		evaluateAll<T>(evaluate, params, out);
		std::vector<double> seconds{};
		for (int run = 0; run < kRuns; ++run)
		{
			seconds.push_back(evaluateAll<T>(evaluate, params, out));
		}
		std::ranges::sort(seconds);
		return {seconds.front(), seconds[seconds.size() / 2], checksum<T>(out)};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The one-time cost of the table: best of kRuns builds.
	template <std::floating_point T>
	double timeTableBuild(geom::NurbsCurve<T> const& curve)
	{
		double best = 0.0;
		for (int run = 0; run < kRuns; ++run)
		{
			Clock::time_point const start = Clock::now();
			geom::BasisTable<T> const table(curve);
			double const seconds = std::chrono::duration<double>(Clock::now() - start).count();
			best = (run == 0 || seconds < best) ? seconds : best;
			checksumSink = checksumSink + static_cast<double>(table.coefficients().back());
		}
		return best;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The largest distance between corresponding points of a and b.
	template <std::floating_point T>
	double maxDistance(std::span<geom::Point3<T> const> a, std::span<geom::Point3<T> const> b)
	{
		double result = 0.0;
		for (std::size_t k = 0; k < a.size(); ++k)
		{
			double const dx = static_cast<double>(a[k].x) - static_cast<double>(b[k].x);
			double const dy = static_cast<double>(a[k].y) - static_cast<double>(b[k].y);
			double const dz = static_cast<double>(a[k].z) - static_cast<double>(b[k].z);
			result = std::max(result, std::sqrt(dx * dx + dy * dy + dz * dz));
		}
		return result;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Both methods on the same curve and parameters.
	template <std::floating_point T>
	Comparison compareMethods(int degree, std::size_t numKnots, std::size_t count)
	{
		geom::NurbsCurve<T> const curve = makeBenchmarkCurve<T>(degree, numKnots);
		std::vector<T> const params = evenlySpacedParameters(curve, count);
		std::vector<geom::Point3<T>> direct(count);
		std::vector<geom::Point3<T>> fromTable(count);
		double const buildSeconds = timeTableBuild(curve);
		geom::BasisTable<T> const table(curve);
		Timing const directTiming = timeMethod<T>([&curve](T u) { return curve(u); }, params, direct);
		Timing const tableTiming = timeMethod<T>([&curve, &table](T u) { return geom::curvePointFromTable(curve, table, u); }, params, fromTable);
		return {directTiming, tableTiming, buildSeconds, maxDistance<T>(direct, fromTable)};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// 1000000 -> "1,000,000".
	std::string withThousands(std::size_t value)
	{
		std::string digits = std::to_string(value);
		for (std::size_t i = digits.size(); i > 3; i -= 3)
		{
			digits.insert(i - 3, ",");
		}
		return digits;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printHeader(std::size_t count, int degree, std::size_t numKnots)
	{
		auto const p = static_cast<std::size_t>(degree);
		std::println("NURBS curve point evaluation (CPU, 1 thread)");
#ifndef NDEBUG
		std::println("WARNING: Debug build with asserts on, the timings are NOT representative");
#endif
		std::println();
		std::println("  curve   degree {}, {} knots, {} spans, {} control points (rational 3D spiral)", degree, withThousands(numKnots), withThousands(numKnots - 2 * p - 1), withThousands(numKnots - p - 1));
		std::println("  points  {} evenly spaced parameters, best and median of {} runs", withThousands(count), kRuns);
		std::println("  direct  every point: findSpan, basisFunctions (A2.2), weighted sum");
		std::println("  table   basis polynomials of all spans computed once, then per point: findSpan, Horner, weighted sum");
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printRow(std::string_view method, Timing const& timing, std::size_t count)
	{
		double const n = static_cast<double>(count);
		std::println("    {:<8}{:>10.2f}{:>13.2f}{:>16.2f}{:>12.1f}", method, timing.bestSeconds * 1e3, timing.medianSeconds * 1e3, timing.bestSeconds * 1e9 / n, n / timing.bestSeconds / 1e6);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One block: the two methods, then the speedup, the table's one-time cost and the agreement.
	template <std::floating_point T>
	Comparison runComparison(std::string_view type, int degree, std::size_t numKnots, std::size_t count)
	{
		Comparison const c = compareMethods<T>(degree, numKnots, count);
		auto const p = static_cast<std::size_t>(degree);
		double const kilobytes = static_cast<double>((numKnots - 2 * p - 1) * ((p + 1) * (p + 1) + 1) * sizeof(T)) / 1024.0;
		std::println();
		std::println("  {:<10}{:>10}{:>13}{:>16}{:>12}", type, "best [ms]", "median [ms]", "per point [ns]", "Mpoints/s");
		printRow("direct", c.direct, count);
		printRow("table", c.table, count);
		std::println("    table is {:.2f}x faster; built once in {:.3f} ms ({:.1f} KB); max |table - direct| = {:.1e}", c.direct.bestSeconds / c.table.bestSeconds, c.buildSeconds * 1e3, kilobytes, c.maxDifference);
		return c;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The checksums only make sure the compiler cannot skip the work; direct and table should match.
	void printChecksums(Comparison const& floats, Comparison const& doubles)
	{
		std::println();
		std::println("  checksums  float  {:.6e} (direct)  {:.6e} (table)", floats.direct.checksum, floats.table.checksum);
		std::println("             double {:.6e} (direct)  {:.6e} (table)", doubles.direct.checksum, doubles.table.checksum);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Saves the benchmark curve (double) with its control polygon as an SVG next to the executable.
	void writeCurveSvg(char const* executable, int degree, std::size_t numKnots)
	{
		std::filesystem::path const directory = std::filesystem::path(executable).parent_path() / "svg";
		std::filesystem::create_directories(directory);
		std::filesystem::path const path = directory / "benchmark_curve.svg";
		viz::SvgDocument svg(800, 800);
		viz::addNurbsCurve(svg, makeBenchmarkCurve<double>(degree, numKnots), "#1f6feb", true);
		svg.save(path);
		std::println("wrote {}", path.string());
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A positive integer, or 0 if the text is not one.
	std::size_t parseNumber(std::string_view text)
	{
		std::size_t value = 0;
		auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
		if (error != std::errc{} || end != text.data() + text.size())
		{
			return 0;
		}
		return value;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Handles argv[i] if it is a flag: --svg, or --knots K (which also consumes K, so it advances i).
	// Returns false for a positional argument.
	bool parseFlag(int argc, char* argv[], int& i, Options& options)
	{
		std::string_view const arg = argv[i];
		if (arg == "--svg")
		{
			options.writeSvg = true;
			return true;
		}
		if (arg == "--knots")
		{
			options.numKnots = i + 1 < argc ? parseNumber(argv[++i]) : 0;
			options.knotsGiven = true;
			return true;
		}
		return false;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// [N] [degree] in that order, plus the optional flags --knots K and --svg anywhere.
	Options parseOptions(int argc, char* argv[])
	{
		Options options{};
		std::vector<std::string_view> numbers{};
		for (int i = 1; i < argc; ++i)
		{
			if (!parseFlag(argc, argv, i, options))
			{
				numbers.push_back(argv[i]);
			}
		}
		options.count = numbers.size() > 0 ? parseNumber(numbers[0]) : options.count;
		options.degree = numbers.size() > 1 ? parseNumber(numbers[1]) : options.degree;
		options.numKnots = options.knotsGiven ? options.numKnots : kDefaultControlPoints + options.degree + 1;
		options.valid = numbers.size() <= 2 && options.count >= 2 && options.degree >= 1 && options.degree <= static_cast<std::size_t>(geom::kMaxDegree) && options.numKnots >= 2 * (options.degree + 1);
		return options;
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main(int argc, char* argv[])
{
	Options const options = parseOptions(argc, argv);
	if (!options.valid)
	{
		std::println(stderr, "usage: bench_curve_point [N >= 2] [1 <= degree <= {}] [--knots K >= 2 * (degree + 1)] [--svg]", geom::kMaxDegree);
		return 1;
	}
	int const degree = static_cast<int>(options.degree);
	if (options.writeSvg)
	{
		writeCurveSvg(argv[0], degree, options.numKnots);
	}
	printHeader(options.count, degree, options.numKnots);
	Comparison const floats = runComparison<float>("float", degree, options.numKnots, options.count);
	Comparison const doubles = runComparison<double>("double", degree, options.numKnots, options.count);
	printChecksums(floats, doubles);
	return 0;
}
