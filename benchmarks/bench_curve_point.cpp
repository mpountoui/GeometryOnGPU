#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "geom/basis_table.hpp"
#include "geom/curve_table.hpp"
#include "geom/nurbs_curve.hpp"
#include "geom/span_lookup.hpp"
#include "viz/svg_document.hpp"

// Benchmark: evaluate N points C(u) on a NURBS curve on the CPU (one thread) and time it, with four
// methods:
//   direct  everything per point: findSpan (A2.1), basisFunctions (A2.2), weighted sum (A4.1);
//   basis   the basis polynomials of every span are computed once (BasisTable); per point
//           findSpan, Horner on the span's p + 1 polynomials, and the same weighted sum;
//   curve   basis polynomials times control points, i.e. the curve itself per span, computed once
//           (CurveTable); per point findSpan and one Horner evaluation of a 4D polynomial;
//   lookup  the curve table, with the span from a SpanLookup table instead of findSpan's binary search.
// The tables' one-time builds are timed separately, and the largest distance between each table
// method's points and the direct ones is printed as a check that they agree.
//
//   bench_curve_point [N] [degree] [--knots K] [--svg]      defaults: N = 1000000, degree = 3
//
// --knots K sets the number of knots (uniform and clamped, so K - degree - 1 control points;
// K >= 2 * (degree + 1)). Without it the curve has 100 control points.
// --svg also saves a picture of the benchmark curve (xy projection) to svg/benchmark_curve.svg
// next to the executable.
//
// Only the evaluation loops are timed: the curve, the parameters, the tables and the output arrays
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
	
	// The four methods on one curve and one type.
	struct Comparison
	{
		Timing direct{};
		Timing basis{};
		Timing curve{};
		Timing lookup{};
		double basisBuildSeconds{};     // best of kRuns builds of the BasisTable
		double curveBuildSeconds{};     // best of kRuns builds of the CurveTable (its BasisTable included)
		double lookupBuildSeconds{};    // best of kRuns builds of the SpanLookup alone
		double basisDifference{};       // largest |C_basis(u) - C_direct(u)| over all u
		double curveDifference{};       // largest |C_curve(u) - C_direct(u)| over all u
		double lookupDifference{};      // largest |C_lookup(u) - C_direct(u)| over all u
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
	
	// A value that depends on the whole build, so the compiler cannot drop the builds timeBuild times.
	template <std::floating_point T>
	double sinkValue(geom::BasisTable<T> const& table)
	{
		return static_cast<double>(table.inverseSpanLengths().back());
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	double sinkValue(geom::CurveTable<T> const& table)
	{
		return static_cast<double>(table.inverseSpanLengths().back());
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	double sinkValue(geom::SpanLookup<T> const& lookup)
	{
		return static_cast<double>(lookup.firstSpans().back());
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The one-time cost of a table (BasisTable, CurveTable or SpanLookup) built from the curve: best of kRuns builds.
	template <typename Table, std::floating_point T>
	double timeBuild(geom::NurbsCurve<T> const& curve)
	{
		double best = 0.0;
		for (int run = 0; run < kRuns; ++run)
		{
			Clock::time_point const start = Clock::now();
			Table const table(curve);
			double const seconds = std::chrono::duration<double>(Clock::now() - start).count();
			best = (run == 0 || seconds < best) ? seconds : best;
			checksumSink = checksumSink + sinkValue(table);
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
	
	// The one-time build costs of the three tables.
	template <std::floating_point T>
	void timeBuilds(geom::NurbsCurve<T> const& curve, Comparison& c)
	{
		c.basisBuildSeconds = timeBuild<geom::BasisTable<T>>(curve);
		c.curveBuildSeconds = timeBuild<geom::CurveTable<T>>(curve);
		c.lookupBuildSeconds = timeBuild<geom::SpanLookup<T>>(curve);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The four methods on the same curve and parameters; out[0] holds the direct points.
	template <std::floating_point T>
	Comparison compareMethods(int degree, std::size_t numKnots, std::size_t count)
	{
		geom::NurbsCurve<T> const curve = makeBenchmarkCurve<T>(degree, numKnots);
		std::vector<T> const params = evenlySpacedParameters(curve, count);
		std::vector<std::vector<geom::Point3<T>>> out(4, std::vector<geom::Point3<T>>(count));
		geom::BasisTable<T> const basis(curve);
		geom::CurveTable<T> const table(curve);
		geom::SpanLookup<T> const lookup(curve);
		Comparison c{};
		c.direct = timeMethod<T>([&curve](T u) { return curve(u); }, params, out[0]);
		c.basis = timeMethod<T>([&curve, &basis](T u) { return geom::curvePointFromTable(curve, basis, u); }, params, out[1]);
		c.curve = timeMethod<T>([&curve, &table](T u) { return geom::curvePointFromTable(curve, table, u); }, params, out[2]);
		c.lookup = timeMethod<T>([&curve, &table, &lookup](T u) { return geom::curvePointFromTable(curve, table, lookup, u); }, params, out[3]);
		timeBuilds(curve, c);
		c.basisDifference = maxDistance<T>(out[0], out[1]);
		c.curveDifference = maxDistance<T>(out[0], out[2]);
		c.lookupDifference = maxDistance<T>(out[0], out[3]);
		return c;
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
		std::println("  basis   basis polynomials of all spans computed once; per point: findSpan, Horner per basis function, weighted sum");
		std::println("  curve   basis x control points (the curve per span) computed once; per point: findSpan, one 4D Horner");
		std::println("  lookup  curve table, with the span from a lookup table instead of the binary search");
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printRow(std::string_view method, Timing const& timing, std::size_t count)
	{
		double const n = static_cast<double>(count);
		std::println("    {:<8}{:>10.2f}{:>13.2f}{:>16.2f}{:>12.1f}", method, timing.bestSeconds * 1e3, timing.medianSeconds * 1e3, timing.bestSeconds * 1e9 / n, n / timing.bestSeconds / 1e6);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The precomputation of one method, everything built before the first point.
	struct Precomputation
	{
		std::string_view name{};
		Timing evaluation{};
		double seconds{};       // total build time
		double kilobytes{};     // total memory
		double difference{};    // largest |C_method(u) - C_direct(u)|
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// After how many points the precomputation has paid for itself: total build time / time saved per
	// point ("never" if the method is not faster than direct).
	std::string paysOffAfter(Timing const& direct, Precomputation const& method, std::size_t count)
	{
		double const savedPerPoint = (direct.bestSeconds - method.evaluation.bestSeconds) / static_cast<double>(count);
		if (!(savedPerPoint > 0.0))
		{
			return "never";
		}
		return withThousands(static_cast<std::size_t>(std::ceil(method.seconds / savedPerPoint))) + " points";
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One row of the precomputation table: speedup, total build time and memory, break-even, agreement.
	void printPrecomputation(Timing const& direct, Precomputation const& method, std::size_t count)
	{
		double const speedup = direct.bestSeconds / method.evaluation.bestSeconds;
		std::println("    {:<9}{:>8.2f}x{:>13.1f}{:>13.1f}{:>20}{:>13.1e}", method.name, speedup, method.seconds * 1e6, method.kilobytes, paysOffAfter(direct, method, count), method.difference);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One block: the four methods, then the precomputation of each table method. Totals: basis = the
	// basis table; curve = the curve table (it builds its basis table inside); lookup = the curve table
	// plus the lookup table.
	template <std::floating_point T>
	Comparison runComparison(std::string_view type, int degree, std::size_t numKnots, std::size_t count)
	{
		Comparison const c = compareMethods<T>(degree, numKnots, count);
		auto const p = static_cast<std::size_t>(degree);
		std::size_t const numSpans = numKnots - 2 * p - 1;
		double const basisKilobytes = static_cast<double>(numSpans * ((p + 1) * (p + 1) + 1) * sizeof(T)) / 1024.0;
		double const curveKilobytes = static_cast<double>(numSpans * (4 * (p + 1) + 1) * sizeof(T)) / 1024.0;
		double const lookupKilobytes = static_cast<double>(2 * numSpans * sizeof(std::uint32_t)) / 1024.0;
		std::println();
		std::println("  {:<10}{:>10}{:>13}{:>16}{:>12}", type, "best [ms]", "median [ms]", "per point [ns]", "Mpoints/s");
		printRow("direct", c.direct, count);
		printRow("basis", c.basis, count);
		printRow("curve", c.curve, count);
		printRow("lookup", c.lookup, count);
		std::println("    {:<9}{:>9}{:>13}{:>13}{:>20}{:>13}", "", "speedup", "precomp [us]", "memory [KB]", "pays off after", "max diff");
		printPrecomputation(c.direct, {"basis", c.basis, c.basisBuildSeconds, basisKilobytes, c.basisDifference}, count);
		printPrecomputation(c.direct, {"curve", c.curve, c.curveBuildSeconds, curveKilobytes, c.curveDifference}, count);
		printPrecomputation(c.direct, {"lookup", c.lookup, c.curveBuildSeconds + c.lookupBuildSeconds, curveKilobytes + lookupKilobytes, c.lookupDifference}, count);
		return c;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The checksums only make sure the compiler cannot skip the work; all four should match.
	void printChecksums(Comparison const& floats, Comparison const& doubles)
	{
		std::println();
		std::println("  checksums  float  {:.6e} (direct)  {:.6e} (basis)  {:.6e} (curve)  {:.6e} (lookup)", floats.direct.checksum, floats.basis.checksum, floats.curve.checksum, floats.lookup.checksum);
		std::println("             double {:.6e} (direct)  {:.6e} (basis)  {:.6e} (curve)  {:.6e} (lookup)", doubles.direct.checksum, doubles.basis.checksum, doubles.curve.checksum, doubles.lookup.checksum);
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
