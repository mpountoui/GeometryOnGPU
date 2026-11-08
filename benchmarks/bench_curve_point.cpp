#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iterator>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "bench_common.hpp"
#include "geom/basis_table.hpp"
#include "geom/curve_table.hpp"
#include "geom/inverse_span_lengths.hpp"
#include "geom/nurbs_curve.hpp"
#include "geom/span_lookup.hpp"
#include "viz/svg_document.hpp"

#ifdef _OPENMP
#include "bench_threaded.hpp"
#endif

#ifdef GEOM_HAS_CUDA
#include <cuda_runtime.h>

#include "gpu/basis_table_gpu.hpp"
#include "gpu/cuda_check.hpp"
#include "gpu/curve_points_gpu.hpp"
#include "gpu/curve_table_gpu.hpp"
#include "gpu/device_buffer.hpp"
#include "gpu/inverse_span_lengths_gpu.hpp"
#include "gpu/span_lookup_gpu.hpp"
#endif

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
// --knots K (or --knots=K) sets the number of knots (uniform and clamped, so K - degree - 1 control points;
// K >= 2 * (degree + 1)). Without it the curve has 100 control points.
// --svg also saves a picture of the benchmark curve (xy projection) to svg/benchmark_curve.svg
// next to the executable.
//
// Only the evaluation loops are timed: the curve, the parameters, the tables and the output arrays
// are set up before. One warm-up run, then the best and the median of kRuns runs. The checksum is printed
// so the compiler cannot skip the work, and it lets float and double results be compared.
//
// With OpenMP the tables are also built with all usable threads; queries stay on one thread, since each u arrives
// on its own. With CUDA a last method builds the tables on the GPU, downloads nothing, and evaluates each point
// with its own kernel launch: u as the kernel's argument, then a download of the point. Each query waits for the
// GPU, so only a limited number of points is measured (kGpuQueries) and the time is scaled per point.
// bench_precompute times the builds themselves in more detail.

namespace
{
	using bench::bestOf;
	using bench::checksumSink;
	using bench::Clock;
	using bench::kRuns;
	using bench::timeCpuBuildOnce;
	using bench::timeCpuInverseSpanLengthsBuildOnce;
	using bench::makeBenchmarkCurve;
	using bench::parseNumber;
	using bench::secondsSince;
	using bench::withThousands;
	
	constexpr std::size_t kDefaultControlPoints = 100;
	
	struct EvaluationTimes
	{
		double bestSeconds{};
		double medianSeconds{};
		double checksum{};
	};
	
	// The four methods on one curve and one type.
	struct MethodTimes
	{
		EvaluationTimes direct{};
		EvaluationTimes basis{};
		EvaluationTimes curve{};
		EvaluationTimes lookup{};
		double basisBuildSeconds{};     // bestOf builds of the BasisTable
		double curveBuildSeconds{};     // bestOf builds of the CurveTable (its BasisTable included)
		double lookupBuildSeconds{};    // bestOf builds of the SpanLookup alone
		double invHBuildSeconds{};      // bestOf builds of the 1 / h array, which every table method needs
		double basisDifference{};      // largest |C_basis(u) - C_direct(u)| over all u
		double curveDifference{};       // largest |C_curve(u) - C_direct(u)| over all u
		double lookupDifference{};      // largest |C_lookup(u) - C_direct(u)| over all u
		int buildThreads{1};            // usable CPU threads; 1 = no threaded rows
		double threadedInvHBuildSeconds{};     // bestOf builds with buildThreads threads, as above
		double threadedBasisBuildSeconds{};
		double threadedCurveBuildSeconds{};
		double threadedLookupBuildSeconds{};
		bool hasGpu{false};
		EvaluationTimes gpu{};          // one point per kernel launch; scaled to the same number of points as the others
		std::size_t gpuPoints{};        // points actually measured
		double gpuBuildSeconds{};       // bestOf builds on the GPU, nothing downloaded
		double gpuDifference{};         // largest |C_gpu(u) - C_direct(u)| over the measured points
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
	double timeEvaluation(Evaluate const& evaluate, std::span<T const> params, std::span<geom::Point3<T>> out)
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
	
	// One warm-up run, then kRuns timed runs: their best and median.
	template <std::floating_point T, typename Evaluate>
	EvaluationTimes measureEvaluation(Evaluate const& evaluate, std::span<T const> params, std::span<geom::Point3<T>> out)
	{
		timeEvaluation<T>(evaluate, params, out);
		std::vector<double> seconds{};
		for (int run = 0; run < kRuns; ++run)
		{
			seconds.push_back(timeEvaluation<T>(evaluate, params, out));
		}
		std::ranges::sort(seconds);
		return {seconds.front(), seconds[seconds.size() / 2], checksum<T>(out)};
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
	
	// The one-time build costs of the three tables and the 1 / h array (build once, bestOf).
	template <std::floating_point T>
	void measureTableBuilds(geom::NurbsCurve<T> const& curve, MethodTimes& c)
	{
		c.basisBuildSeconds  = bestOf([&] { return timeCpuBuildOnce<geom::BasisTable<T>>(curve); });
		c.curveBuildSeconds  = bestOf([&] { return timeCpuBuildOnce<geom::CurveTable<T>>(curve); });
		c.lookupBuildSeconds = bestOf([&] { return timeCpuBuildOnce<geom::SpanLookup<T>>(curve); });
		c.invHBuildSeconds   = bestOf([&] { return timeCpuInverseSpanLengthsBuildOnce<T>(curve); });
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
#ifdef _OPENMP
	// The 1 / h array with OpenMP, allocation included.
	template <std::floating_point T>
	double timeThreadedInverseSpanLengthsBuild(geom::NurbsCurve<T> const& curve, int threads)
	{
		Clock::time_point const start = Clock::now();
		std::vector<T> inverseSpanLengths(curve.knots().size() - 2 * static_cast<std::size_t>(curve.degree()) - 1);
		bench::fillInverseSpanLengthsThreaded<T>(curve.degree(), curve.knots(), inverseSpanLengths, threads);
		double const seconds = secondsSince(start);
		checksumSink = checksumSink + static_cast<double>(inverseSpanLengths.back());
		return seconds;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The basis table with OpenMP, allocation included.
	template <std::floating_point T>
	double timeThreadedBasisBuild(geom::NurbsCurve<T> const& curve, int threads)
	{
		auto const p = static_cast<std::size_t>(curve.degree());
		Clock::time_point const start = Clock::now();
		std::vector<T> basis((curve.knots().size() - 2 * p - 1) * (p + 1) * (p + 1));
		bench::fillBasisThreaded<T>(curve.degree(), curve.knots(), basis, threads);
		double const seconds = secondsSince(start);
		checksumSink = checksumSink + static_cast<double>(basis.back());
		return seconds;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The curve table with OpenMP, allocation included. Its basis table is built inside and freed inside, as in
	// the CurveTable(curve) constructor.
	template <std::floating_point T>
	double timeThreadedCurveBuild(geom::NurbsCurve<T> const& curve, int threads)
	{
		auto const p = static_cast<std::size_t>(curve.degree());
		std::size_t const numSpans = curve.knots().size() - 2 * p - 1;
		Clock::time_point const start = Clock::now();
		std::vector<geom::Point4<T>> table(numSpans * (p + 1));
		{
			std::vector<T> basis(numSpans * (p + 1) * (p + 1));
			bench::fillBasisThreaded<T>(curve.degree(), curve.knots(), basis, threads);
			bench::fillCurveThreaded<T>(curve.degree(), basis, curve.weightedControlPoints(), table, threads);
		}
		double const seconds = secondsSince(start);
		checksumSink = checksumSink + static_cast<double>(table.back().w);
		return seconds;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The lookup table with OpenMP, allocation included. lookup only provides the domain values.
	template <std::floating_point T>
	double timeThreadedLookupBuild(geom::NurbsCurve<T> const& curve, geom::SpanLookup<T> const& lookup, int threads)
	{
		Clock::time_point const start = Clock::now();
		std::vector<std::uint32_t> cells(lookup.firstSpans().size());
		bench::fillLookupThreaded<T>(curve.degree(), curve.knots(), lookup, cells, threads);
		double const seconds = secondsSince(start);
		checksumSink = checksumSink + static_cast<double>(cells.back());
		return seconds;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The same builds as measureTableBuilds, with all usable threads. Nothing to do on a single thread.
	template <std::floating_point T>
	void measureThreadedBuilds(geom::NurbsCurve<T> const& curve, MethodTimes& c)
	{
		int const threads = bench::usableThreads();
		if (threads == 1)
		{
			return;
		}
		geom::SpanLookup<T> const lookup(curve);
		c.buildThreads = threads;
		c.threadedInvHBuildSeconds   = bestOf([&] { return timeThreadedInverseSpanLengthsBuild<T>(curve, threads); });
		c.threadedBasisBuildSeconds  = bestOf([&] { return timeThreadedBasisBuild<T>(curve, threads); });
		c.threadedCurveBuildSeconds  = bestOf([&] { return timeThreadedCurveBuild<T>(curve, threads); });
		c.threadedLookupBuildSeconds = bestOf([&] { return timeThreadedLookupBuild<T>(curve, lookup, threads); });
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#endif
	
#ifdef GEOM_HAS_CUDA
	// A query waits for the GPU each time, so this many points at most are measured.
	constexpr std::size_t kGpuQueries = 10'000;
	
	// What queries on the GPU read: everything stays in device memory.
	template <std::floating_point T>
	struct GpuCurve
	{
		gpu::DeviceBuffer<T> knots{};
		gpu::DeviceBuffer<T> inverseSpanLengths{};
		gpu::DeviceCurveTable<T> table{};
		gpu::DeviceSpanLookup<T> lookup{};
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Builds everything queries need on the GPU and waits for it. The basis table and the control points are
	// temporaries: freed here, as the CPU's CurveTable(curve) frees its basis table.
	template <std::floating_point T>
	GpuCurve<T> buildGpuCurve(geom::NurbsCurve<T> const& curve)
	{
		auto const [domainStart, domainEnd] = curve.domain();
		gpu::DeviceBuffer<T> knots{curve.knots()};
		gpu::DeviceBuffer<T> inverseSpanLengths = gpu::buildInverseSpanLengthsOnDevice<T>(curve.degree(), knots);
		gpu::DeviceSpanLookup<T> lookup = gpu::buildSpanLookupOnDevice<T>(curve.degree(), knots, domainStart, domainEnd);
		gpu::DeviceCurveTable<T> table = gpu::allocateCurveTableOnDevice<T>(curve.degree(), inverseSpanLengths.size());
		{
			gpu::DeviceBuffer<geom::Point4<T>> const points{curve.weightedControlPoints()};
			gpu::DeviceBasisTable<T> const basis = gpu::buildBasisTableOnDevice<T>(curve.degree(), knots);
			gpu::fillCurveTableOnDevice<T>(basis, points, table);
			gpu::checkCuda(cudaDeviceSynchronize());
		}
		return {std::move(knots), std::move(inverseSpanLengths), std::move(table), std::move(lookup)};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	double timeGpuBuild(geom::NurbsCurve<T> const& curve)
	{
		Clock::time_point const start = Clock::now();
		GpuCurve<T> const device = buildGpuCurve<T>(curve);
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One query: launch one thread with u as its argument, then download the point.
	template <std::floating_point T>
	geom::Point3<T> evaluateOnGpu(GpuCurve<T> const& device, T u, gpu::DeviceBuffer<geom::Point3<T>>& result)
	{
		gpu::evaluateCurvePointOnDevice<T>(device.knots, device.inverseSpanLengths, device.table, device.lookup, u, result);
		geom::Point3<T> point{};
		result.download(std::span<geom::Point3<T>>(&point, 1));
		return point;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The queries, at most kGpuQueries of them, with their times scaled to count points like the other methods.
	template <std::floating_point T>
	void measureGpuQueries(geom::NurbsCurve<T> const& curve, std::size_t count, MethodTimes& c)
	{
		GpuCurve<T> const device = buildGpuCurve<T>(curve);
		gpu::DeviceBuffer<geom::Point3<T>> result(1);
		c.gpuPoints = std::min(count, kGpuQueries);
		std::vector<T> const params = evenlySpacedParameters(curve, c.gpuPoints);
		std::vector<geom::Point3<T>> points(c.gpuPoints);
		std::vector<geom::Point3<T>> direct(c.gpuPoints);
		std::ranges::transform(params, direct.begin(), [&curve](T u) { return curve(u); });
		c.gpu = measureEvaluation<T>([&device, &result](T u) { return evaluateOnGpu<T>(device, u, result); }, params, points);
		double const scale = static_cast<double>(count) / static_cast<double>(c.gpuPoints);
		c.gpu.bestSeconds = c.gpu.bestSeconds * scale;
		c.gpu.medianSeconds = c.gpu.medianSeconds * scale;
		c.gpuDifference = maxDistance<T>(direct, points);
		c.hasGpu = true;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#endif
	
	// The builds of all table methods, and the queries of the GPU method (which needs its own tables).
	template <std::floating_point T>
	void measurePrecomputation(geom::NurbsCurve<T> const& curve, [[maybe_unused]] std::size_t count, MethodTimes& c)
	{
		measureTableBuilds(curve, c);
#ifdef _OPENMP
		measureThreadedBuilds<T>(curve, c);
#endif
#ifdef GEOM_HAS_CUDA
		c.gpuBuildSeconds = bestOf([&] { return timeGpuBuild<T>(curve); });
		measureGpuQueries<T>(curve, count, c);
#endif
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The four methods on the same curve and parameters; out[0] holds the direct points.
	template <std::floating_point T>
	MethodTimes measureMethods(int degree, std::size_t numKnots, std::size_t count)
	{
		geom::NurbsCurve<T> const curve = makeBenchmarkCurve<T>(degree, numKnots);
		std::vector<T> const params = evenlySpacedParameters(curve, count);
		std::vector<std::vector<geom::Point3<T>>> out(4, std::vector<geom::Point3<T>>(count));
		geom::BasisTable<T> const basis(curve);
		geom::CurveTable<T> const table(curve);
		geom::SpanLookup<T> const lookup(curve);
		std::vector<T> const invH = geom::buildInverseSpanLengths<T>(curve.degree(), curve.knots());
		MethodTimes c{};
		c.direct = measureEvaluation<T>([&curve](T u) { return curve(u); }, params, out[0]);
		c.basis = measureEvaluation<T>([&curve, &basis, &invH](T u) { return geom::curvePointFromTable(curve, basis, invH, u); }, params, out[1]);
		c.curve = measureEvaluation<T>([&curve, &table, &invH](T u) { return geom::curvePointFromTable(curve, table, invH, u); }, params, out[2]);
		c.lookup = measureEvaluation<T>([&curve, &table, &invH, &lookup](T u) { return geom::curvePointFromTable(curve, table, invH, lookup, u); }, params, out[3]);
		measurePrecomputation<T>(curve, count, c);
		c.basisDifference = maxDistance<T>(out[0], out[1]);
		c.curveDifference = maxDistance<T>(out[0], out[2]);
		c.lookupDifference = maxDistance<T>(out[0], out[3]);
		return c;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One row of the precomputation section: everything built before the first point.
	struct Precomputation
	{
		std::string name{};
		EvaluationTimes evaluation{};
		double seconds{};       // total build time
		double kilobytes{};     // total memory
	};
	
	// What queries keep, per method.
	struct MethodMemory
	{
		double basisKilobytes{};
		double curveKilobytes{};
		double lookupKilobytes{};
		double gpuKilobytes{};
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The 1 / h array counts for all; lookup also holds the curve table, the GPU also the knots.
	template <std::floating_point T>
	MethodMemory methodMemory(int degree, std::size_t numKnots)
	{
		auto const p = static_cast<std::size_t>(degree);
		std::size_t const numSpans = numKnots - 2 * p - 1;
		double const basis  = static_cast<double>(numSpans * ((p + 1) * (p + 1) + 1) * sizeof(T)) / 1024.0;
		double const curve  = static_cast<double>(numSpans * (4 * (p + 1) + 1) * sizeof(T)) / 1024.0;
		double const lookup = curve + static_cast<double>(2 * numSpans * sizeof(std::uint32_t)) / 1024.0;
		return {basis, curve, lookup, lookup + static_cast<double>(numKnots * sizeof(T)) / 1024.0};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Totals, each with the 1 / h array: basis = the basis table; curve = the curve table (its basis table built
	// inside); lookup = the curve table plus the lookup table. The queries are the same whatever built the tables.
	std::vector<Precomputation> threadedRows(MethodTimes const& c, MethodMemory const& memory)
	{
		std::string const threads = std::format(", {} threads", c.buildThreads);
		Precomputation const basis{"basis" + threads, c.basis, c.threadedBasisBuildSeconds + c.threadedInvHBuildSeconds, memory.basisKilobytes};
		Precomputation const curve{"curve" + threads, c.curve, c.threadedCurveBuildSeconds + c.threadedInvHBuildSeconds, memory.curveKilobytes};
		Precomputation const lookup{"lookup" + threads, c.lookup, c.threadedCurveBuildSeconds + c.threadedLookupBuildSeconds + c.threadedInvHBuildSeconds, memory.lookupKilobytes};
		return {basis, curve, lookup};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The rows of one type: one thread, then with threads, then the GPU, each only if it was measured.
	template <std::floating_point T>
	std::vector<Precomputation> tableMethods(MethodTimes const& c, int degree, std::size_t numKnots)
	{
		MethodMemory const memory = methodMemory<T>(degree, numKnots);
		std::vector<Precomputation> rows{{"basis", c.basis, c.basisBuildSeconds + c.invHBuildSeconds, memory.basisKilobytes}, {"curve", c.curve, c.curveBuildSeconds + c.invHBuildSeconds, memory.curveKilobytes}, {"lookup", c.lookup, c.curveBuildSeconds + c.lookupBuildSeconds + c.invHBuildSeconds, memory.lookupKilobytes}};
		if (c.buildThreads > 1)
		{
			std::ranges::copy(threadedRows(c, memory), std::back_inserter(rows));
		}
		if (c.hasGpu)
		{
			rows.push_back({"gpu", c.gpu, c.gpuBuildSeconds, memory.gpuKilobytes});
		}
		return rows;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// After how many points the precomputation has paid for itself: total build time / time saved per
	// point ("never" if the method is not faster than direct).
	std::string paysOffAfter(EvaluationTimes const& direct, Precomputation const& method, std::size_t count)
	{
		double const savedPerPoint = (direct.bestSeconds - method.evaluation.bestSeconds) / static_cast<double>(count);
		if (!(savedPerPoint > 0.0))
		{
			return "never";
		}
		return withThousands(static_cast<std::size_t>(std::ceil(method.seconds / savedPerPoint))) + " points";
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printHeader(std::size_t count, int degree, std::size_t numKnots, MethodTimes const& floats)
	{
		auto const p = static_cast<std::size_t>(degree);
		std::println("Curve point evaluation (queries on one thread, one u at a time)");
#ifndef NDEBUG
		std::println("WARNING: Debug build with asserts on, the timings are NOT representative");
#endif
		std::println("  curve    degree {}, {} knots, {} spans, {} control points (rational 3D spiral)", degree, withThousands(numKnots), withThousands(numKnots - 2 * p - 1), withThousands(numKnots - p - 1));
		std::println("  queries  {} evenly spaced parameters, the same for every method", withThousands(count));
		std::println();
		std::println("METHODS");
		std::println("  direct  every point: findSpan, basisFunctions (A2.2), weighted sum");
		std::println("  basis   basis polynomials of all spans built once; per point: findSpan, Horner per basis function, weighted sum");
		std::println("  curve   basis x control points (the curve per span) built once; per point: findSpan, one 4D Horner");
		std::println("  lookup  curve table, with the span from a lookup table instead of the binary search");
		if (floats.hasGpu)
		{
			std::println("  gpu     curve table and lookup built on the GPU, nothing downloaded; per point: one kernel launch (u as its argument), download the point");
			std::println("          ({} points measured, scaled per point)", withThousands(floats.gpuPoints));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One type's columns of the time per point section.
	std::string timeColumns(EvaluationTimes const& t, EvaluationTimes const& direct, std::size_t count, bool isDirect)
	{
		double const n = static_cast<double>(count);
		std::string const relative = isDirect ? std::string("reference") : bench::speedInWords(direct.bestSeconds, t.bestSeconds);
		return std::format("{:>10.2f}{:>13.2f}{:>11.1f}  {:<13}", t.bestSeconds * 1e9 / n, t.medianSeconds * 1e9 / n, n / t.bestSeconds / 1e6, relative);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printTimeSection(MethodTimes const& f, MethodTimes const& d, std::size_t count)
	{
		std::println();
		std::println("TIME PER POINT  (best and median of {} runs, after one warm-up run)", kRuns);
		std::println("  {:<8}{:-^49}  {:-^49}", "", " float ", " double ");
		std::println("  {:<8}{:>10}{:>13}{:>11}  {:<13}  {:>10}{:>13}{:>11}  {}", "", "best [ns]", "median [ns]", "Mpoints/s", "vs direct", "best [ns]", "median [ns]", "Mpoints/s", "vs direct");
		std::println("  {:<8}{}  {}", "direct", timeColumns(f.direct, f.direct, count, true), timeColumns(d.direct, d.direct, count, true));
		std::println("  {:<8}{}  {}", "basis", timeColumns(f.basis, f.direct, count, false), timeColumns(d.basis, d.direct, count, false));
		std::println("  {:<8}{}  {}", "curve", timeColumns(f.curve, f.direct, count, false), timeColumns(d.curve, d.direct, count, false));
		std::println("  {:<8}{}  {}", "lookup", timeColumns(f.lookup, f.direct, count, false), timeColumns(d.lookup, d.direct, count, false));
		if (f.hasGpu)
		{
			std::println("  {:<8}{}  {}", "gpu", timeColumns(f.gpu, f.direct, count, false), timeColumns(d.gpu, d.direct, count, false));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One type's columns of the precomputation section.
	std::string precomputationColumns(EvaluationTimes const& direct, Precomputation const& method, std::size_t count)
	{
		return std::format("{:>11.1f}{:>12.1f}{:>20}", method.seconds * 1e6, method.kilobytes, paysOffAfter(direct, method, count));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printPrecomputationLegend(MethodTimes const& floats)
	{
		std::println("  build [us]      everything built, incl. the 1 / h array; best of at least {} runs and {} s", kRuns, bench::kMinSeconds);
		std::println("                  basis: basis table; curve: curve table, its basis table built inside; lookup: curve table + lookup table");
		if (floats.buildThreads > 1)
		{
			std::println("                  N threads: the same tables built with OpenMP; queries stay on one thread, each u arrives on its own");
		}
		if (floats.hasGpu)
		{
			std::println("                  gpu: upload knots and points, 1 / h, basis, curve and lookup kernels until done; nothing downloaded");
		}
		std::println("  memory [KB]     what queries keep: those tables and the 1 / h array (lookup: 2 cells per span; gpu: on the device, also the knots)");
		std::println("  pays off after  build time divided by the time saved per point against direct");
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printPrecomputationSection(std::span<Precomputation const> f, std::span<Precomputation const> d, MethodTimes const& floats, MethodTimes const& doubles, std::size_t count)
	{
		std::println();
		std::println("PRECOMPUTATION  (what each method builds before its first point)");
		printPrecomputationLegend(floats);
		std::println("  {:<20}{:-^43}  {:-^43}", "", " float ", " double ");
		std::println("  {:<20}{:>11}{:>12}{:>20}  {:>11}{:>12}{:>20}", "", "build [us]", "memory [KB]", "pays off after", "build [us]", "memory [KB]", "pays off after");
		for (std::size_t k = 0; k < f.size(); ++k)
		{
			std::println("  {:<20}{}  {}", f[k].name, precomputationColumns(floats.direct, f[k], count), precomputationColumns(doubles.direct, d[k], count));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printAgreementRow(std::string_view name, std::string_view floatDistance, std::string_view floatChecksum, std::string_view doubleDistance, std::string_view doubleChecksum)
	{
		std::println("  {:<8}{:>14}{:>16}  {:>14}{:>16}", name, floatDistance, floatChecksum, doubleDistance, doubleChecksum);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printAgreementSection(MethodTimes const& f, MethodTimes const& d)
	{
		auto const distance = [](double value) { return std::format("{:.1e}", value); };
		auto const sum = [](double value) { return std::format("{:.6e}", value); };
		std::println();
		std::println("AGREEMENT WITH DIRECT  (largest distance from direct's points; the checksums sum all points and should match)");
		std::println("  {:<8}{:-^30}  {:-^30}", "", " float ", " double ");
		std::println("  {:<8}{:>14}{:>16}  {:>14}{:>16}", "", "max distance", "checksum", "max distance", "checksum");
		printAgreementRow("direct", "reference", sum(f.direct.checksum), "reference", sum(d.direct.checksum));
		printAgreementRow("basis", distance(f.basisDifference), sum(f.basis.checksum), distance(d.basisDifference), sum(d.basis.checksum));
		printAgreementRow("curve", distance(f.curveDifference), sum(f.curve.checksum), distance(d.curveDifference), sum(d.curve.checksum));
		printAgreementRow("lookup", distance(f.lookupDifference), sum(f.lookup.checksum), distance(d.lookupDifference), sum(d.lookup.checksum));
		if (f.hasGpu)
		{
			printAgreementRow("gpu", distance(f.gpuDifference), "-", distance(d.gpuDifference), "-");
		}
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
	
	// Handles argv[i] if it is a flag: --svg, or --knots K / --knots=K (the first form also consumes K, so it advances i).
	// Returns false for a positional argument.
	bool parseFlag(int argc, char* argv[], int& i, Options& options)
	{
		std::string_view const arg = argv[i];
		if (arg == "--svg")
		{
			options.writeSvg = true;
			return true;
		}
		if (bench::parseKnotsFlag(argc, argv, i, options.numKnots))
		{
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
	MethodTimes const floats = measureMethods<float>(degree, options.numKnots, options.count);
	MethodTimes const doubles = measureMethods<double>(degree, options.numKnots, options.count);
	printHeader(options.count, degree, options.numKnots, floats);
	printTimeSection(floats, doubles, options.count);
	printPrecomputationSection(tableMethods<float>(floats, degree, options.numKnots), tableMethods<double>(doubles, degree, options.numKnots), floats, doubles, options.count);
	printAgreementSection(floats, doubles);
	return 0;
}
