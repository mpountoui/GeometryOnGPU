#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "bench_common.hpp"
#include "geom/basis_table.hpp"
#include "geom/curve_table.hpp"
#include "geom/inverse_span_lengths.hpp"
#include "geom/nurbs_curve.hpp"
#include "geom/span_lookup.hpp"
#include "viz/svg_document.hpp"

#ifdef GEOM_HAS_CUDA
#include <cuda_runtime.h>

#include "gpu/basis_table_gpu.hpp"
#include "gpu/cuda_check.hpp"
#include "gpu/cuda_event.hpp"
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
// --knots K sets the number of knots (uniform and clamped, so K - degree - 1 control points;
// K >= 2 * (degree + 1)). Without it the curve has 100 control points.
// --svg also saves a picture of the benchmark curve (xy projection) to svg/benchmark_curve.svg
// next to the executable.
//
// Only the evaluation loops are timed: the curve, the parameters, the tables and the output arrays
// are set up before. One warm-up run, then the best and the median of kRuns runs. The checksum is printed
// so the compiler cannot skip the work, and it lets float and double results be compared.
//
// With CUDA, a last part compares the basis table and curve table builds on the CPU and the GPU, one
// comparison (the same work on both, float and double) per printed row: as a rebuild (memory allocated
// beforehand: CPU fill functions) and as a build once (allocating inside: CPU constructors), the GPU
// with or without the copies to and from CPU memory, and the
// first build of each type (float's basis table runs before any other CUDA call, so it also shows the
// cost of starting CUDA). The printed legend explains the rows; results in docs/benchmarks.md.

namespace
{
	using bench::bestOf;
	using bench::Clock;
	using bench::formatMicroseconds;
	using bench::kRuns;
	using bench::kMinSeconds;
	using bench::makeBenchmarkCurve;
	using bench::parseNumber;
	using bench::secondsSince;
	using bench::withThousands;
	
	constexpr std::size_t kDefaultControlPoints = 100;
	
	// Written by timeCpuBuildOnce so the compiler cannot drop the table builds it times.
	double volatile checksumSink = 0.0;
	
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
	
	// A value that depends on the whole build, so the compiler cannot drop the builds timeCpuBuildOnce times.
	template <std::floating_point T>
	double sinkValue(geom::BasisTable<T> const& table)
	{
		return static_cast<double>(table.coefficients().back());
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	double sinkValue(geom::CurveTable<T> const& table)
	{
		return static_cast<double>(table.coefficients().back().w);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	double sinkValue(geom::SpanLookup<T> const& lookup)
	{
		return static_cast<double>(lookup.firstSpans().back());
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A build once of a table (BasisTable, CurveTable or SpanLookup) from the curve: its constructor,
	// allocation included.
	template <typename Table, std::floating_point T>
	double timeCpuBuildOnce(geom::NurbsCurve<T> const& curve)
	{
		Clock::time_point const start = Clock::now();
		Table const table(curve);
		double const seconds = secondsSince(start);
		checksumSink = checksumSink + sinkValue(table);
		return seconds;
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
	
	// A build once of the 1 / h array, allocation included.
	template <std::floating_point T>
	double timeCpuInverseSpanLengthsBuildOnce(geom::NurbsCurve<T> const& curve)
	{
		Clock::time_point const start = Clock::now();
		std::vector<T> const inverseSpanLengths = geom::buildInverseSpanLengths<T>(curve.degree(), curve.knots());
		double const seconds = secondsSince(start);
		checksumSink = checksumSink + static_cast<double>(inverseSpanLengths.back());
		return seconds;
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
		measureTableBuilds(curve, c);
		c.basisDifference = maxDistance<T>(out[0], out[1]);
		c.curveDifference = maxDistance<T>(out[0], out[2]);
		c.lookupDifference = maxDistance<T>(out[0], out[3]);
		return c;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printMethodsHeader(std::size_t count, int degree, std::size_t numKnots)
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
	
	void printEvaluationRow(std::string_view method, EvaluationTimes const& timing, std::size_t count)
	{
		double const n = static_cast<double>(count);
		std::println("    {:<8}{:>10.2f}{:>13.2f}{:>16.2f}{:>12.1f}", method, timing.bestSeconds * 1e3, timing.medianSeconds * 1e3, timing.bestSeconds * 1e9 / n, n / timing.bestSeconds / 1e6);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The precomputation of one method, everything built before the first point.
	struct Precomputation
	{
		std::string_view name{};
		EvaluationTimes evaluation{};
		double seconds{};       // total build time
		double kilobytes{};     // total memory
		double difference{};    // largest |C_method(u) - C_direct(u)|
	};
	
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
	
	// One row of the precomputation table: speedup, total build time and memory, break-even, agreement.
	void printPrecomputation(EvaluationTimes const& direct, Precomputation const& method, std::size_t count)
	{
		double const speedup = direct.bestSeconds / method.evaluation.bestSeconds;
		std::println("    {:<9}{:>8.2f}x{:>13.1f}{:>13.1f}{:>20}{:>13.1e}", method.name, speedup, method.seconds * 1e6, method.kilobytes, paysOffAfter(direct, method, count), method.difference);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One block: the four methods, then the precomputation of each table method. Totals, each with the
	// 1 / h array: basis = the basis table; curve = the curve table (it builds its basis table inside);
	// lookup = the curve table plus the lookup table.
	template <std::floating_point T>
	MethodTimes compareMethods(std::string_view type, int degree, std::size_t numKnots, std::size_t count)
	{
		MethodTimes const c = measureMethods<T>(degree, numKnots, count);
		auto const p = static_cast<std::size_t>(degree);
		std::size_t const numSpans = numKnots - 2 * p - 1;
		double const basisKilobytes  = static_cast<double>(numSpans * ((p + 1) * (p + 1) + 1) * sizeof(T)) / 1024.0;
		double const curveKilobytes  = static_cast<double>(numSpans * (4 * (p + 1) + 1) * sizeof(T)) / 1024.0;
		double const lookupKilobytes = static_cast<double>(2 * numSpans * sizeof(std::uint32_t)) / 1024.0;
		std::println();
		std::println("  {:<10}{:>10}{:>13}{:>16}{:>12}", type, "best [ms]", "median [ms]", "per point [ns]", "Mpoints/s");
		printEvaluationRow("direct", c.direct, count);
		printEvaluationRow("basis", c.basis, count);
		printEvaluationRow("curve", c.curve, count);
		printEvaluationRow("lookup", c.lookup, count);
		std::println("    {:<9}{:>9}{:>13}{:>13}{:>20}{:>13}", "", "speedup", "precomp [us]", "memory [KB]", "pays off after", "max diff");
		printPrecomputation(c.direct, {"basis", c.basis, c.basisBuildSeconds + c.invHBuildSeconds, basisKilobytes, c.basisDifference}, count);
		printPrecomputation(c.direct, {"curve", c.curve, c.curveBuildSeconds + c.invHBuildSeconds, curveKilobytes, c.curveDifference}, count);
		printPrecomputation(c.direct, {"lookup", c.lookup, c.curveBuildSeconds + c.lookupBuildSeconds + c.invHBuildSeconds, curveKilobytes + lookupKilobytes, c.lookupDifference}, count);
		return c;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The checksums only make sure the compiler cannot skip the work; all four should match.
	void printChecksums(MethodTimes const& floats, MethodTimes const& doubles)
	{
		std::println();
		std::println("  checksums  float  {:.6e} (direct)  {:.6e} (basis)  {:.6e} (curve)  {:.6e} (lookup)", floats.direct.checksum, floats.basis.checksum, floats.curve.checksum, floats.lookup.checksum);
		std::println("             double {:.6e} (direct)  {:.6e} (basis)  {:.6e} (curve)  {:.6e} (lookup)", doubles.direct.checksum, doubles.basis.checksum, doubles.curve.checksum, doubles.lookup.checksum);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
#ifdef GEOM_HAS_CUDA
	// Each printed row compares the same work on the CPU and the GPU:
	//   Rebuild:    all memory (host and device) allocated before the clock starts, as in a program that
	//               rebuilds a table into reused memory; the CPU uses fillBasisTable / fillCurveTable;
	//   Build once: everything allocated inside the timed region, as for a table built once; the CPU uses
	//               the constructors, the GPU cudaMallocs its buffers and downloads into new host vectors;
	// the GPU either for the computation only (inputs already on the device) or with the copies to and from
	// CPU memory. The resulting table is never freed in a timed region (it is what the caller keeps), but
	// a build once's temporaries are: the CPU CurveTable(curve) frees its BasisTable inside, the GPU incl.
	// copies cudaFrees all its device buffers (the result is the downloaded host table). Allocating large
	// host memory costs page faults that depend on what the program allocated and freed before (glibc), so
	// build once varies more.
	
	// The same work timed on the CPU and the GPU, in seconds: one printed row.
	struct CpuGpu
	{
		double cpuSeconds{};
		double gpuSeconds{};
	};
	
	// The basis table rows of one type (bestOf, except the first build), in the printed order.
	struct BasisTableTimes
	{
		CpuGpu rebuildCompute{};        // CPU fillBasisTable vs the kernel alone (CUDA events)
		CpuGpu rebuildWithCopies{};     // CPU fillBasisTable vs upload knots, kernel, download into existing memory
		CpuGpu buildOnceCompute{};      // CPU constructor vs buildBasisTableOnDevice until done (knots on the device)
		CpuGpu buildOnceWithCopies{};   // CPU constructor vs allocate and upload knots, build, download into new vectors
		CpuGpu firstBuild{};            // the same, the type's first time (float: also starts CUDA)
		double difference{};            // largest |GPU - CPU| over all coefficients
	};
	
	// The curve table rows of one type (bestOf, except the first build), in the printed order. The curve
	// kernel only does stage 2 (basis table -> curve table), so stage 2 has its own rows. The whole job
	// is what a query needs, so it also builds the 1 / h array.
	struct CurveTableTimes
	{
		CpuGpu stage2RebuildCompute{};     // CPU fillCurveTable vs fillCurveTableOnDevice (the kernel alone, CUDA events)
		CpuGpu stage2BuildOnceCompute{};   // CPU CurveTable(basis, points) vs buildCurveTableOnDevice until done
		CpuGpu rebuildWithCopies{};        // CPU the three fills vs upload knots and points, three kernels, download
		CpuGpu buildOnceWithCopies{};      // CPU CurveTable(curve) and 1 / h vs allocate and upload, three builds, download into new vectors
		CpuGpu firstBuild{};               // the same, the type's first time (loads the curve and 1 / h kernels)
		double difference{};               // largest |GPU - CPU| over all coefficient components
	};
	
	// The rows of the 1 / h array of one type (bestOf), in the printed order.
	struct InverseSpanLengthTimes
	{
		CpuGpu rebuildCompute{};        // CPU fillInverseSpanLengths vs the kernel alone (CUDA events)
		CpuGpu rebuildWithCopies{};     // CPU fillInverseSpanLengths vs upload knots, kernel, download
		CpuGpu buildOnceCompute{};      // CPU buildInverseSpanLengths vs buildInverseSpanLengthsOnDevice until done (knots on the device)
		CpuGpu buildOnceWithCopies{};   // CPU buildInverseSpanLengths vs allocate and upload knots, build, download into a new vector
	};
	
	// The span lookup rows of one type (bestOf, except the first build), in the printed order.
	struct SpanLookupTimes
	{
		CpuGpu rebuildCompute{};        // CPU fillSpanLookup vs the kernel alone (CUDA events)
		CpuGpu rebuildWithCopies{};     // CPU fillSpanLookup vs upload knots, kernel, download into existing memory
		CpuGpu buildOnceCompute{};      // CPU constructor vs buildSpanLookupOnDevice until done (knots on the device)
		CpuGpu buildOnceWithCopies{};   // CPU constructor vs allocate and upload knots, build, download into a new vector
		CpuGpu firstBuild{};            // the same, the type's first time (loads the lookup kernel)
		std::size_t differingCells{};   // cells whose span differs from the CPU's; should be 0
	};
	
	// What the whole curve job leaves in host memory: the curve table and the 1 / h array.
	template <std::floating_point T>
	struct HostCurveResult
	{
		std::vector<geom::Point4<T>> coefficients{};
		std::vector<T> inverseSpanLengths{};
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Basis table, rebuild, CPU: into existing memory.
	template <std::floating_point T>
	double timeCpuBasisRebuild(geom::NurbsCurve<T> const& curve, std::vector<T>& host)
	{
		Clock::time_point const start = Clock::now();
		geom::fillBasisTable<T>(curve.degree(), curve.knots(), host);
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Basis table, rebuild incl. copies, GPU: new knots into the existing device buffer, the kernel into
	// the existing device table, download into existing host memory.
	template <std::floating_point T>
	double timeGpuBasisRebuildWithCopies(std::span<T const> knots, gpu::DeviceBuffer<T>& deviceKnots, gpu::DeviceBasisTable<T>& table, std::vector<T>& host)
	{
		Clock::time_point const start = Clock::now();
		deviceKnots.upload(knots);
		gpu::fillBasisTableOnDevice<T>(deviceKnots, table);
		table.coefficients.download(std::span<T>(host));
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A basis table built on the GPU and downloaded into new host vectors, as a one-off build would: allocate
	// and upload the knots, build, download. The device buffers are temporaries, freed (cudaFree) on return.
	template <std::floating_point T>
	std::vector<T> buildBasisTableViaGpu(int degree, std::span<T const> knots)
	{
		gpu::DeviceBuffer<T> const deviceKnots{knots};
		gpu::DeviceBasisTable<T> const table = gpu::buildBasisTableOnDevice<T>(degree, deviceKnots);
		return table.coefficients.download();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Basis table, build once incl. copies, GPU: buildBasisTableViaGpu, the free of its device temporaries
	// included. The result is not freed in the timed region: the previous run's is freed before the clock
	// starts (a single build has no previous result to free), this run's is kept in result.
	template <std::floating_point T>
	double timeGpuBasisBuildOnceWithCopies(int degree, std::span<T const> knots, std::vector<T>& result)
	{
		result = {};
		Clock::time_point const start = Clock::now();
		result = buildBasisTableViaGpu<T>(degree, knots);
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// buildBasisTableOnDevice with the knots already on the GPU. The launch returns at once, so the clock
	// is read only after cudaDeviceSynchronize, which waits until the GPU is done.
	template <std::floating_point T>
	double timeGpuBasisBuildOnceCompute(int degree, gpu::DeviceBuffer<T> const& knots)
	{
		Clock::time_point const start = Clock::now();
		gpu::DeviceBasisTable<T> const table = gpu::buildBasisTableOnDevice<T>(degree, knots);
		gpu::checkCuda(cudaDeviceSynchronize());
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The kernel alone: events directly around the launch, into an allocated table, so no allocation
	// (which would leave the GPU idle between the events) is measured.
	template <std::floating_point T>
	double timeGpuBasisRebuildCompute(gpu::DeviceBuffer<T> const& knots, gpu::DeviceBasisTable<T>& table)
	{
		gpu::CudaEvent start{};
		gpu::CudaEvent stop{};
		start.record();
		gpu::fillBasisTableOnDevice<T>(knots, table);
		stop.record();
		return gpu::elapsedSeconds(start, stop);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	double maxDifference(std::span<T const> a, std::span<T const> b)
	{
		double result = 0.0;
		for (std::size_t k = 0; k < a.size(); ++k)
		{
			result = std::max(result, std::abs(static_cast<double>(a[k]) - static_cast<double>(b[k])));
		}
		return result;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The build once rows. The first build comes first, so that for the first type it also pays for starting
	// CUDA. The GPU's last result is compared with cpuTable.
	template <std::floating_point T>
	void measureBasisBuildOnce(geom::NurbsCurve<T> const& curve, geom::BasisTable<T> const& cpuTable, BasisTableTimes& t)
	{
		std::vector<T> result{};
		t.firstBuild.gpuSeconds = timeGpuBasisBuildOnceWithCopies<T>(curve.degree(), curve.knots(), result);
		double const cpuSeconds = bestOf([&] { return timeCpuBuildOnce<geom::BasisTable<T>>(curve); });
		gpu::DeviceBuffer<T> const deviceKnots{curve.knots()};
		t.buildOnceCompute = {cpuSeconds, bestOf([&] { return timeGpuBasisBuildOnceCompute<T>(curve.degree(), deviceKnots); })};
		t.buildOnceWithCopies = {cpuSeconds, bestOf([&] { return timeGpuBasisBuildOnceWithCopies<T>(curve.degree(), curve.knots(), result); })};
		t.firstBuild.cpuSeconds = cpuSeconds;
		t.difference = std::max(t.difference, maxDifference<T>(cpuTable.coefficients(), result));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The rebuild rows, with all host and device memory allocated here, before timing. The GPU's last result
	// is compared with cpuTable.
	template <std::floating_point T>
	void measureBasisRebuild(geom::NurbsCurve<T> const& curve, geom::BasisTable<T> const& cpuTable, BasisTableTimes& t)
	{
		std::vector<T> host(cpuTable.coefficients().size());
		double const cpuSeconds = bestOf([&] { return timeCpuBasisRebuild<T>(curve, host); });
		gpu::DeviceBuffer<T> deviceKnots{curve.knots()};
		gpu::DeviceBasisTable<T> table = gpu::allocateBasisTableOnDevice<T>(curve.degree(), curve.knots().size());
		t.rebuildCompute = {cpuSeconds, bestOf([&] { return timeGpuBasisRebuildCompute<T>(deviceKnots, table); })};
		t.rebuildWithCopies = {cpuSeconds, bestOf([&] { return timeGpuBasisRebuildWithCopies<T>(curve.knots(), deviceKnots, table, host); })};
		t.difference = std::max(t.difference, maxDifference<T>(cpuTable.coefficients(), host));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Build once first: it times the first build. cpuTable is the reference the GPU results are compared with.
	template <std::floating_point T>
	BasisTableTimes measureBasisTable(int degree, std::size_t numKnots)
	{
		geom::NurbsCurve<T> const curve = makeBenchmarkCurve<T>(degree, numKnots);
		geom::BasisTable<T> const cpuTable(curve);
		BasisTableTimes t{};
		measureBasisBuildOnce<T>(curve, cpuTable, t);
		measureBasisRebuild<T>(curve, cpuTable, t);
		return t;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Curve table stage 2, build once, CPU: the CurveTable constructor from a basis table built before.
	template <std::floating_point T>
	double timeCpuCurveStage2BuildOnce(geom::BasisTable<T> const& basis, std::span<geom::Point4<T> const> weightedPoints)
	{
		Clock::time_point const start = Clock::now();
		geom::CurveTable<T> const table(basis, weightedPoints);
		double const seconds = secondsSince(start);
		checksumSink = checksumSink + sinkValue(table);
		return seconds;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Curve table stage 2, rebuild, CPU: fillCurveTable from a basis table built before, into existing memory.
	template <std::floating_point T>
	double timeCpuCurveStage2Rebuild(geom::NurbsCurve<T> const& curve, geom::BasisTable<T> const& basis, std::vector<geom::Point4<T>>& host)
	{
		Clock::time_point const start = Clock::now();
		geom::fillCurveTable<T>(curve.degree(), basis.coefficients(), curve.weightedControlPoints(), host);
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Curve table, whole job, rebuild, CPU: 1 / h and both tables into existing memory (the basis table
	// into basisHost).
	template <std::floating_point T>
	double timeCpuCurveRebuild(geom::NurbsCurve<T> const& curve, std::vector<T>& basisHost, HostCurveResult<T>& host)
	{
		Clock::time_point const start = Clock::now();
		geom::fillInverseSpanLengths<T>(curve.degree(), curve.knots(), host.inverseSpanLengths);
		geom::fillBasisTable<T>(curve.degree(), curve.knots(), basisHost);
		geom::fillCurveTable<T>(curve.degree(), basisHost, curve.weightedControlPoints(), host.coefficients);
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Curve table, whole job, build once, CPU: the CurveTable constructor from the curve and the 1 / h array.
	template <std::floating_point T>
	double timeCpuCurveBuildOnce(geom::NurbsCurve<T> const& curve)
	{
		Clock::time_point const start = Clock::now();
		std::vector<T> const inverseSpanLengths = geom::buildInverseSpanLengths<T>(curve.degree(), curve.knots());
		geom::CurveTable<T> const table(curve);
		double const seconds = secondsSince(start);
		checksumSink = checksumSink + sinkValue(table) + static_cast<double>(inverseSpanLengths.back());
		return seconds;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Curve table, whole job, rebuild incl. copies, GPU: new knots and points into the existing device
	// buffers, the three kernels into the existing device memory, download into existing host memory.
	template <std::floating_point T>
	double timeGpuCurveRebuildWithCopies(geom::NurbsCurve<T> const& curve, gpu::DeviceBuffer<T>& knots, gpu::DeviceBuffer<geom::Point4<T>>& points, gpu::DeviceBuffer<T>& inverseSpanLengths, gpu::DeviceBasisTable<T>& basis, gpu::DeviceCurveTable<T>& table, HostCurveResult<T>& host)
	{
		Clock::time_point const start = Clock::now();
		knots.upload(curve.knots());
		points.upload(curve.weightedControlPoints());
		gpu::fillInverseSpanLengthsOnDevice<T>(curve.degree(), knots, inverseSpanLengths);
		gpu::fillBasisTableOnDevice<T>(knots, basis);
		gpu::fillCurveTableOnDevice<T>(basis, points, table);
		table.coefficients.download(std::span<geom::Point4<T>>(host.coefficients));
		inverseSpanLengths.download(std::span<T>(host.inverseSpanLengths));
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A curve table and 1 / h array built on the GPU and downloaded into new host vectors, as a one-off build
	// would: allocate and upload knots and points, the three builds, download. The device buffers are
	// temporaries, freed (cudaFree) on return.
	template <std::floating_point T>
	HostCurveResult<T> buildCurveTableViaGpu(geom::NurbsCurve<T> const& curve)
	{
		gpu::DeviceBuffer<T> const deviceKnots{curve.knots()};
		gpu::DeviceBuffer<geom::Point4<T>> const devicePoints{curve.weightedControlPoints()};
		gpu::DeviceBuffer<T> const inverseSpanLengths = gpu::buildInverseSpanLengthsOnDevice<T>(curve.degree(), deviceKnots);
		gpu::DeviceBasisTable<T> const basis = gpu::buildBasisTableOnDevice<T>(curve.degree(), deviceKnots);
		gpu::DeviceCurveTable<T> const table = gpu::buildCurveTableOnDevice<T>(basis, devicePoints);
		return {table.coefficients.download(), inverseSpanLengths.download()};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Curve table, build once incl. copies, GPU: buildCurveTableViaGpu, the free of its device temporaries
	// included, the result's not (see timeGpuBasisBuildOnceWithCopies).
	template <std::floating_point T>
	double timeGpuCurveBuildOnceWithCopies(geom::NurbsCurve<T> const& curve, HostCurveResult<T>& result)
	{
		result = {};
		Clock::time_point const start = Clock::now();
		result = buildCurveTableViaGpu<T>(curve);
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// buildCurveTableOnDevice with the basis table and points already on the GPU, until the GPU is done.
	template <std::floating_point T>
	double timeGpuCurveStage2BuildOnceCompute(gpu::DeviceBasisTable<T> const& basis, gpu::DeviceBuffer<geom::Point4<T>> const& weightedPoints)
	{
		Clock::time_point const start = Clock::now();
		gpu::DeviceCurveTable<T> const table = gpu::buildCurveTableOnDevice<T>(basis, weightedPoints);
		gpu::checkCuda(cudaDeviceSynchronize());
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The kernel alone (fillCurveTableOnDevice into an allocated table), with events around the launch.
	template <std::floating_point T>
	double timeGpuCurveStage2RebuildCompute(gpu::DeviceBasisTable<T> const& basis, gpu::DeviceBuffer<geom::Point4<T>> const& weightedPoints, gpu::DeviceCurveTable<T>& table)
	{
		gpu::CudaEvent start{};
		gpu::CudaEvent stop{};
		start.record();
		gpu::fillCurveTableOnDevice<T>(basis, weightedPoints, table);
		stop.record();
		return gpu::elapsedSeconds(start, stop);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The largest |a - b| over the four components x, y, z, w of all points.
	template <std::floating_point T>
	double maxComponentDifference(std::span<geom::Point4<T> const> a, std::span<geom::Point4<T> const> b)
	{
		auto const difference = [](T u, T v) { return std::abs(static_cast<double>(u) - static_cast<double>(v)); };
		double result = 0.0;
		for (std::size_t k = 0; k < a.size(); ++k)
		{
			result = std::max({result, difference(a[k].x, b[k].x), difference(a[k].y, b[k].y), difference(a[k].z, b[k].z), difference(a[k].w, b[k].w)});
		}
		return result;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The GPU build once rows. The first build comes first (it loads the curve kernel). The GPU's last result
	// is compared with cpuTable.
	template <std::floating_point T>
	void measureCurveGpuBuildOnce(geom::NurbsCurve<T> const& curve, geom::CurveTable<T> const& cpuTable, CurveTableTimes& t)
	{
		HostCurveResult<T> result{};
		t.firstBuild.gpuSeconds = timeGpuCurveBuildOnceWithCopies<T>(curve, result);
		t.buildOnceWithCopies.gpuSeconds = bestOf([&] { return timeGpuCurveBuildOnceWithCopies<T>(curve, result); });
		t.difference = std::max(t.difference, maxComponentDifference<T>(cpuTable.coefficients(), result.coefficients));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// All CPU rows. The rebuild rows write the curve table (sized like cpuTable), the 1 / h array and the
	// intermediate basis table into memory allocated here, before timing.
	template <std::floating_point T>
	void measureCurveCpu(geom::NurbsCurve<T> const& curve, geom::BasisTable<T> const& basis, geom::CurveTable<T> const& cpuTable, CurveTableTimes& t)
	{
		HostCurveResult<T> host{std::vector<geom::Point4<T>>(cpuTable.coefficients().size()), std::vector<T>(cpuTable.numSpans())};
		std::vector<T> basisHost(basis.coefficients().size());
		t.stage2RebuildCompute.cpuSeconds = bestOf([&] { return timeCpuCurveStage2Rebuild<T>(curve, basis, host.coefficients); });
		t.stage2BuildOnceCompute.cpuSeconds = bestOf([&] { return timeCpuCurveStage2BuildOnce<T>(basis, curve.weightedControlPoints()); });
		t.rebuildWithCopies.cpuSeconds = bestOf([&] { return timeCpuCurveRebuild<T>(curve, basisHost, host); });
		t.buildOnceWithCopies.cpuSeconds = bestOf([&] { return timeCpuCurveBuildOnce<T>(curve); });
		t.firstBuild.cpuSeconds = t.buildOnceWithCopies.cpuSeconds;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The GPU rows that start from device memory allocated here: both stage 2 rows and the rebuild with copies.
	// The basis table is built once and waited for, so that its kernel does not run into the first timed
	// build. The rebuild downloads into host memory allocated here; its last result is compared with cpuTable.
	template <std::floating_point T>
	void measureCurveGpuFromDevice(geom::NurbsCurve<T> const& curve, geom::CurveTable<T> const& cpuTable, CurveTableTimes& t)
	{
		HostCurveResult<T> host{std::vector<geom::Point4<T>>(cpuTable.coefficients().size()), std::vector<T>(cpuTable.numSpans())};
		gpu::DeviceBuffer<T> deviceKnots{curve.knots()};
		gpu::DeviceBuffer<T> deviceInverseSpanLengths(cpuTable.numSpans());
		gpu::DeviceBuffer<geom::Point4<T>> devicePoints{curve.weightedControlPoints()};
		gpu::DeviceBasisTable<T> basis = gpu::buildBasisTableOnDevice<T>(curve.degree(), deviceKnots);
		gpu::checkCuda(cudaDeviceSynchronize());
		t.stage2BuildOnceCompute.gpuSeconds = bestOf([&] { return timeGpuCurveStage2BuildOnceCompute<T>(basis, devicePoints); });
		gpu::DeviceCurveTable<T> table = gpu::allocateCurveTableOnDevice<T>(curve.degree(), basis.numSpans);
		t.stage2RebuildCompute.gpuSeconds = bestOf([&] { return timeGpuCurveStage2RebuildCompute<T>(basis, devicePoints, table); });
		t.rebuildWithCopies.gpuSeconds = bestOf([&] { return timeGpuCurveRebuildWithCopies<T>(curve, deviceKnots, devicePoints, deviceInverseSpanLengths, basis, table, host); });
		t.difference = std::max(t.difference, maxComponentDifference<T>(cpuTable.coefficients(), host.coefficients));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The GPU build once first: it times the first build. cpuTable is the reference the GPU results are
	// compared with.
	template <std::floating_point T>
	CurveTableTimes measureCurveTable(int degree, std::size_t numKnots)
	{
		geom::NurbsCurve<T> const curve = makeBenchmarkCurve<T>(degree, numKnots);
		geom::BasisTable<T> const cpuBasis(curve);
		geom::CurveTable<T> const cpuTable(cpuBasis, curve.weightedControlPoints());
		CurveTableTimes t{};
		measureCurveGpuBuildOnce<T>(curve, cpuTable, t);
		measureCurveCpu<T>(curve, cpuBasis, cpuTable, t);
		measureCurveGpuFromDevice<T>(curve, cpuTable, t);
		return t;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// 1 / h, rebuild, CPU: into existing memory.
	template <std::floating_point T>
	double timeCpuInverseSpanLengthsRebuild(geom::NurbsCurve<T> const& curve, std::vector<T>& host)
	{
		Clock::time_point const start = Clock::now();
		geom::fillInverseSpanLengths<T>(curve.degree(), curve.knots(), host);
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The 1 / h kernel alone: events directly around the launch, into an allocated buffer.
	template <std::floating_point T>
	double timeGpuInverseSpanLengthsRebuildCompute(int degree, gpu::DeviceBuffer<T> const& knots, gpu::DeviceBuffer<T>& inverseSpanLengths)
	{
		gpu::CudaEvent start{};
		gpu::CudaEvent stop{};
		start.record();
		gpu::fillInverseSpanLengthsOnDevice<T>(degree, knots, inverseSpanLengths);
		stop.record();
		return gpu::elapsedSeconds(start, stop);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// 1 / h, rebuild incl. copies, GPU: new knots into the existing device buffer, the kernel, download
	// into existing host memory.
	template <std::floating_point T>
	double timeGpuInverseSpanLengthsRebuildWithCopies(geom::NurbsCurve<T> const& curve, gpu::DeviceBuffer<T>& knots, gpu::DeviceBuffer<T>& inverseSpanLengths, std::vector<T>& host)
	{
		Clock::time_point const start = Clock::now();
		knots.upload(curve.knots());
		gpu::fillInverseSpanLengthsOnDevice<T>(curve.degree(), knots, inverseSpanLengths);
		inverseSpanLengths.download(std::span<T>(host));
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// buildInverseSpanLengthsOnDevice with the knots already on the GPU, until the GPU is done.
	template <std::floating_point T>
	double timeGpuInverseSpanLengthsBuildOnceCompute(int degree, gpu::DeviceBuffer<T> const& knots)
	{
		Clock::time_point const start = Clock::now();
		gpu::DeviceBuffer<T> const inverseSpanLengths = gpu::buildInverseSpanLengthsOnDevice<T>(degree, knots);
		gpu::checkCuda(cudaDeviceSynchronize());
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The 1 / h array built on the GPU and downloaded into a new host vector, as a one-off build would. The
	// device buffers are temporaries, freed (cudaFree) on return.
	template <std::floating_point T>
	std::vector<T> buildInverseSpanLengthsViaGpu(int degree, std::span<T const> knots)
	{
		gpu::DeviceBuffer<T> const deviceKnots{knots};
		gpu::DeviceBuffer<T> const inverseSpanLengths = gpu::buildInverseSpanLengthsOnDevice<T>(degree, deviceKnots);
		return inverseSpanLengths.download();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// 1 / h, build once incl. copies, GPU: the result is not freed in the timed region (see
	// timeGpuBasisBuildOnceWithCopies).
	template <std::floating_point T>
	double timeGpuInverseSpanLengthsBuildOnceWithCopies(int degree, std::span<T const> knots, std::vector<T>& result)
	{
		result = {};
		Clock::time_point const start = Clock::now();
		result = buildInverseSpanLengthsViaGpu<T>(degree, knots);
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The build once rows of the 1 / h array.
	template <std::floating_point T>
	void measureInverseSpanLengthsBuildOnce(geom::NurbsCurve<T> const& curve, InverseSpanLengthTimes& t)
	{
		double const cpuSeconds = bestOf([&] { return timeCpuInverseSpanLengthsBuildOnce<T>(curve); });
		gpu::DeviceBuffer<T> const deviceKnots{curve.knots()};
		std::vector<T> result{};
		t.buildOnceCompute = {cpuSeconds, bestOf([&] { return timeGpuInverseSpanLengthsBuildOnceCompute<T>(curve.degree(), deviceKnots); })};
		t.buildOnceWithCopies = {cpuSeconds, bestOf([&] { return timeGpuInverseSpanLengthsBuildOnceWithCopies<T>(curve.degree(), curve.knots(), result); })};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The rows of the 1 / h array. The rebuild rows have all host and device memory allocated here, before timing.
	template <std::floating_point T>
	InverseSpanLengthTimes measureInverseSpanLengths(int degree, std::size_t numKnots)
	{
		geom::NurbsCurve<T> const curve = makeBenchmarkCurve<T>(degree, numKnots);
		std::vector<T> host(numKnots - 2 * static_cast<std::size_t>(degree) - 1);
		double const cpuSeconds = bestOf([&] { return timeCpuInverseSpanLengthsRebuild<T>(curve, host); });
		gpu::DeviceBuffer<T> deviceKnots{curve.knots()};
		gpu::DeviceBuffer<T> inverseSpanLengths(host.size());
		InverseSpanLengthTimes t{};
		t.rebuildCompute = {cpuSeconds, bestOf([&] { return timeGpuInverseSpanLengthsRebuildCompute<T>(degree, deviceKnots, inverseSpanLengths); })};
		t.rebuildWithCopies = {cpuSeconds, bestOf([&] { return timeGpuInverseSpanLengthsRebuildWithCopies<T>(curve, deviceKnots, inverseSpanLengths, host); })};
		measureInverseSpanLengthsBuildOnce<T>(curve, t);
		return t;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Span lookup, rebuild, CPU: into existing memory.
	template <std::floating_point T>
	double timeCpuSpanLookupRebuild(geom::NurbsCurve<T> const& curve, geom::SpanLookup<T> const& cpuLookup, std::vector<std::uint32_t>& host)
	{
		Clock::time_point const start = Clock::now();
		geom::fillSpanLookup<T>(curve.degree(), curve.knots(), cpuLookup.start(), cpuLookup.cellsPerUnit(), host);
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The lookup kernel alone: events directly around the launch, into an allocated table.
	template <std::floating_point T>
	double timeGpuSpanLookupRebuildCompute(gpu::DeviceBuffer<T> const& knots, gpu::DeviceSpanLookup<T>& lookup)
	{
		gpu::CudaEvent start{};
		gpu::CudaEvent stop{};
		start.record();
		gpu::fillSpanLookupOnDevice<T>(knots, lookup);
		stop.record();
		return gpu::elapsedSeconds(start, stop);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Span lookup, rebuild incl. copies, GPU: new knots into the existing device buffer, the kernel into
	// the existing device table, download into existing host memory.
	template <std::floating_point T>
	double timeGpuSpanLookupRebuildWithCopies(std::span<T const> knots, gpu::DeviceBuffer<T>& deviceKnots, gpu::DeviceSpanLookup<T>& lookup, std::vector<std::uint32_t>& host)
	{
		Clock::time_point const start = Clock::now();
		deviceKnots.upload(knots);
		gpu::fillSpanLookupOnDevice<T>(deviceKnots, lookup);
		lookup.firstSpans.download(std::span<std::uint32_t>(host));
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A lookup table built on the GPU and downloaded into a new host vector, as a one-off build would: allocate
	// and upload the knots, build, download. The device buffers are temporaries, freed (cudaFree) on return.
	template <std::floating_point T>
	std::vector<std::uint32_t> buildSpanLookupViaGpu(geom::NurbsCurve<T> const& curve)
	{
		auto const [domainStart, domainEnd] = curve.domain();
		gpu::DeviceBuffer<T> const deviceKnots{curve.knots()};
		gpu::DeviceSpanLookup<T> const lookup = gpu::buildSpanLookupOnDevice<T>(curve.degree(), deviceKnots, domainStart, domainEnd);
		return lookup.firstSpans.download();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Span lookup, build once incl. copies, GPU: the result is not freed in the timed region (see
	// timeGpuBasisBuildOnceWithCopies).
	template <std::floating_point T>
	double timeGpuSpanLookupBuildOnceWithCopies(geom::NurbsCurve<T> const& curve, std::vector<std::uint32_t>& result)
	{
		result = {};
		Clock::time_point const start = Clock::now();
		result = buildSpanLookupViaGpu<T>(curve);
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// buildSpanLookupOnDevice with the knots already on the GPU, until the GPU is done.
	template <std::floating_point T>
	double timeGpuSpanLookupBuildOnceCompute(geom::NurbsCurve<T> const& curve, gpu::DeviceBuffer<T> const& knots)
	{
		auto const [domainStart, domainEnd] = curve.domain();
		Clock::time_point const start = Clock::now();
		gpu::DeviceSpanLookup<T> const lookup = gpu::buildSpanLookupOnDevice<T>(curve.degree(), knots, domainStart, domainEnd);
		gpu::checkCuda(cudaDeviceSynchronize());
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// How many cells differ between two lookup tables.
	std::size_t countDifferingCells(std::span<std::uint32_t const> a, std::span<std::uint32_t const> b)
	{
		std::size_t count = 0;
		for (std::size_t k = 0; k < a.size(); ++k)
		{
			count += (a[k] != b[k]) ? 1 : 0;
		}
		return count;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The build once rows. The first build comes first, so that it pays for loading the lookup kernel. The
	// GPU's last result is compared with cpuLookup.
	template <std::floating_point T>
	void measureSpanLookupBuildOnce(geom::NurbsCurve<T> const& curve, geom::SpanLookup<T> const& cpuLookup, SpanLookupTimes& t)
	{
		std::vector<std::uint32_t> result{};
		t.firstBuild.gpuSeconds = timeGpuSpanLookupBuildOnceWithCopies<T>(curve, result);
		double const cpuSeconds = bestOf([&] { return timeCpuBuildOnce<geom::SpanLookup<T>>(curve); });
		gpu::DeviceBuffer<T> const deviceKnots{curve.knots()};
		t.buildOnceCompute = {cpuSeconds, bestOf([&] { return timeGpuSpanLookupBuildOnceCompute<T>(curve, deviceKnots); })};
		t.buildOnceWithCopies = {cpuSeconds, bestOf([&] { return timeGpuSpanLookupBuildOnceWithCopies<T>(curve, result); })};
		t.firstBuild.cpuSeconds = cpuSeconds;
		t.differingCells = t.differingCells + countDifferingCells(cpuLookup.firstSpans(), result);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The rebuild rows, with all host and device memory allocated here, before timing. The GPU's last result
	// is compared with cpuLookup.
	template <std::floating_point T>
	void measureSpanLookupRebuild(geom::NurbsCurve<T> const& curve, geom::SpanLookup<T> const& cpuLookup, SpanLookupTimes& t)
	{
		auto const [domainStart, domainEnd] = curve.domain();
		std::vector<std::uint32_t> host(cpuLookup.firstSpans().size());
		double const cpuSeconds = bestOf([&] { return timeCpuSpanLookupRebuild<T>(curve, cpuLookup, host); });
		gpu::DeviceBuffer<T> deviceKnots{curve.knots()};
		gpu::DeviceSpanLookup<T> lookup = gpu::allocateSpanLookupOnDevice<T>(curve.degree(), curve.knots().size(), domainStart, domainEnd);
		t.rebuildCompute = {cpuSeconds, bestOf([&] { return timeGpuSpanLookupRebuildCompute<T>(deviceKnots, lookup); })};
		t.rebuildWithCopies = {cpuSeconds, bestOf([&] { return timeGpuSpanLookupRebuildWithCopies<T>(curve.knots(), deviceKnots, lookup, host); })};
		t.differingCells = t.differingCells + countDifferingCells(cpuLookup.firstSpans(), host);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Build once first: it times the first build. cpuLookup is the reference the GPU results are compared with.
	template <std::floating_point T>
	SpanLookupTimes measureSpanLookup(int degree, std::size_t numKnots)
	{
		geom::NurbsCurve<T> const curve = makeBenchmarkCurve<T>(degree, numKnots);
		geom::SpanLookup<T> const cpuLookup(curve);
		SpanLookupTimes t{};
		measureSpanLookupBuildOnce<T>(curve, cpuLookup, t);
		measureSpanLookupRebuild<T>(curve, cpuLookup, t);
		return t;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// How the GPU compares with the CPU, in words: "14x faster" or "1.9x slower".
	std::string gpuSpeedInWords(CpuGpu const& t)
	{
		return bench::speedInWords(t.cpuSeconds, t.gpuSeconds);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One comparison, the same work on the CPU and the GPU, for float and double side by side.
	void printComparison(std::string_view name, CpuGpu const& floats, CpuGpu const& doubles)
	{
		std::println("  {:<38}{:>9}{:>10}  {:<15}{:>9}{:>10}  {}", name, formatMicroseconds(floats.cpuSeconds), formatMicroseconds(floats.gpuSeconds), gpuSpeedInWords(floats), formatMicroseconds(doubles.cpuSeconds), formatMicroseconds(doubles.gpuSeconds), gpuSpeedInWords(doubles));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A section title and the column headers above its comparisons.
	void printSectionHeader(std::string_view title)
	{
		std::println();
		std::println("{}", title);
		std::println("  {:<38}{:-^34}  {:-^34}", "", " float ", " double ");
		std::println("  {:<38}{:>9}{:>10}  {:<15}{:>9}{:>10}  {}", "", "CPU", "GPU", "GPU is", "CPU", "GPU", "GPU is");
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The largest |GPU - CPU| coefficient, under the GPU columns.
	void printDifferences(double floats, double doubles)
	{
		std::println("  {:<38}{:>19}  {:<15}{:>19}", "Largest |GPU - CPU| coefficient", std::format("{:.1e}", floats), "", std::format("{:.1e}", doubles));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printCpuVsGpuHeader(int degree, std::size_t numKnots)
	{
		cudaDeviceProp properties{};
		gpu::checkCuda(cudaGetDeviceProperties(&properties, 0));
		std::size_t const numSpans = numKnots - 2 * static_cast<std::size_t>(degree) - 1;
		std::println();
		std::println("Table builds on the CPU (1 thread) vs the GPU ({}, sm_{}{})", properties.name, properties.major, properties.minor);
		std::println("  {} spans, degree {}. Times in µs: best of at least {} runs and {} s (first build: one run).", withThousands(numSpans), degree, kRuns, kMinSeconds);
		std::println("  Rebuild    = memory allocated beforehand and reused (e.g. a table rebuilt whenever the curve changes).");
		std::println("  Build once = the build allocates its own memory (CPU: constructor; GPU: cudaMalloc + new host vectors),\n"
		             "               and frees its temporaries (GPU incl. copies: all device buffers); the result is kept.");
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printBasisSection(BasisTableTimes const& f, BasisTableTimes const& d)
	{
		printSectionHeader("BASIS TABLE  (knots -> basis table)");
		printComparison("Rebuild, computation only", f.rebuildCompute, d.rebuildCompute);
		printComparison("Rebuild, incl. copies CPU <-> GPU", f.rebuildWithCopies, d.rebuildWithCopies);
		printComparison("Build once, computation only", f.buildOnceCompute, d.buildOnceCompute);
		printComparison("Build once, incl. copies CPU <-> GPU", f.buildOnceWithCopies, d.buildOnceWithCopies);
		printComparison("First build in this program", f.firstBuild, d.firstBuild);
		printDifferences(f.difference, d.difference);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printInverseSpanLengthsSection(InverseSpanLengthTimes const& f, InverseSpanLengthTimes const& d)
	{
		printSectionHeader("1 / H ARRAY  (knots -> 1 / h; one array per knot vector, shared by all its tables)");
		printComparison("Rebuild, computation only", f.rebuildCompute, d.rebuildCompute);
		printComparison("Rebuild, incl. copies CPU <-> GPU", f.rebuildWithCopies, d.rebuildWithCopies);
		printComparison("Build once, computation only", f.buildOnceCompute, d.buildOnceCompute);
		printComparison("Build once, incl. copies CPU <-> GPU", f.buildOnceWithCopies, d.buildOnceWithCopies);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printSpanLookupSection(SpanLookupTimes const& f, SpanLookupTimes const& d)
	{
		printSectionHeader("SPAN LOOKUP  (knots -> lookup table, 2 cells per span)");
		printComparison("Rebuild, computation only", f.rebuildCompute, d.rebuildCompute);
		printComparison("Rebuild, incl. copies CPU <-> GPU", f.rebuildWithCopies, d.rebuildWithCopies);
		printComparison("Build once, computation only", f.buildOnceCompute, d.buildOnceCompute);
		printComparison("Build once, incl. copies CPU <-> GPU", f.buildOnceWithCopies, d.buildOnceWithCopies);
		printComparison("First build in this program", f.firstBuild, d.firstBuild);
		std::println("  {:<38}{:>19}  {:<15}{:>19}", "Cells that differ from the CPU", f.differingCells, "", d.differingCells);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printCurveSections(CurveTableTimes const& f, CurveTableTimes const& d)
	{
		printSectionHeader("CURVE TABLE, STAGE 2 ONLY  (basis table -> curve table; basis table already built)");
		printComparison("Rebuild, computation only", f.stage2RebuildCompute, d.stage2RebuildCompute);
		printComparison("Build once, computation only", f.stage2BuildOnceCompute, d.stage2BuildOnceCompute);
		printSectionHeader("CURVE TABLE, WHOLE JOB  (curve: knots + control points -> curve table and 1 / h)");
		printComparison("Rebuild, incl. copies CPU <-> GPU", f.rebuildWithCopies, d.rebuildWithCopies);
		printComparison("Build once, incl. copies CPU <-> GPU", f.buildOnceWithCopies, d.buildOnceWithCopies);
		printComparison("First build in this program", f.firstBuild, d.firstBuild);
		printDifferences(f.difference, d.difference);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printCpuVsGpuLegend()
	{
		std::println();
		std::println("What the GPU column includes:");
		std::println("  computation only   the kernel(s) on data already on the GPU (build once: also cudaMalloc of the table)");
		std::println("  incl. copies       upload the inputs, the kernel(s), download the table into CPU memory");
		std::println("  First build        build once incl. copies, the first time: the program's first CUDA call also starts");
		std::println("                     CUDA (float basis table, ~0.2 s), every other first build loads its kernel");
		std::println("The basis table and stage 2 rows do not include 1 / h, on either side; the whole job does.");
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Measures everything before printing: asking for the device name is a CUDA call too, and would
	// create the context before float's first call. The 1 / h rows come after the curve rows, so that the whole job's
	// first build also pays for loading the 1 / h kernel.
	void compareCpuWithGpu(int degree, std::size_t numKnots)
	{
		BasisTableTimes const floats = measureBasisTable<float>(degree, numKnots);
		BasisTableTimes const doubles = measureBasisTable<double>(degree, numKnots);
		CurveTableTimes const curveFloats = measureCurveTable<float>(degree, numKnots);
		CurveTableTimes const curveDoubles = measureCurveTable<double>(degree, numKnots);
		InverseSpanLengthTimes const invHFloats = measureInverseSpanLengths<float>(degree, numKnots);
		InverseSpanLengthTimes const invHDoubles = measureInverseSpanLengths<double>(degree, numKnots);
		SpanLookupTimes const lookupFloats = measureSpanLookup<float>(degree, numKnots);
		SpanLookupTimes const lookupDoubles = measureSpanLookup<double>(degree, numKnots);
		printCpuVsGpuHeader(degree, numKnots);
		printBasisSection(floats, doubles);
		printCurveSections(curveFloats, curveDoubles);
		printInverseSpanLengthsSection(invHFloats, invHDoubles);
		printSpanLookupSection(lookupFloats, lookupDoubles);
		printCpuVsGpuLegend();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#endif
	
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
	printMethodsHeader(options.count, degree, options.numKnots);
	MethodTimes const floats  = compareMethods<float >("float" , degree, options.numKnots, options.count);
	MethodTimes const doubles = compareMethods<double>("double", degree, options.numKnots, options.count);
	printChecksums(floats, doubles);
#ifdef GEOM_HAS_CUDA
	compareCpuWithGpu(degree, options.numKnots);
#endif
	return 0;
}
