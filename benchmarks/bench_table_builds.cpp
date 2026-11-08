#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "bench_common.hpp"
#include "bench_table_builds.hpp"
#include "geom/basis_table.hpp"
#include "geom/curve_table.hpp"
#include "geom/inverse_span_lengths.hpp"
#include "geom/nurbs_curve.hpp"
#include "geom/span_lookup.hpp"
#include "gpu/basis_table_gpu.hpp"
#include "gpu/cuda_check.hpp"
#include "gpu/cuda_event.hpp"
#include "gpu/curve_table_gpu.hpp"
#include "gpu/device_buffer.hpp"
#include "gpu/inverse_span_lengths_gpu.hpp"
#include "gpu/span_lookup_gpu.hpp"

// Part of the bench_precompute executable: each table's build on one CPU thread against the GPU, one
// comparison (the same work on both, float and double) per printed row: as a rebuild (memory allocated
// beforehand: CPU fill functions) and as a build once (allocating inside: CPU constructors), the GPU with
// or without the copies to and from CPU memory, and the first build of each type (float's basis table runs
// before any other CUDA call, so it also shows the cost of starting CUDA). The printed legend explains the
// rows; results in docs/benchmarks.md.

namespace
{
	using bench::bestOf;
	using bench::checksumSink;
	using bench::Clock;
	using bench::formatMicroseconds;
	using bench::kMinSeconds;
	using bench::kRuns;
	using bench::makeBenchmarkCurve;
	using bench::secondsSince;
	using bench::sinkValue;
	using bench::timeCpuBuildOnce;
	using bench::timeCpuInverseSpanLengthsBuildOnce;
	using bench::withThousands;
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Each printed row compares the same work on the CPU and the GPU:
	//   Rebuild:    all memory (host and device) allocated before the clock starts, as in a program that
	//               rebuilds a table into reused memory; the CPU uses the fill functions;
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
	// kernel only does stage 2 (basis table -> curve table), so stage 2 has its own row.
	struct CurveTableTimes
	{
		CpuGpu stage2BuildOnceCompute{};   // CPU CurveTable(basis, points) vs buildCurveTableOnDevice until done
		CpuGpu buildOnceWithCopies{};      // CPU CurveTable(curve) vs allocate and upload, both builds, download into a new vector
		CpuGpu firstBuild{};               // the same, the type's first time (loads the curve kernel)
		double difference{};               // largest |GPU - CPU| over all coefficient components
	};
	
	// The rows of the 1 / h array of one type (bestOf), in the printed order.
	struct InverseSpanLengthTimes
	{
		CpuGpu rebuildCompute{};        // CPU fillInverseSpanLengths vs the kernel alone (CUDA events)
		CpuGpu rebuildWithCopies{};     // CPU fillInverseSpanLengths vs upload knots, kernel, download
		CpuGpu buildOnceCompute{};      // CPU buildInverseSpanLengths vs buildInverseSpanLengthsOnDevice until done (knots on the device)
		CpuGpu buildOnceWithCopies{};   // CPU buildInverseSpanLengths vs allocate and upload knots, build, download into a new vector
		CpuGpu firstBuild{};            // the same, the type's first time (loads the 1 / h kernel)
		std::size_t differingValues{};  // values that differ from the CPU's; should be 0
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
	
	// A curve table built on the GPU and downloaded into a new host vector, as a one-off build would: allocate
	// and upload knots and points, both builds, download. The device buffers (inputs, basis table, curve
	// table) are temporaries, freed (cudaFree) on return.
	template <std::floating_point T>
	std::vector<geom::Point4<T>> buildCurveTableViaGpu(geom::NurbsCurve<T> const& curve)
	{
		gpu::DeviceBuffer<T> const deviceKnots{curve.knots()};
		gpu::DeviceBuffer<geom::Point4<T>> const devicePoints{curve.weightedControlPoints()};
		gpu::DeviceBasisTable<T> const basis = gpu::buildBasisTableOnDevice<T>(curve.degree(), deviceKnots);
		gpu::DeviceCurveTable<T> const table = gpu::buildCurveTableOnDevice<T>(basis, devicePoints);
		return table.coefficients.download();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Curve table, build once incl. copies, GPU: buildCurveTableViaGpu, the free of its device temporaries
	// included, the result's not (see timeGpuBasisBuildOnceWithCopies).
	template <std::floating_point T>
	double timeGpuCurveBuildOnceWithCopies(geom::NurbsCurve<T> const& curve, std::vector<geom::Point4<T>>& result)
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
		std::vector<geom::Point4<T>> result{};
		t.firstBuild.gpuSeconds = timeGpuCurveBuildOnceWithCopies<T>(curve, result);
		t.buildOnceWithCopies.gpuSeconds = bestOf([&] { return timeGpuCurveBuildOnceWithCopies<T>(curve, result); });
		t.difference = std::max(t.difference, maxComponentDifference<T>(cpuTable.coefficients(), result));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// All CPU rows.
	template <std::floating_point T>
	void measureCurveCpu(geom::NurbsCurve<T> const& curve, geom::BasisTable<T> const& basis, CurveTableTimes& t)
	{
		t.stage2BuildOnceCompute.cpuSeconds = bestOf([&] { return timeCpuCurveStage2BuildOnce<T>(basis, curve.weightedControlPoints()); });
		t.buildOnceWithCopies.cpuSeconds = bestOf([&] { return timeCpuBuildOnce<geom::CurveTable<T>>(curve); });
		t.firstBuild.cpuSeconds = t.buildOnceWithCopies.cpuSeconds;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The stage 2 row, from a basis table and points already on the GPU. The basis table is built once and
	// waited for, so that its kernel does not run into the first timed build.
	template <std::floating_point T>
	void measureCurveGpuFromDevice(geom::NurbsCurve<T> const& curve, CurveTableTimes& t)
	{
		gpu::DeviceBuffer<T> const deviceKnots{curve.knots()};
		gpu::DeviceBuffer<geom::Point4<T>> const devicePoints{curve.weightedControlPoints()};
		gpu::DeviceBasisTable<T> const basis = gpu::buildBasisTableOnDevice<T>(curve.degree(), deviceKnots);
		gpu::checkCuda(cudaDeviceSynchronize());
		t.stage2BuildOnceCompute.gpuSeconds = bestOf([&] { return timeGpuCurveStage2BuildOnceCompute<T>(basis, devicePoints); });
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
		measureCurveCpu<T>(curve, cpuBasis, t);
		measureCurveGpuFromDevice<T>(curve, t);
		return t;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// How many values differ between two arrays of the same size.
	template <typename Value>
	std::size_t countDiffering(std::span<Value const> a, std::span<Value const> b)
	{
		std::size_t count = 0;
		for (std::size_t k = 0; k < a.size(); ++k)
		{
			count += (a[k] != b[k]) ? 1 : 0;
		}
		return count;
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
	
	// The build once rows of the 1 / h array. The first build comes first, so that it pays for loading the
	// kernel. The GPU's last result is compared with reference.
	template <std::floating_point T>
	void measureInverseSpanLengthsBuildOnce(geom::NurbsCurve<T> const& curve, std::span<T const> reference, InverseSpanLengthTimes& t)
	{
		std::vector<T> result{};
		t.firstBuild.gpuSeconds = timeGpuInverseSpanLengthsBuildOnceWithCopies<T>(curve.degree(), curve.knots(), result);
		double const cpuSeconds = bestOf([&] { return timeCpuInverseSpanLengthsBuildOnce<T>(curve); });
		gpu::DeviceBuffer<T> const deviceKnots{curve.knots()};
		t.buildOnceCompute = {cpuSeconds, bestOf([&] { return timeGpuInverseSpanLengthsBuildOnceCompute<T>(curve.degree(), deviceKnots); })};
		t.buildOnceWithCopies = {cpuSeconds, bestOf([&] { return timeGpuInverseSpanLengthsBuildOnceWithCopies<T>(curve.degree(), curve.knots(), result); })};
		t.firstBuild.cpuSeconds = cpuSeconds;
		t.differingValues = t.differingValues + countDiffering<T>(reference, result);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The rebuild rows of the 1 / h array, with all host and device memory allocated here, before timing. The
	// GPU's last result is compared with reference.
	template <std::floating_point T>
	void measureInverseSpanLengthsRebuild(geom::NurbsCurve<T> const& curve, std::span<T const> reference, InverseSpanLengthTimes& t)
	{
		std::vector<T> host(reference.size());
		double const cpuSeconds = bestOf([&] { return timeCpuInverseSpanLengthsRebuild<T>(curve, host); });
		gpu::DeviceBuffer<T> deviceKnots{curve.knots()};
		gpu::DeviceBuffer<T> inverseSpanLengths(host.size());
		t.rebuildCompute = {cpuSeconds, bestOf([&] { return timeGpuInverseSpanLengthsRebuildCompute<T>(curve.degree(), deviceKnots, inverseSpanLengths); })};
		t.rebuildWithCopies = {cpuSeconds, bestOf([&] { return timeGpuInverseSpanLengthsRebuildWithCopies<T>(curve, deviceKnots, inverseSpanLengths, host); })};
		t.differingValues = t.differingValues + countDiffering<T>(reference, host);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Build once first: it times the first build.
	template <std::floating_point T>
	InverseSpanLengthTimes measureInverseSpanLengths(int degree, std::size_t numKnots)
	{
		geom::NurbsCurve<T> const curve = makeBenchmarkCurve<T>(degree, numKnots);
		std::vector<T> const reference = geom::buildInverseSpanLengths<T>(degree, curve.knots());
		InverseSpanLengthTimes t{};
		measureInverseSpanLengthsBuildOnce<T>(curve, reference, t);
		measureInverseSpanLengthsRebuild<T>(curve, reference, t);
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
		t.differingCells = t.differingCells + countDiffering<std::uint32_t>(cpuLookup.firstSpans(), result);
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
		t.differingCells = t.differingCells + countDiffering<std::uint32_t>(cpuLookup.firstSpans(), host);
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
		std::println("  {} spans, degree {}. Times in µs: best of at least {} runs and {} s (first build: GPU one run, CPU as in build once).", withThousands(numSpans), degree, kRuns, kMinSeconds);
		std::println("  Rebuild    = memory allocated beforehand and reused (e.g. a table rebuilt whenever the curve changes).");
		std::println("  Build once = the build allocates its own memory (CPU: constructor, or buildInverseSpanLengths; GPU: cudaMalloc,\n"
		             "               and new host vectors incl. copies), and frees its temporaries (GPU incl. copies: all device buffers); the result is kept.");
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
		printComparison("First build in this program", f.firstBuild, d.firstBuild);
		std::println("  {:<38}{:>19}  {:<15}{:>19}", "Values that differ from the CPU", f.differingValues, "", d.differingValues);
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
		printComparison("Build once, computation only", f.stage2BuildOnceCompute, d.stage2BuildOnceCompute);
		printSectionHeader("CURVE TABLE, WHOLE JOB  (knots + control points -> basis table -> curve table)");
		printComparison("Build once, incl. copies CPU <-> GPU", f.buildOnceWithCopies, d.buildOnceWithCopies);
		printComparison("First build in this program", f.firstBuild, d.firstBuild);
		printDifferences(f.difference, d.difference);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printCpuVsGpuLegend()
	{
		std::println();
		std::println("What the GPU column includes:");
		std::println("  computation only   the kernel(s) on data already on the GPU. Rebuild: CUDA events around the launch;");
		std::println("                     build once: host clock until the GPU is done, incl. cudaMalloc of the result");
		std::println("  incl. copies       upload the inputs, the kernel(s), download the result into CPU memory (host clock)");
		std::println("  First build        build once incl. copies, the first time: the program's first CUDA call also starts");
		std::println("                     CUDA (float basis table, ~0.2 s), every other first build loads its kernel");
		std::println("The 1 / h array is measured only in its own section here; the precomputation below includes it.");
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

namespace bench
{
	// Measures everything before printing: asking for the device name is a CUDA call too, and would
	// create the context before float's first call.
	void compareTableBuilds(int degree, std::size_t numKnots)
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
}  // namespace bench
