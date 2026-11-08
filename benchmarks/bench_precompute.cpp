#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <limits>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <omp.h>

#include "bench_common.hpp"
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

// Benchmark: everything a curve needs before its first query (1 / h, basis table, curve table, span
// lookup), on the CPU with 1, 8 and all usable threads and on the GPU, and where the GPU's time goes.
// Results in docs/benchmarks.md.
//
//   bench_precompute [knots] [degree]      defaults: knots = 10004, degree = 3
//
// Every row is a rebuild: all memory is allocated before the clock starts. One warm-up first, so that
// neither OpenMP creating its threads nor CUDA loading the kernels is timed.

namespace
{
	using bench::bestOf;
	using bench::Clock;
	using bench::formatMicroseconds;
	using bench::makeBenchmarkCurve;
	using bench::parseNumber;
	using bench::secondsSince;
	using bench::speedInWords;
	using bench::withThousands;
	
	constexpr std::size_t kDefaultKnots = 10'004;
	constexpr std::size_t kNumStages = 9;
	constexpr std::array<std::string_view, kNumStages> kStageNames{"upload knots", "upload control points", "1 / h kernel", "basis table kernel", "curve table kernel", "span lookup kernel", "download curve table", "download 1 / h", "download span lookup"};
	
	// The results in host memory: what the CPU fills, and where the GPU downloads what queries need.
	template <std::floating_point T>
	struct HostTables
	{
		std::vector<T> inverseSpanLengths{};
		std::vector<T> basis{};
		std::vector<geom::Point4<T>> curve{};
		std::vector<std::uint32_t> lookup{};
	};
	
	// The inputs and results in device memory.
	template <std::floating_point T>
	struct DeviceTables
	{
		gpu::DeviceBuffer<T> knots{};
		gpu::DeviceBuffer<geom::Point4<T>> points{};
		gpu::DeviceBuffer<T> inverseSpanLengths{};
		gpu::DeviceBasisTable<T> basis{};
		gpu::DeviceCurveTable<T> curve{};
		gpu::DeviceSpanLookup<T> lookup{};
	};
	
	// One scenario of one type, in seconds.
	struct ScenarioTimes
	{
		std::vector<double> cpuSeconds{};   // one per entry of the thread counts
		double gpuComputeSeconds{};         // the kernels on data already on the GPU (CUDA events)
		double gpuWithCopiesSeconds{};      // upload the inputs, the kernels, download the results (host clock)
	};
	
	// Where one GPU run with copies spends its time: the fastest of several runs.
	struct Breakdown
	{
		std::array<double, kNumStages> stageSeconds{};   // from CUDA events, GPU side
		double eventSeconds{};                           // first to last event
		double hostSeconds{};                            // host clock around the whole run
	};
	
	struct TypeTimes
	{
		ScenarioTimes all{};             // knots and points changed: 1 / h, basis, curve table, lookup
		ScenarioTimes pointsChanged{};   // only the points changed: the curve table
		Breakdown breakdown{};
		bool threadedMatches{true};      // every threaded result equals the library's sequential one
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The CPUs a container's quota allows (docker --cpus), rounded down, at least 1; 0 if there is no quota.
	// cgroup v2 has "quota period" (or "max period") in cpu.max, cgroup v1 two separate files.
	int cgroupCpuLimit()
	{
		long quota{};
		long period{};
		std::ifstream v2("/sys/fs/cgroup/cpu.max");
		std::ifstream v1Quota("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
		std::ifstream v1Period("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
		bool const found = static_cast<bool>(v2 >> quota >> period) || static_cast<bool>((v1Quota >> quota) && (v1Period >> period));
		return (found && quota > 0 && period > 0) ? std::max(1, static_cast<int>(quota / period)) : 0;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Threads above a container's CPU quota get throttled for milliseconds, so they must not be measured:
	// on a 128-thread machine with a quota of 13.6 CPUs an empty OpenMP region could take over 10 ms.
	int usableThreads()
	{
		int const hardware = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
		int const limit = cgroupCpuLimit();
		return limit > 0 ? std::min(hardware, limit) : hardware;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The thread counts to measure: 1, 8 and all usable threads, as far as this machine has them.
	std::vector<int> threadCountsToMeasure()
	{
		int const usable = usableThreads();
		std::vector<int> counts{1};
		if (usable > 8)
		{
			counts.push_back(8);
		}
		if (usable > 1)
		{
			counts.push_back(usable);
		}
		return counts;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Host memory for all tables, written here (std::vector writes every element), not in a timed region.
	template <std::floating_point T>
	HostTables<T> allocateHostTables(geom::NurbsCurve<T> const& curve, geom::SpanLookup<T> const& lookup)
	{
		auto const p = static_cast<std::size_t>(curve.degree());
		std::size_t const numSpans = curve.knots().size() - 2 * p - 1;
		return {std::vector<T>(numSpans), std::vector<T>(numSpans * (p + 1) * (p + 1)), std::vector<geom::Point4<T>>(numSpans * (p + 1)), std::vector<std::uint32_t>(lookup.firstSpans().size())};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void fillInverseSpanLengthsThreaded(int degree, std::span<T const> knots, std::span<T> inverseSpanLengths, int threads)
	{
		auto const p = static_cast<std::size_t>(degree);
#pragma omp parallel for num_threads(threads) schedule(static)
		for (std::size_t row = 0; row < inverseSpanLengths.size(); ++row)
		{
			inverseSpanLengths[row] = geom::detail::inverseSpanLength<T>(row + p, knots);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void fillBasisThreaded(int degree, std::span<T const> knots, std::span<T> coefficients, int threads)
	{
		std::size_t const numSpans = knots.size() - 2 * static_cast<std::size_t>(degree) - 1;
#pragma omp parallel for num_threads(threads) schedule(static)
		for (std::size_t row = 0; row < numSpans; ++row)
		{
			geom::detail::buildBasisRow<T>(row, degree, knots, coefficients);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void fillCurveThreaded(int degree, std::span<T const> basis, std::span<geom::Point4<T> const> points, std::span<geom::Point4<T>> coefficients, int threads)
	{
		auto const p = static_cast<std::size_t>(degree);
		std::size_t const numSpans = basis.size() / ((p + 1) * (p + 1));
#pragma omp parallel for num_threads(threads) schedule(static)
		for (std::size_t row = 0; row < numSpans; ++row)
		{
			for (std::size_t d = 0; d <= p; ++d)
			{
				geom::detail::buildCurveCoefficient<T>(row, d, degree, basis, points, coefficients);
			}
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One thread's chunk [first, last) of the cells: one findSpan for its first cell, then the CPU's sweep.
	template <std::floating_point T>
	void fillLookupChunk(int degree, std::span<T const> knots, geom::SpanLookup<T> const& lookup, std::span<std::uint32_t> cells, std::size_t first, std::size_t last)
	{
		std::size_t const n = knots.size() - static_cast<std::size_t>(degree) - 2;
		std::size_t span = geom::detail::firstSpanOfCell<T>(first, degree, knots, lookup.start(), lookup.cellsPerUnit());
		for (std::size_t cell = first; cell < last; ++cell)
		{
			T const leftEdge = geom::detail::cellLeftEdge<T>(cell, lookup.start(), lookup.cellsPerUnit());
			while (span < n && leftEdge >= knots[span + 1])
			{
				++span;
			}
			cells[cell] = static_cast<std::uint32_t>(span);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Splits the cells evenly over the threads; a thread with no cells does nothing.
	template <std::floating_point T>
	void fillLookupThreaded(int degree, std::span<T const> knots, geom::SpanLookup<T> const& lookup, std::span<std::uint32_t> cells, int threads)
	{
#pragma omp parallel num_threads(threads)
		{
			auto const thread = static_cast<std::size_t>(omp_get_thread_num());
			auto const team = static_cast<std::size_t>(omp_get_num_threads());
			std::size_t const first = cells.size() * thread / team;
			std::size_t const last = cells.size() * (thread + 1) / team;
			if (first < last)
			{
				fillLookupChunk<T>(degree, knots, lookup, cells, first, last);
			}
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void fillAllThreaded(geom::NurbsCurve<T> const& curve, geom::SpanLookup<T> const& lookup, int threads, HostTables<T>& host)
	{
		fillInverseSpanLengthsThreaded<T>(curve.degree(), curve.knots(), host.inverseSpanLengths, threads);
		fillBasisThreaded<T>(curve.degree(), curve.knots(), host.basis, threads);
		fillCurveThreaded<T>(curve.degree(), host.basis, curve.weightedControlPoints(), host.curve, threads);
		fillLookupThreaded<T>(curve.degree(), curve.knots(), lookup, host.lookup, threads);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// All tables with the library's sequential functions, without OpenMP.
	template <std::floating_point T>
	void fillAllSequential(geom::NurbsCurve<T> const& curve, geom::SpanLookup<T> const& lookup, HostTables<T>& host)
	{
		geom::fillInverseSpanLengths<T>(curve.degree(), curve.knots(), host.inverseSpanLengths);
		geom::fillBasisTable<T>(curve.degree(), curve.knots(), host.basis);
		geom::fillCurveTable<T>(curve.degree(), host.basis, curve.weightedControlPoints(), host.curve);
		geom::fillSpanLookup<T>(curve.degree(), curve.knots(), lookup.start(), lookup.cellsPerUnit(), host.lookup);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One thread is the library's sequential code, so that the reference row carries no OpenMP overhead.
	template <std::floating_point T>
	void fillAll(geom::NurbsCurve<T> const& curve, geom::SpanLookup<T> const& lookup, int threads, HostTables<T>& host)
	{
		if (threads == 1)
		{
			fillAllSequential<T>(curve, lookup, host);
		}
		else
		{
			fillAllThreaded<T>(curve, lookup, threads, host);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	double timeCpuAll(geom::NurbsCurve<T> const& curve, geom::SpanLookup<T> const& lookup, int threads, HostTables<T>& host)
	{
		Clock::time_point const start = Clock::now();
		fillAll<T>(curve, lookup, threads, host);
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Only the control points changed: the curve table, from the basis table already in host.
	template <std::floating_point T>
	double timeCpuPointsChanged(geom::NurbsCurve<T> const& curve, int threads, HostTables<T>& host)
	{
		Clock::time_point const start = Clock::now();
		if (threads == 1)
		{
			geom::fillCurveTable<T>(curve.degree(), host.basis, curve.weightedControlPoints(), host.curve);
		}
		else
		{
			fillCurveThreaded<T>(curve.degree(), host.basis, curve.weightedControlPoints(), host.curve, threads);
		}
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The same tables from the library's sequential functions, to check the threaded loops against.
	template <std::floating_point T>
	HostTables<T> sequentialTables(geom::NurbsCurve<T> const& curve, geom::SpanLookup<T> const& lookup)
	{
		HostTables<T> host = allocateHostTables<T>(curve, lookup);
		fillAllSequential<T>(curve, lookup, host);
		return host;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	bool tablesEqual(HostTables<T> const& a, HostTables<T> const& b)
	{
		return a.inverseSpanLengths == b.inverseSpanLengths && a.basis == b.basis && a.curve == b.curve && a.lookup == b.lookup;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Each thread count starts with an untimed build, so that OpenMP's team of that size already exists.
	template <std::floating_point T>
	void measureCpu(geom::NurbsCurve<T> const& curve, geom::SpanLookup<T> const& lookup, std::span<int const> threadCounts, TypeTimes& t)
	{
		HostTables<T> const reference = sequentialTables<T>(curve, lookup);
		HostTables<T> host = allocateHostTables<T>(curve, lookup);
		for (int const threads : threadCounts)
		{
			fillAll<T>(curve, lookup, threads, host);
			t.all.cpuSeconds.push_back(bestOf([&] { return timeCpuAll<T>(curve, lookup, threads, host); }));
			t.pointsChanged.cpuSeconds.push_back(bestOf([&] { return timeCpuPointsChanged<T>(curve, threads, host); }));
			t.threadedMatches = t.threadedMatches && tablesEqual(host, reference);
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The inputs uploaded, the results allocated. The caller passes the domain from the curve.
	template <std::floating_point T>
	DeviceTables<T> allocateDeviceTables(geom::NurbsCurve<T> const& curve)
	{
		auto const p = static_cast<std::size_t>(curve.degree());
		std::size_t const numSpans = curve.knots().size() - 2 * p - 1;
		auto const [start, end] = curve.domain();
		return {gpu::DeviceBuffer<T>{curve.knots()}, gpu::DeviceBuffer<geom::Point4<T>>{curve.weightedControlPoints()}, gpu::DeviceBuffer<T>(numSpans), gpu::allocateBasisTableOnDevice<T>(curve.degree(), curve.knots().size()), gpu::allocateCurveTableOnDevice<T>(curve.degree(), numSpans), gpu::allocateSpanLookupOnDevice<T>(curve.degree(), curve.knots().size(), start, end)};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void launchAllKernels(int degree, DeviceTables<T>& device)
	{
		gpu::fillInverseSpanLengthsOnDevice<T>(degree, device.knots, device.inverseSpanLengths);
		gpu::fillBasisTableOnDevice<T>(device.knots, device.basis);
		gpu::fillCurveTableOnDevice<T>(device.basis, device.points, device.curve);
		gpu::fillSpanLookupOnDevice<T>(device.knots, device.lookup);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The four kernels, with events around them.
	template <std::floating_point T>
	double timeGpuAllCompute(int degree, DeviceTables<T>& device)
	{
		gpu::CudaEvent start{};
		gpu::CudaEvent stop{};
		start.record();
		launchAllKernels<T>(degree, device);
		stop.record();
		return gpu::elapsedSeconds(start, stop);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// What queries on the CPU would need: the basis table stays on the GPU.
	template <std::floating_point T>
	double timeGpuAllWithCopies(geom::NurbsCurve<T> const& curve, DeviceTables<T>& device, HostTables<T>& host)
	{
		Clock::time_point const start = Clock::now();
		device.knots.upload(curve.knots());
		device.points.upload(curve.weightedControlPoints());
		launchAllKernels<T>(curve.degree(), device);
		device.curve.coefficients.download(std::span<geom::Point4<T>>(host.curve));
		device.inverseSpanLengths.download(std::span<T>(host.inverseSpanLengths));
		device.lookup.firstSpans.download(std::span<std::uint32_t>(host.lookup));
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Only the control points changed: the curve kernel, with events around it.
	template <std::floating_point T>
	double timeGpuPointsChangedCompute(DeviceTables<T>& device)
	{
		gpu::CudaEvent start{};
		gpu::CudaEvent stop{};
		start.record();
		gpu::fillCurveTableOnDevice<T>(device.basis, device.points, device.curve);
		stop.record();
		return gpu::elapsedSeconds(start, stop);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Only the control points changed: upload them, the curve kernel, download the curve table.
	template <std::floating_point T>
	double timeGpuPointsChangedWithCopies(geom::NurbsCurve<T> const& curve, DeviceTables<T>& device, HostTables<T>& host)
	{
		Clock::time_point const start = Clock::now();
		device.points.upload(curve.weightedControlPoints());
		gpu::fillCurveTableOnDevice<T>(device.basis, device.points, device.curve);
		device.curve.coefficients.download(std::span<geom::Point4<T>>(host.curve));
		return secondsSince(start);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The upload stages; events[k] is recorded after stage k.
	template <std::floating_point T>
	void runUploadStages(geom::NurbsCurve<T> const& curve, DeviceTables<T>& device, std::span<gpu::CudaEvent> events)
	{
		device.knots.upload(curve.knots());
		events[0].record();
		device.points.upload(curve.weightedControlPoints());
		events[1].record();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The kernel stages; events[k] is recorded after stage k.
	template <std::floating_point T>
	void runKernelStages(int degree, DeviceTables<T>& device, std::span<gpu::CudaEvent> events)
	{
		gpu::fillInverseSpanLengthsOnDevice<T>(degree, device.knots, device.inverseSpanLengths);
		events[0].record();
		gpu::fillBasisTableOnDevice<T>(device.knots, device.basis);
		events[1].record();
		gpu::fillCurveTableOnDevice<T>(device.basis, device.points, device.curve);
		events[2].record();
		gpu::fillSpanLookupOnDevice<T>(device.knots, device.lookup);
		events[3].record();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The download stages; events[k] is recorded after stage k.
	template <std::floating_point T>
	void runDownloadStages(DeviceTables<T>& device, HostTables<T>& host, std::span<gpu::CudaEvent> events)
	{
		device.curve.coefficients.download(std::span<geom::Point4<T>>(host.curve));
		events[0].record();
		device.inverseSpanLengths.download(std::span<T>(host.inverseSpanLengths));
		events[1].record();
		device.lookup.firstSpans.download(std::span<std::uint32_t>(host.lookup));
		events[2].record();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One run with copies, with an event between all stages.
	template <std::floating_point T>
	Breakdown timeGpuStages(geom::NurbsCurve<T> const& curve, DeviceTables<T>& device, HostTables<T>& host)
	{
		std::array<gpu::CudaEvent, kNumStages + 1> events{};
		std::span<gpu::CudaEvent> const after = std::span<gpu::CudaEvent>(events).subspan(1);
		Clock::time_point const start = Clock::now();
		events[0].record();
		runUploadStages<T>(curve, device, after.subspan(0, 2));
		runKernelStages<T>(curve.degree(), device, after.subspan(2, 4));
		runDownloadStages<T>(device, host, after.subspan(6, 3));
		Breakdown result{};
		result.hostSeconds = secondsSince(start);
		for (std::size_t k = 0; k < kNumStages; ++k)
		{
			result.stageSeconds[k] = gpu::elapsedSeconds(events[k], events[k + 1]);
		}
		result.eventSeconds = gpu::elapsedSeconds(events[0], events[kNumStages]);
		return result;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The breakdown of the fastest run (by the host clock).
	template <std::floating_point T>
	Breakdown measureBreakdown(geom::NurbsCurve<T> const& curve, DeviceTables<T>& device, HostTables<T>& host)
	{
		Breakdown best{};
		best.hostSeconds = std::numeric_limits<double>::infinity();
		auto const run = [&] { Breakdown const b = timeGpuStages<T>(curve, device, host); best = (b.hostSeconds < best.hostSeconds) ? b : best; return b.hostSeconds; };
		bestOf(run);
		return best;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// One untimed build first, so that the kernels are loaded.
	template <std::floating_point T>
	void measureGpu(geom::NurbsCurve<T> const& curve, geom::SpanLookup<T> const& lookup, TypeTimes& t)
	{
		DeviceTables<T> device = allocateDeviceTables<T>(curve);
		HostTables<T> host = allocateHostTables<T>(curve, lookup);
		launchAllKernels<T>(curve.degree(), device);
		gpu::checkCuda(cudaDeviceSynchronize());
		t.all.gpuComputeSeconds = bestOf([&] { return timeGpuAllCompute<T>(curve.degree(), device); });
		t.all.gpuWithCopiesSeconds = bestOf([&] { return timeGpuAllWithCopies<T>(curve, device, host); });
		t.pointsChanged.gpuComputeSeconds = bestOf([&] { return timeGpuPointsChangedCompute<T>(device); });
		t.pointsChanged.gpuWithCopiesSeconds = bestOf([&] { return timeGpuPointsChangedWithCopies<T>(curve, device, host); });
		t.breakdown = measureBreakdown<T>(curve, device, host);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	TypeTimes measureType(int degree, std::size_t numKnots, std::span<int const> threadCounts)
	{
		geom::NurbsCurve<T> const curve = makeBenchmarkCurve<T>(degree, numKnots);
		geom::SpanLookup<T> const lookup(curve);
		TypeTimes t{};
		measureCpu<T>(curve, lookup, threadCounts, t);
		measureGpu<T>(curve, lookup, t);
		return t;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// "reference" for the row all others are compared with, else "14x faster" or "1.9x slower".
	std::string relativeText(double reference, double seconds, bool isReference)
	{
		return isReference ? std::string("reference") : speedInWords(reference, seconds);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printRow(std::string_view name, double floatSeconds, double doubleSeconds, ScenarioTimes const& f, ScenarioTimes const& d, bool isReference)
	{
		std::println("  {:<30}{:>11}  {:<16}  {:>11}  {}", name, formatMicroseconds(floatSeconds), relativeText(f.cpuSeconds[0], floatSeconds, isReference), formatMicroseconds(doubleSeconds), relativeText(d.cpuSeconds[0], doubleSeconds, isReference));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printScenario(std::string_view title, ScenarioTimes const& f, ScenarioTimes const& d, std::span<int const> threadCounts)
	{
		std::println();
		std::println("{}", title);
		std::println("  {:<30}{:-^29}  {:-^29}", "", " float ", " double ");
		std::println("  {:<30}{:>11}  {:<16}  {:>11}  {}", "", "time [us]", "vs 1 CPU thread", "time [us]", "vs 1 CPU thread");
		for (std::size_t k = 0; k < threadCounts.size(); ++k)
		{
			printRow(std::format("CPU, {} thread{}", threadCounts[k], threadCounts[k] == 1 ? "" : "s"), f.cpuSeconds[k], d.cpuSeconds[k], f, d, k == 0);
		}
		printRow("GPU, computation only", f.gpuComputeSeconds, d.gpuComputeSeconds, f, d, false);
		printRow("GPU, incl. copies CPU <-> GPU", f.gpuWithCopiesSeconds, d.gpuWithCopiesSeconds, f, d, false);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// "12.3 us" and its share of the total.
	void printStageRow(std::string_view name, double floatSeconds, double floatTotal, double doubleSeconds, double doubleTotal)
	{
		std::println("  {:<28}{:>11}{:>9.1f} %{:>13}{:>9.1f} %", name, formatMicroseconds(floatSeconds), 100.0 * floatSeconds / floatTotal, formatMicroseconds(doubleSeconds), 100.0 * doubleSeconds / doubleTotal);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printBreakdown(Breakdown const& f, Breakdown const& d)
	{
		std::println();
		std::println("WHERE THE GPU TIME GOES  (all tables, incl. copies; the fastest run)");
		std::println("  {:<28}{:>11}{:>11}{:>13}{:>11}", "", "float [us]", "share", "double [us]", "share");
		for (std::size_t k = 0; k < kNumStages; ++k)
		{
			printStageRow(kStageNames[k], f.stageSeconds[k], f.eventSeconds, d.stageSeconds[k], d.eventSeconds);
		}
		printStageRow("total (CUDA events)", f.eventSeconds, f.eventSeconds, d.eventSeconds, d.eventSeconds);
		printStageRow("total (host clock)", f.hostSeconds, f.eventSeconds, d.hostSeconds, d.eventSeconds);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printHeader(int degree, std::size_t numKnots)
	{
		cudaDeviceProp properties{};
		gpu::checkCuda(cudaGetDeviceProperties(&properties, 0));
		std::size_t const numSpans = numKnots - 2 * static_cast<std::size_t>(degree) - 1;
		std::println("Precomputation on the CPU ({} usable of {} hardware threads) and the GPU ({}, sm_{}{})", usableThreads(), std::thread::hardware_concurrency(), properties.name, properties.major, properties.minor);
		std::println("  {} spans, degree {}. Times in µs: best of at least {} runs and {} s, after one warm-up build.", withThousands(numSpans), degree, bench::kRuns, bench::kMinSeconds);
		std::println("  All memory is allocated beforehand. 1 thread = the library's sequential code; more threads use OpenMP, the lookup one chunk of cells per thread.");
		std::println("  GPU, incl. copies: upload the inputs, the kernels, download what CPU queries need (the basis table stays on the GPU).");
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void printAll(int degree, std::size_t numKnots, TypeTimes const& f, TypeTimes const& d, std::span<int const> threadCounts)
	{
		printHeader(degree, numKnots);
		printScenario("PRECOMPUTATION  (knots + control points -> 1 / h, basis table, curve table, span lookup)", f.all, d.all, threadCounts);
		printScenario("CONTROL POINTS CHANGED  (control points -> curve table; 1 / h, basis table, lookup kept)", f.pointsChanged, d.pointsChanged, threadCounts);
		printBreakdown(f.breakdown, d.breakdown);
		std::println();
		std::println("Threaded CPU tables equal the sequential ones: {}", f.threadedMatches && d.threadedMatches ? "yes" : "NO");
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main(int argc, char* argv[])
{
	std::size_t const numKnots = argc > 1 ? parseNumber(argv[1]) : kDefaultKnots;
	std::size_t const degreeArg = argc > 2 ? parseNumber(argv[2]) : 3;
	if (argc > 3 || degreeArg < 1 || degreeArg > static_cast<std::size_t>(geom::kMaxDegree) || numKnots < 2 * (degreeArg + 1))
	{
		std::println(stderr, "usage: bench_precompute [knots >= 2 * (degree + 1)] [1 <= degree <= {}]", geom::kMaxDegree);
		return 1;
	}
	int const degree = static_cast<int>(degreeArg);
	std::vector<int> const threadCounts = threadCountsToMeasure();
	TypeTimes const floats = measureType<float>(degree, numKnots, threadCounts);
	TypeTimes const doubles = measureType<double>(degree, numKnots, threadCounts);
	printAll(degree, numKnots, floats, doubles, threadCounts);
	return 0;
}
