# Precomputation on the GPU: step-by-step plan

Goal: build all precomputed tables on the GPU (`BasisTable`, `CurveTable`, `SpanLookup`; see
`docs/algorithms/basis-polynomials.md` and `span-lookup.md`), and measure against the CPU builds.

This is also a CUDA learning project. Way of working: one CUDA idea per step; Claude proposes each
step (design, code sketch, decisions) and it is discussed **before** anything is applied; then the
code is written in small parts, with every line explained.

Update the status column when a step is done, and record agreed decisions below.

## Steps

| Step | Build | CUDA idea | Status |
|---|---|---|---|
| 0 | g++-14, clean build, ctest, CPU benchmark on the RTX 2000 Ada box | Toolchain | done 2026-10-26 |
| 1 | Basis table kernel, one thread per span; test GPU == CPU `BasisTable` | `__global__`, `.cu` in CMake, thread indexing, `DeviceBuffer` | done 2026-10-26 |
| 1b | GPU basis table build in `bench_curve_point`: kernel only (CUDA events), build, end to end, first call | Asynchronous launches, CUDA events, warm-up | done 2026-10-26 |
| 1c | Profile the basis table kernel with `ncu`: confirm why it is ~10x slower than its memory limit | Nsight Compute, memory throughput, coalescing | postponed: profile with step 2's kernel |
| 2 | Curve table kernel, one thread per `(span, d)`, reading the basis table on the device | 2D indexing, coalesced memory access | done 2026-11-01 |
| 2b | GPU curve table build in `bench_curve_point`, as in step 1b, against the CPU curve stage and the full CPU build | Choosing a fair baseline | done 2026-11-07 |
| 2c | One 1 / h array per knot vector, owned by neither table (CPU and GPU), with its own kernel | Removing a copy by changing who owns the data | done 2026-11-08 |
| 3 | Span lookup kernel, one thread per cell (each does its own `findSpan`, instead of the CPU's sequential sweep) | Sequential vs parallel algorithms | done 2026-11-08 |
| 3b | GPU span lookup build in `bench_curve_point`, as in steps 1b and 2b; CPU `fillSpanLookup` for the rebuild rows | Comparing a parallel algorithm with a sequential one | done 2026-11-08 |
| 4 | Precompute benchmark: CPU (1 thread, multithreaded) vs GPU (kernel only, with copies, after warm-up) | CUDA events, where the time goes | done 2026-11-08 |
| 5 | Basis table kernel with one block per span, levels in shared memory | `__shared__`, `__syncthreads()`, warps | |
| 6 | Fuse kernels, several spans per block, float vs double, profiling | Occupancy, `nsys`/`ncu` | |

Later: queries on the GPU ("Running queries on the GPU" in `basis-polynomials.md`).

## CPU baseline (2026-10-26, AMD EPYC 7443, 1 thread, g++-14 -O3, p = 3, float)

| knots | precompute: curve table + lookup | per point (lookup) |
|---|---|---|
| 104 | 5.7 us | 9.3 ns |
| 10,004 | 589 us | 9.3 ns |
| 100,004 | 5.9 ms | 9.3 ns |

About 60 ns per span. Expectation: a kernel launch plus copies costs 10-30 us, so the GPU loses on
small curves and should win from tens of thousands of spans (or many curves in one launch).

## Step 1: agreed decisions (implemented)

- Move the body of `BasisTable::buildSpan` into a `GEOM_HD` free function `detail::buildBasisRow` on
  spans; `BasisTable` calls it in its loop, the kernel once per thread (one implementation for both).
- The result stays on the device: `gpu::DeviceBasisTable<T>` holds two `DeviceBuffer`s (coefficients,
  inverse span lengths), with `download()` for tests. Step 2 reads it there.
- `src/gpu/basis_table_gpu.cu`: kernel `basisTableKernel` (row = `blockIdx.x * blockDim.x + threadIdx.x`,
  256 threads per block, spare threads return) and host function `buildBasisTableOnDevice<T>(degree, knots)`
  (upload, launch, `checkLaunch`). `basis_table_gpu.hpp` declares it only, so `.cpp` files can call it;
  the `.cu` instantiates `float` and `double` explicitly.
- Test `test_basis_table_gpu.cpp`: against the CPU `BasisTable`, p = 1, 3, 5, 9, uniform and repeated
  knots (empty spans), float and double, with a tolerance of a few ulps (nvcc fuses `a*b + c` into FMA,
  g++ on x86 does not by default).
- Not yet: coalesced writes (each thread writes its own `(p+1)^2` block), shared memory (step 5).

Findings (2026-10-26, RTX 2000 Ada):

- GPU and CPU coefficients differ by at most 2 ulps (times the row's largest value); with
  `--fmad=false` they are bit-identical, so FMA is the only cause. `1 / h` is identical. Test tolerance: 8 ulps.
- `buildBasisRow` must write zeros for empty spans: `cudaMalloc` memory is not zeroed. The test fills
  freed device memory with NaN first (`dirtyDeviceMemory`); without the zero writes the GPU test fails
  while the CPU test still passes (`std::vector` zeroes its memory).

## Step 1b: agreed decisions (implemented)

- New section in `bench_curve_point`, float and double, best of 5 after a warm-up, speedup and max diff
  against the CPU `BasisTable`: CPU 1 thread; GPU kernel only (CUDA events); GPU build
  (`buildBasisTableOnDevice` + `cudaDeviceSynchronize`, includes `cudaMalloc`); GPU end to end
  (upload knots, build, download). Also print the first call (context creation, module loading).
- `gpu::CudaEvent`: move-only RAII wrapper for `cudaEvent_t`, in `src/gpu/`.
- The GPU part is under `#ifdef GEOM_HAS_CUDA`; the benchmark links `gpu` only with CUDA.
- `cudaMalloc` stays inside the timed build for now; revisit if it dominates.
- Changed while implementing: events around `buildBasisTableOnDevice` would also measure the CPU-side
  `cudaMalloc` (the GPU sits idle between the events), so the host function is split into
  `allocateBasisTableOnDevice` and `fillBasisTableOnDevice` (launch only); `build` = both.
- Results in `docs/benchmarks.md`: from ~10k spans the kernel is 11-14x faster than one CPU core, but
  ~10x slower than its memory limit (memory-access bound, see there); copies dominate end to end.

## Step 2: agreed decisions (implemented)

- Move the body of `CurveTable::buildRow` into a `GEOM_HD` free function `detail::buildCurveCoefficient`
  that computes one `A_j[d]`; the CPU constructor calls it per `(row, d)`, the kernel once per thread.
  Empty spans need no branch: their basis row is zero, so `A = 0` is written anyway.
- Kernel `curveTableKernel` in `src/gpu/curve_table_gpu.cu`: 2D block `(p + 1, 256 / (p + 1))`
  (`threadIdx.x = d`, `threadIdx.y` = row in the block), 1D grid of `ceil(numSpans / blockDim.y)` blocks
  along `x` (`gridDim.y` is capped at 65,535). `row = blockIdx.x * blockDim.y + threadIdx.y`. Threads are
  numbered x-first, so a warp writes 32 neighbouring `Point4`s (coalesced), unlike step 1.
- `gpu::DeviceCurveTable<T>` (degree, numSpans, `DeviceBuffer<Point4<T>>` coefficients, `DeviceBuffer<T>`
  inverse span lengths), with `allocate`/`fill`/`build` functions as in step 1.
- The 1/h values were copied from the `DeviceBasisTable` with a device-to-device `cudaMemcpy` after the
  launch (the kernel only computes `A_j[d]`, the basis table stays intact for other curves with the same
  knots). Replaced in step 2c by one array owned by neither table.
- Test `test_curve_table_gpu.cpp`: GPU basis table + GPU curve table against the CPU `CurveTable`, same curves
  as step 1, float and double, ulp tolerance per row; `dirtyDeviceMemory` and `checkRow` move to a shared
  test header.
- No benchmark in this step (step 4 measures all precomputation); added after all in step 2b.

Findings (2026-11-01, RTX A4000):

- GPU and CPU coefficients differ by at most 5.4 (float) and 5.9 (double) units of epsilon times the
  row's scale (largest |c| times largest |Pw| component: the sums cancel, so errors scale with the terms,
  not with A_j[d]). That is the step 1 basis table error plus FMA in the sum. With `--fmad=false` both
  tables are bit-identical. Test tolerance: 16.
- 1/h is bit-identical (copied). A kernel that skips one coefficient per row fails the test (NaN fill).

## Step 2b: agreed decisions (implemented)

- New section in `bench_curve_point` after the basis table one, same pattern (float and double, best of 5,
  `GpuCurveBuild`). Two CPU baselines, so each GPU row is compared with the same work: the CPU curve stage
  alone (`CurveTable(basis, points)`, basis table built before) for kernel only and allocate + kernel; the
  full `CurveTable(curve)` for upload/build/download and first call.
- Kernel only includes the device-to-device copy of 1/h inside `fillCurveTableOnDevice` (same stream; removed in step 2c).
- All GPU sections are measured before anything is printed, so float's first basis table call still
  pays for the CUDA context.

Findings (2026-11-07, RTX A4500; details and numbers in `docs/benchmarks.md`):

- Changed while measuring: each GPU row is compared with the CPU doing the same kind of build, a rebuild
  (all memory allocated before; CPU `fillBasisTable`/`fillCurveTable`, added for this) or a build once
  (allocating inside; CPU constructors). Host allocation of large tables costs as much as the build
  itself (~37 ms for 64 MB) and depends on glibc's state, so mixing kinds made the comparison meaningless.
- `bestOf` runs at least 5 times and 0.2 s: best of 5 was 1.5-2.7x off for runs of ~0.5 ms.
- The result table is never freed in a timed region, but a build once's temporaries are (GPU incl. copies:
  all device buffers, as the CPU `CurveTable(curve)` frees its `BasisTable`): up to ~1.2 ms at 100k spans.
- The curve kernel is near its memory limit (~80% at 1M spans), 15-53x faster than one CPU core from 10k spans.
- Curve table, whole job, rebuild: 6-10x faster than one CPU core (float); the copies take
  ~80% at 1M spans (pageable memory, ~11 GB/s).
- `cudaMalloc` maps a new 2 MB chunk when an allocation does not fit in an already mapped one (~130 us).
- The printed output and `docs/benchmarks.md` show one comparison per row (same work on CPU and GPU, float and
  double side by side, "Nx faster/slower"), with a legend, so they can be read without the code.

## Step 2c: agreed decisions (implemented)

- Neither table needs 1/h to build it, only queries do, and it depends only on the knots. So there is one array
  per knot vector, owned by the caller: `std::vector<T>` from `geom::buildInverseSpanLengths` on the CPU,
  `DeviceBuffer<T>` from `gpu::buildInverseSpanLengthsOnDevice` on the GPU. No `cudaMemcpy`, no struct around it.
- Both sides changed together, so the CPU and GPU rows of the benchmark do the same work. Queries that take a
  table object take the array too: `curvePointFromTable(curve, table, invH, u)`.
- `detail::inverseSpanLength` is the shared `GEOM_HD` per-span function; `buildBasisRow` writes coefficients only.
- Its own kernel, one thread per span. Costs one more launch whenever the knots change; step 6 (fuse kernels)
  can remove it. A rebuild after only the control points change does less work than before.
- Grouping knots, 1/h and lookup cells in a `Knots`/`DeviceKnots`: not done, see step 3.

Findings (2026-11-08, RTX 4000 Ada; numbers in `docs/benchmarks.md`):

- The 1/h kernel takes 3-9 us (float, rebuild) up to 1M spans, so it costs about one launch.
- Without the copy the curve kernel (stage 2, rebuild, 97 spans, float) went from 8.5 to about 4.7 us. Whole job with
  copies, same size: 32.7 to about 36-38 us, because the extra launch and download outweigh the removed copy at this
  size (before: a single run on the same machine).

## Step 3: agreed decisions (implemented)

- Kernel `spanLookupKernel` in `src/gpu/span_lookup_gpu.cu`: one thread per cell, each with its own `findSpan` on the
  cell's left edge, because a thread cannot continue from the previous cell as the CPU sweep does. More total work
  (one binary search per cell, `O(cells log knots)` against `O(cells + knots)`), but independent per cell.
- `gpu::DeviceSpanLookup<T>` with the usual `allocate`/`fill`/`build` functions. The caller passes the domain
  `start, end`, because the host cannot read knots in device memory.
- Shared with the CPU: `detail::numLookupCells`, `lookupCellsPerUnit`, `cellLeftEdge`. The CPU keeps its sweep.
- `detail::firstSpanOfCell` clamps the left edge to `b`: rounding can push it past the end, and `findSpan` asserts
  that `u` is in the domain in debug builds.
- Knots, 1/h and the lookup stay separate: the lookup also depends on `cellsPerSpan`, and nothing takes all three
  until a GPU query kernel exists. Decide on a `DeviceKnots` then, from what that kernel needs.
- No benchmark in this step; added after all in step 3b.

Findings (2026-11-08, RTX 4000 Ada):

- The GPU table is identical to the CPU table in float and double, for the special and random knot vectors with
  1, 2 and 4 cells per span. A kernel that skips one cell fails the test (0xFFFFFFFF fill).

## Step 3b: agreed decisions (implemented)

- New section in `bench_curve_point`, same rows as the basis table. `SpanLookup::fillCells` became the free function
  `geom::fillSpanLookup`, so the CPU rebuild rows build into reused memory (the constructor = allocate + fill).

Findings (2026-11-08, RTX 4000 Ada; numbers in `docs/benchmarks.md`):

- Computation only, rebuild, float: 4.9x faster than the CPU's sweep at 10k spans, 29x at 1M, despite a binary search
  per cell. 15x slower at 97 spans (launch).

## Step 4: agreed decisions (implemented)

- New executable `bench_precompute` (only with CUDA and OpenMP, which is linked into it alone); shared helpers moved to
  `benchmarks/bench_common.hpp`. Whole precomputation and "control points changed" as rebuilds, 1 thread (the library's
  sequential code), 8, and all usable threads, against the GPU with and without copies; a per-stage breakdown from CUDA events.
- OpenMP rather than `std::thread`: it keeps its threads alive between runs, so small curves do not pay for creating them.
- The threaded lookup gives each thread a chunk of cells, one `findSpan` and then the sweep, so the CPU keeps its best algorithm.
- Afterwards the per-table CPU vs GPU sections moved from `bench_curve_point` into `bench_precompute`
  (`bench_table_builds.cpp`), so `bench_curve_point` is only about queries and needs no CUDA.

Findings (2026-11-08, RTX 4000 Ada, numbers in `docs/benchmarks.md`):

- Changed while measuring: "all threads" became all *usable* threads. This container reports 128 hardware threads but has a
  CPU quota of 13.6; with 128 threads an empty OpenMP region averaged ~12 ms even with active waiting.
- Against 13 CPU threads the GPU wins on computation alone (float: 2.4-3.0x from 10k spans) but loses with copies (1.8-2.3x).
  At 1M spans copies are ~79% of the GPU's time. So the results should stay on the GPU: the next gain is queries there.
