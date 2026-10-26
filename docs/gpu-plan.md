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
| 1b | GPU basis table build in `bench_curve_point`: kernel only (CUDA events), build, end to end, first call | Asynchronous launches, CUDA events, warm-up | agreed, next |
| 2 | Curve table kernel, one thread per `(span, d)`, reading the basis table on the device | 2D indexing, coalesced memory access | |
| 3 | Span lookup kernel, one thread per cell (each does its own `findSpan`, instead of the CPU's sequential sweep) | Sequential vs parallel algorithms | |
| 4 | Precompute benchmark: CPU (1 thread, multithreaded) vs GPU (kernel only, with copies, after warm-up) | CUDA events, where the time goes | |
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

## Step 1b: agreed decisions

- New section in `bench_curve_point`, float and double, best of 5 after a warm-up, speedup and max diff
  against the CPU `BasisTable`: CPU 1 thread; GPU kernel only (CUDA events); GPU build
  (`buildBasisTableOnDevice` + `cudaDeviceSynchronize`, includes `cudaMalloc`); GPU end to end
  (upload knots, build, download). Also print the first call (context creation, module loading).
- `gpu::CudaEvent`: move-only RAII wrapper for `cudaEvent_t`, in `src/gpu/`.
- The GPU part is under `#ifdef GEOM_HAS_CUDA`; the benchmark links `gpu` only with CUDA.
- `cudaMalloc` stays inside the timed build for now; revisit if it dominates.
