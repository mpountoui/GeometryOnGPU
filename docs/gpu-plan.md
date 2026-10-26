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
| 1 | Basis table kernel, one thread per span; test GPU == CPU `BasisTable` | `__global__`, `.cu` in CMake, thread indexing, `DeviceBuffer` | agreed, next |
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

## Step 1: agreed decisions

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
