# GeometryOnGPU

NURBS curve evaluation in C++23 and CUDA. The algorithms of *The NURBS Book* (Piegl & Tiller) are restructured
into precomputed tables that make point queries fast on the CPU and can be built on the GPU.

It is also a CUDA learning project: each step adds one CUDA idea and measures it against a fair CPU baseline
(see [docs/gpu-plan.md](docs/gpu-plan.md)).

## What it does

- **`NurbsCurve<T>`**: a validated, immutable rational curve for `float` and `double` (clamped knots,
  degree up to 9, homogeneous control points).
- **Direct evaluation**, as in the book: span search (A2.1), basis functions (A2.2), curve point (A4.1).
- **Precomputed tables**, built once per curve and then queried:
  - `BasisTable`: each span's basis functions as polynomials in a local parameter.
  - `CurveTable`: the curve itself as one 4D polynomial per span. Rebuild it when the control points change.
  - `SpanLookup`: finds the span in O(1) instead of by binary search.
  - The inverse span lengths (1 / h), one small array per knot vector, which the queries use.
- **GPU kernels** that build each of these tables, and one that evaluates a single point from them. The kernels
  call the same per-item functions as the CPU code, so the two cannot drift apart.
- A **Newton-Raphson** root finder, and **SVG** plots of curves, control polygons and Newton steps.

## Results

Cubic curves, `float`. Hardware, method and the full tables are in [docs/benchmarks.md](docs/benchmarks.md).

| Measurement | Result |
|---|---|
| Point query, curve table + span lookup vs direct evaluation (1 CPU thread) | **3-17x faster**: 4 ns (Apple M1) to 13 ns (AMD EPYC) per point, almost independent of the number of knots |
| All tables for 1M spans, GPU kernels vs 13 CPU threads | **2.7x faster** (2.4 ms vs 6.3 ms) |
| The same, including uploading the inputs and downloading the tables | **1.8x slower** (11.4 ms): the copies take ~80 % of the time |
| One query at a time on the GPU | **~10 µs per point** (a launch and a blocking download), far slower than the CPU |

So far the GPU pays off only when its results stay on the GPU.

## Build

Requires CMake >= 3.24 and a C++23 compiler with `<print>` (GCC >= 14, or a recent Clang with libc++).
CUDA (tested with nvcc 12.8) and OpenMP are optional. Without nvcc the project builds CPU-only.

```sh
cmake -S . -B build                 # add -DGEOM_ENABLE_CUDA=OFF to force a CPU-only build
cmake --build build                 # Release by default
ctest --test-dir build --output-on-failure
```

- On Ubuntu 24.04 (default GCC 13), install `g++-14` and configure a fresh build directory with
  `-DCMAKE_CXX_COMPILER=g++-14`.
- `CMAKE_CUDA_ARCHITECTURES` defaults to `native`. If your nvcc does not know your GPU (e.g. sm_103 with CUDA 12.8),
  pass a compatible older architecture, such as `-DCMAKE_CUDA_ARCHITECTURES=100`.

## Run

```sh
./build/geometry_on_gpu                 # lists the CUDA devices
./build/geometry_on_gpu --svg [dir]     # writes demo curve and Newton-Raphson images (default: build/svg/)

./build/bench_curve_point [N] [degree] [--knots K] [--svg]   # point queries: direct vs tables
./build/bench_precompute [degree] [--knots K]                # table builds: CPU threads vs GPU
```

`bench_curve_point` also builds the tables with threads when OpenMP is available, and runs queries on the GPU when
CUDA is available. `bench_precompute` needs both CUDA and OpenMP.

## Layout

| Path | Contents |
|---|---|
| [src/geom/](src/geom/) | Header-only templates: the curve, basis functions, tables and Newton-Raphson. The algorithms also compile into CUDA kernels. |
| [src/gpu/](src/gpu/) | CUDA kernels, error checking, and RAII device buffers and events. |
| [src/viz/](src/viz/) | SVG output (host only). |
| [tests/](tests/) | One test executable per file, for both CPU and GPU. GPU results are compared with an ulp tolerance. |
| [benchmarks/](benchmarks/) | `bench_curve_point` and `bench_precompute`. |
| [docs/algorithms/](docs/algorithms/) | One note per algorithm, with its book reference and design choices. |
| [docs/benchmarks.md](docs/benchmarks.md) | Measured results, how they were obtained, and observations. |
| [docs/gpu-plan.md](docs/gpu-plan.md) | The step-by-step plan for moving the work to the GPU, with its status and decisions. |

Architecture notes and code conventions for contributors are in [CLAUDE.md](CLAUDE.md).

## Reference

Les Piegl, Wayne Tiller, *The NURBS Book*, 2nd ed., Springer. Algorithm numbers (A2.1, A2.2, A4.1) refer to it.
