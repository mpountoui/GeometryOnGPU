# GeometryOnGPU

Geometry processing on the GPU, written in C/C++ and CUDA.

> Keep this file short. Record only what can't be learned from the code or `git log`.
> Update it when a change affects build steps, architecture, conventions, or a decision
> with a non-obvious reason. Delete entries that are no longer true.

## Build, run, test

CMake (>= 3.24). CUDA is optional: without a CUDA compiler the project builds CPU-only.
Tests and benchmarks use `<print>`, so the host compiler must be GCC >= 14 (or a recent Clang/libc++);
on the NVIDIA Linux boxes (Ubuntu 24.04, default GCC 13) install `g++-14` and configure with
`-DCMAKE_CXX_COMPILER=g++-14` (a fresh build dir: CMake caches the compiler).
All boxes have nvcc 12.8. The RTX A4000 (sm_86) and RTX 2000 Ada (sm_89) boxes work with `native`.
The B300 box (sm_103) does not,
because nvcc 12.8 does not know sm_103: there also pass `-DCMAKE_CUDA_ARCHITECTURES=100`
(sm_100 code runs on sm_103). Drop this once CUDA is >= 12.9.
The dev Mac (Apple Silicon) has no `nvcc`, so CUDA code can only be built and run on an NVIDIA machine.

```
cmake -S . -B build                  # configure (-DGEOM_ENABLE_CUDA=OFF to force CPU-only)
cmake --build build                  # build
./build/geometry_on_gpu              # run
./build/geometry_on_gpu --svg [dir]  # write demo curve images (default: build/svg/)
ctest --test-dir build --output-on-failure   # test: one executable per tests/test_<name>.cpp,
                                             # registered with geom_add_test(<name>), CHECK from tests/check.hpp
./build/bench_curve_point [N] [degree] [--knots K] [--svg]   # benchmark (Release build); results in docs/benchmarks.md
```

## Architecture

- `src/geom/`: header-only templates (`geom` INTERFACE target in CMake), parameterized on
  `std::floating_point T` so float/double can be compared before fixing the kernel precision.
- `NurbsCurve<T>` (`src/geom/nurbs_curve.hpp`) is a host-side, immutable, validated owner of
  degree, knots and weighted control points. GPU code will get a separate non-owning view.
- Algorithms are free functions on spans, with no allocation or exceptions so they can move
  into kernels: `basis.hpp` (A2.1 `findSpan`, A2.2 `basisFunctions`), `curve_point.hpp`
  (A4.1 `curvePoint`, on raw data), `newton_raphson.hpp` (general 1D root finder for any `f`, `f'`),
  `basis_polynomials.hpp` (a span's basis functions as polynomials in a local `t`),
  `basis_table.hpp` (`BasisTable`: those polynomials for every span, built once; `curvePointFromTable`
  evaluates points from it, 1.2-2.5x faster than `curvePoint` on the CPU), `curve_table.hpp`
  (`CurveTable`: basis table times control points, the curve as one 4D polynomial per span;
  `curvePointFromCurveTable`, 2.7-15x faster than `curvePoint`; rebuild when control points change),
  `span_lookup.hpp` (`SpanLookup`: finds the span in O(1) with a cell table, same result as `findSpan`;
  with the curve table about 4 ns per point independent of the number of knots).
  Each has a doc in `docs/algorithms/`.
  Functions callable from kernels are marked `GEOM_HD` (`host_device.hpp`). They may use
  `std::span`/`std::array` because `geom` adds `--expt-relaxed-constexpr` for `.cu` files.
- `curve(u)` (`NurbsCurve::operator()`) evaluates a point; it calls `curvePoint` with the curve's data.
  So `nurbs_curve.hpp` includes `curve_point.hpp`, never the other way round.
- `src/viz/`: host-only visualization, the `viz` static library. `SvgDocument`
  (`svg_document.hpp/.cpp`) draws curves, control polygons and points as SVG (xy projection,
  y up; `setKeepAspectRatio(false)` for function plots). `newton_plot.hpp/.cpp` plots Newton-Raphson
  steps. Never include `viz` headers from `.cu` files.
- `src/gpu/`: host-side CUDA helpers, the `gpu` static library (only built with CUDA):
  `checkCuda`/`checkLaunch` (`cuda_check.hpp/.cpp`, throw `std::runtime_error`) and the
  move-only RAII `DeviceBuffer<T>` (`device_buffer.hpp`, synchronous `upload`/`download`).
  Kernels live here too: `basis_table_gpu.hpp/.cu` builds a `BasisTable` on the GPU
  (`buildBasisTableOnDevice`, one thread per span, result kept in device memory as `DeviceBasisTable`).
  Pattern: the `.hpp` declares a host function with no CUDA syntax (so `.cpp` files can call it),
  the `.cu` holds the kernel and explicitly instantiates `float` and `double`. Kernels call the same
  `GEOM_HD` per-item functions as the CPU code (e.g. `detail::buildBasisRow`), so the two cannot drift.

Host/device data flow: inputs are uploaded once by the caller (the builders take `DeviceBuffer`s)
and results stay on the device until a test or user downloads them. Longer notes go in `docs/`.

## Conventions

- Languages: C++23 for host code, C++20 for CUDA (`nvcc` has no C++23, so keep C++23-only
  features out of `.cu` files and the headers they include). `CMAKE_CUDA_ARCHITECTURES`
  defaults to `native`; override with `-D`.
- Templates stay in headers (each `T` needs the definition, and so does device code). Non-template
  code goes in a `.cpp` file compiled into a static library (e.g. `viz`), so headers have
  declarations only and no `inline` is needed.
- Code guarded by `GEOM_HAS_CUDA` is only compiled when CUDA is available; keep a CPU path.
- Precision: TODO (`float` vs `double` in kernels).
- Data layout: TODO (AoS vs SoA, row-major vs column-major, coordinate conventions).
- Every CUDA API call and kernel launch is error-checked (use one shared macro or helper).
- Host and device buffers are owned by RAII wrappers, not raw `cudaMalloc`/`cudaFree` pairs.

## Code style

Follow these when writing or changing code:

1. **Functions are at most 20 lines** (signature to closing brace). Split longer ones into
   small named helpers.
2. **Class layout.** The class body holds declarations only, in this order, each group in
   its own labelled access section (repeat `public:`/`private:` per group, do not merge them):
   1. `public:` constructors
   2. `public:` methods
   3. `private:` methods
   4. `private:` members, named with an `m_` prefix (`m_degree`)
   5. `private:` static methods, always last

   All definitions come after the class body (for templates, below the class in the same
   header), in the same order as the declarations. A comment line of the form
   `/*-----...-----*/` at column 0 separates the class from the first definition and every
   definition from the next (copy the exact line from `src/geom/nurbs_curve.hpp`).
3. **Braces are Allman: the opening brace is on its own line** for namespaces, classes,
   structs, functions, `if`/`else`/`for`/`while`/`try`/`catch`. `else` and `catch` go on their
   own line after the closing brace. **Always use braces after `if`** (and `else`), even for a
   single statement: never `if (x) fail("...");`. Lambdas and brace-initializers stay on one line.
4. **Indentation is tabs, everywhere** (also configured in `.editorconfig`). Everything inside
   a `namespace` is indented one tab. No blank line after a namespace's opening brace or before
   its closing `}  // namespace name`. **Blank lines carry tabs** equal to the brace depth they
   are in (empty only at depth 0), so keep `trim_trailing_whitespace` off for code.
5. **Function signatures stay on one line**, declarations and definitions alike, however long.
   Do not wrap them.
6. **Always initialize variables**, locals and the members declared in classes and structs
   alike. Use `{}` for the default value (`int m_degree{};`, `std::vector<T> m_knots{};`,
   `T x{};`), or `= value` / `{value}` when there is a meaningful starting value.
7. **`const` goes after the type it applies to ("east const")**, everywhere: references
   `Point3<T> const& p`, pointers `char const* name`, values `T const w = ...`,
   `auto const [a, b] = ...`, and template arguments `std::span<T const>`. Never `const Type`.
   (`) const` on member functions is already in that position.)

```cpp
namespace geom
{
	class NurbsCurve
	{
	public:
		// constructors (declarations)

	public:
		// methods (declarations)

	private:
		// methods (declarations)

	private:
		// members
		int m_degree{};
		
	private:
		// static methods (declarations)
	};

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	// definitions, in declaration order
}  // namespace geom
```

## Gotchas and decisions

Record each with its reason, for example: "Kernel X assumes sorted input because ...".

- Control points are stored homogeneous, Pw = (w*x, w*y, w*z, w), as in the book (Section 13.3):
  its algorithms operate on Pw directly. `Point4` is aligned to 4*sizeof(T) to match CUDA `float4`.
- Knot vectors must be clamped (first/last knot repeated exactly p+1 times) and interior
  multiplicity <= p, because the book's algorithms assume this (p.117). Periodic curves would
  need a separate type.
- Degree is capped at `kMaxDegree` (9, in `basis.hpp`) and `NurbsCurve` rejects higher degrees:
  evaluation keeps the p+1 basis values in a fixed-size local array (registers on the GPU).
  Raise the constant if a higher degree is ever needed.
- GPU results are not bit-identical to the CPU: nvcc fuses `a * b + c` into FMA, g++ on x86-64 does not.
  GPU tests compare with an ulp tolerance (`--fmad=false` makes them identical, useful to check that
  FMA is the only cause). `cudaMalloc` memory is not zeroed, so kernels write every output value, and
  GPU tests fill freed device memory with NaN first so a missed write fails (`test_basis_table_gpu.cpp`).
- `.cu` files get `-Wall -Wextra` but not `-Wpedantic`: nvcc forwards it to the host compiler, which then
  warns about every line marker in nvcc's generated code.

## Where things live

- `references/`: local-only PDFs (gitignored). Algorithms follow *The NURBS Book*; before
  implementing one, read `references/README.md` (it has the page mapping) and the relevant
  book pages. The PDF is a scan: read it as page images, a few pages per request.
- `docs/`: design notes, algorithm choices, benchmark results (create when needed).
  One file per algorithm in `docs/algorithms/`, written before or while implementing it.
  `docs/gpu-plan.md`: the step-by-step plan (and way of working) for moving the precomputation to
  the GPU; read it first when continuing that work, and update its status as steps finish.
- `.claude/skills/`: repeatable workflows, added once a task has been done by hand twice.
