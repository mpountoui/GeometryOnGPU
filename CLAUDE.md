# GeometryOnGPU

Geometry processing on the GPU, written in C/C++ and CUDA.

> Keep this file short. Record only what can't be learned from the code or `git log`.
> Update it when a change affects build steps, architecture, conventions, or a decision
> with a non-obvious reason. Delete entries that are no longer true.

## Build, run, test

CMake (>= 3.24). CUDA is optional: without a CUDA compiler the project builds CPU-only.
The dev Mac (Apple Silicon) has no `nvcc`, so CUDA code can only be built and run on an NVIDIA machine.

```
cmake -S . -B build                  # configure (-DGEOM_ENABLE_CUDA=OFF to force CPU-only)
cmake --build build                  # build
./build/geometry_on_gpu              # run
ctest --test-dir build --output-on-failure   # test: one executable per tests/test_<name>.cpp,
                                             # registered with geom_add_test(<name>), CHECK from tests/check.hpp
# benchmark:
```

## Architecture

- `src/geom/`: header-only templates (`geom` INTERFACE target in CMake), parameterized on
  `std::floating_point T` so float/double can be compared before fixing the kernel precision.
- `NurbsCurve<T>` (`src/geom/nurbs_curve.hpp`) is a host-side, immutable, validated owner of
  degree, knots and weighted control points. GPU code will get a separate non-owning view.
- Algorithms are free functions on spans (`src/geom/basis.hpp`: A2.1 `findSpan`, A2.2
  `basisFunctions`), with no allocation or exceptions so they can move into kernels.

TODO: host/device data flow, once the first kernel exists. Longer notes go in `docs/`.

## Conventions

- Languages: C++23 for host code, C++20 for CUDA (`nvcc` has no C++23, so keep C++23-only
  features out of `.cu` files and the headers they include). `CMAKE_CUDA_ARCHITECTURES`
  defaults to `native`; override with `-D`.
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
		int m_degree;
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

## Where things live

- `references/`: local-only PDFs (gitignored). Algorithms follow *The NURBS Book*; before
  implementing one, read `references/README.md` (it has the page mapping) and the relevant
  book pages. The PDF is a scan: read it as page images, a few pages per request.
- `docs/`: design notes, algorithm choices, benchmark results (create when needed).
  One file per algorithm in `docs/algorithms/`, written before or while implementing it.
- `.claude/skills/`: repeatable workflows, added once a task has been done by hand twice.
