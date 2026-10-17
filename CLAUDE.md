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
# test:
# benchmark:
```

## Architecture

TODO: a few lines on the main components and how data flows between host and device.
Link longer notes in `docs/` instead of inlining them.

## Conventions

- Languages: C++23 for host code, C++20 for CUDA (`nvcc` has no C++23, so keep C++23-only
  features out of `.cu` files and the headers they include). `CMAKE_CUDA_ARCHITECTURES`
  defaults to `native`; override with `-D`.
- Code guarded by `GEOM_HAS_CUDA` is only compiled when CUDA is available; keep a CPU path.
- Precision: TODO (`float` vs `double` in kernels).
- Data layout: TODO (AoS vs SoA, row-major vs column-major, coordinate conventions).
- Every CUDA API call and kernel launch is error-checked (use one shared macro or helper).
- Host and device buffers are owned by RAII wrappers, not raw `cudaMalloc`/`cudaFree` pairs.

## Gotchas and decisions

Record each with its reason, for example: "Kernel X assumes sorted input because ...".

- (none yet)

## Where things live

- `references/`: local-only PDFs (gitignored). Algorithms follow *The NURBS Book*; before
  implementing one, read `references/README.md` (it has the page mapping) and the relevant
  book pages. The PDF is a scan: read it as page images, a few pages per request.
- `docs/`: design notes, algorithm choices, benchmark results (create when needed).
  One file per algorithm in `docs/algorithms/`, written before or while implementing it.
- `.claude/skills/`: repeatable workflows, added once a task has been done by hand twice.
