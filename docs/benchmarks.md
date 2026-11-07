# Benchmarks

How to reproduce: build in Release (the default), then run the benchmark by hand.

```
cmake -S . -B build && cmake --build build
./build/bench_curve_point [N] [degree] [--knots K] [--svg]   # --svg also saves build/svg/benchmark_curve.svg
```

## Curve point evaluation, CPU baseline (`bench_curve_point`)

What is measured: `out[k] = curve(u_k)` for N evenly spaced parameters, one thread, results
written to a pre-allocated array. Curve: a rational 3D spiral (control points on a widening
spiral in xy, rising in z, weights 0.5 to 1.5), 100 control points (or K - degree - 1 with
`--knots K`), uniform clamped knots.
Run with `--svg` to see it.
Best of 5 runs after one warm-up. Code: `benchmarks/bench_curve_point.cpp`.

### 2026-10-18, Apple M1 (8 cores, 1 used), Apple clang 21, `-O3 -DNDEBUG`, code as of commit "Add CPU benchmark for curve point evaluation"

| degree | N | type | best [ms] | per point [ns] | Mpoints/s |
|---|---|---|---|---|---|
| 3 | 1,000,000 | float  | 15.0 | 15.0 | 66.8 |
| 3 | 1,000,000 | double | 17.6 | 17.6 | 56.9 |
| 5 | 1,000,000 | float  | 38.9 | 38.9 | 25.7 |
| 5 | 1,000,000 | double | 42.6 | 42.6 | 23.5 |

Observations:

- Cost grows faster than linearly in the degree (about 2.6x from p = 3 to p = 5), as expected:
  A2.2 is O(p^2) per point, while the span search and the sum are cheaper.
- `float` is only ~10-15% faster than `double` on this CPU. On GPUs the gap is usually much
  larger (consumer NVIDIA GPUs have far less double-precision throughput), which matters for
  the precision decision in CLAUDE.md.
- Single-run timings vary by a few percent; compare best-of-5 numbers, not single runs.

## Direct vs precomputed basis table (`bench_curve_point`, since 2026-10-25)

Two methods on the same curve and parameters:

- **direct**: per point `findSpan` (A2.1), `basisFunctions` (A2.2), weighted sum (A4.1), i.e. `curve(u)`.
- **table**: `BasisTable` holds every span's basis polynomials in `t` (built once); per point
  `findSpan`, Horner on the span's `p + 1` polynomials, the same weighted sum
  (`curvePointFromTable`). See `docs/algorithms/basis-polynomials.md`.

The build time of the table is measured separately (best of 5). `max diff` is the largest distance
between the two methods' points over all N.

### 2026-10-25, Apple M1 (1 core), Apple clang 21, `-O3 -DNDEBUG`, N = 1,000,000

| degree | knots | type | direct [ns/pt] | table [ns/pt] | speedup | table build | table size | max diff |
|---|---|---|---|---|---|---|---|---|
| 3 | 104 | float  | 15.0 | 13.1 | 1.15x | 0.01 ms | 6.4 KB | 3.1e-6 |
| 3 | 104 | double | 18.8 | 14.3 | 1.31x | 0.004 ms | 12.9 KB | 5.7e-15 |
| 5 | 106 | float  | 39.0 | 21.5 | 1.81x | 0.01 ms | 13.7 KB | 3.0e-6 |
| 5 | 106 | double | 43.0 | 22.7 | 1.89x | 0.008 ms | 27.5 KB | 6.2e-15 |
| 9 | 110 | float  | 106.6 | 44.6 | 2.39x | 0.02 ms | 35.9 KB | 2.6e-5 |
| 9 | 110 | double | 117.3 | 47.1 | 2.49x | 0.02 ms | 71.8 KB | 5.0e-14 |
| 3 | 10,004 | float  | 17.9 | 17.3 | 1.03x | 0.34 ms | 664 KB | 2.6e-4 |
| 3 | 10,004 | double | 20.9 | 18.6 | 1.12x | 0.34 ms | 1.3 MB | 4.9e-13 |

Observations:

- The table wins more as the degree grows (1.2-1.3x at `p = 3`, 2.4-2.5x at `p = 9`): it replaces
  A2.2, which is `O(p^2)` with divisions, by `p + 1` Horner evaluations without divisions.
- With 10,000 knots the gain at `p = 3` almost disappears: both methods pay the same deeper span
  search (14 instead of 7 steps), and the table (0.7-1.3 MB) no longer fits in the L1 cache.
- The build is cheap: about 0.03 microseconds per span at `p = 3`, so it pays off after a few hundred
  points per curve already.
- The differences are rounding only. In `float` they grow with the coordinates (the 10,000-knot spiral
  reaches radius 800, so 2.6e-4 is about 3e-7 relative) and with the degree (power-basis
  cancellation, see "Precision" in the algorithm doc).

## Direct vs basis table vs curve table (`bench_curve_point`, since 2026-10-25)

Third method **curve**: `CurveTable` holds per span the curve itself as a 4D polynomial in `t`
(basis table times control points, built once); per point `findSpan`, one Horner evaluation of a
4D polynomial, `C = Cw / w` (`curvePointFromCurveTable`). The curve table's build time includes
building its basis table.

### 2026-10-25, Apple M1 (1 core), Apple clang 21, `-O3 -DNDEBUG`, N = 1,000,000, ns per point

| degree | knots | type | direct | basis table | curve table | curve vs direct | curve build | max diff (curve) |
|---|---|---|---|---|---|---|---|---|
| 3 | 104 | float  | 14.8 | 13.1 | 5.6 | 2.65x | 0.004 ms | 3.1e-6 |
| 3 | 104 | double | 17.7 | 14.3 | 5.7 | 3.12x | 0.005 ms | 5.7e-15 |
| 5 | 106 | float  | 39.2 | 21.5 | 6.2 | 6.27x | 0.010 ms | 3.9e-6 |
| 5 | 106 | double | 42.6 | 22.8 | 6.3 | 6.81x | 0.009 ms | 6.0e-15 |
| 9 | 110 | float  | 106.9 | 44.7 | 7.4 | 14.47x | 0.027 ms | 4.6e-6 |
| 9 | 110 | double | 116.6 | 45.9 | 7.5 | 15.63x | 0.028 ms | 3.3e-14 |
| 3 | 10,004 | float  | 17.6 | 17.4 | 10.0 | 1.75x | 0.42 ms | 3.3e-4 |
| 3 | 10,004 | double | 19.6 | 18.7 | 10.1 | 1.94x | 0.43 ms | 5.4e-13 |

Observations:

- The curve table's time per point barely depends on the degree (5.6 ns at `p = 3`, 7.4 ns at `p = 9`):
  only `4p` multiply-adds remain besides the span search. So its advantage grows with the degree,
  to about 15x at `p = 9`.
- With 10,000 knots it is still about 1.8-1.9x faster; the shared span search and the larger table
  (0.7-1.3 MB, beyond the L1 cache) take a bigger share there.
- Building it costs a little more than the basis table (one extra small matrix product per span) and
  still pays off after a few hundred points.
- In `float` at `p = 9` its error (4.6e-6) is smaller than the basis table's (2.6e-5): it does fewer
  operations per point, so fewer roundings.

## Curve table with a span lookup table (`bench_curve_point`, since 2026-10-26)

Fourth method **lookup**: the curve table, with the span from a `SpanLookup` table (2 cells per span)
instead of `findSpan`'s binary search (`curvePointFromTable(curve, table, lookup, u)`; see
`docs/algorithms/span-lookup.md`). The build time shown is the lookup table's alone.

### 2026-10-26, Apple M1 (1 core), Apple clang 21, `-O3 -DNDEBUG`, N = 1,000,000, ns per point (float / double)

| degree | knots | direct | curve table | **curve table + lookup** | lookup vs direct | lookup build | lookup size |
|---|---|---|---|---|---|---|---|
| 3 | 104 | 14.9 / 17.7 | 5.6 / 5.7 | **3.9 / 4.0** | 3.9x / 4.5x | 0.2 us | 0.8 KB |
| 9 | 110 | 107.1 / 116.8 | 7.4 / 7.4 | **7.0 / 7.0** | 15.2x / 16.7x | 0.2 us | 0.7 KB |
| 3 | 10,004 | 17.6 / 19.5 | 10.0 / 10.1 | **3.9 / 4.0** | 4.6x / 4.9x | 20 us | 78 KB |
| 3 | 100,004 | 21.2 / 22.3 | 16.0 / 17.0 | **3.9 / 4.0** | 5.4x / 5.6x | 193 us | 781 KB |

Observations:

- With the lookup the time per point no longer depends on the number of knots: 3.9-4.0 ns from 100 to
  100,000 control points, while the binary search grows from 5.6 to 16 ns. The gain over the curve
  table is 1.45x with 100 knots, 2.6x with 10,000 and 4.1x with 100,000.
- At `p = 9` the gain is small (7.4 -> 7.0 ns): there the 4D Horner, not the span search, dominates.
- A rough estimate (2026-10-26) predicted about 2.4 ns per point; that used a bare `floor` formula for
  uniform knots. The real lookup also checks the domain ends, loads its table and walks over the knots,
  about 1.5 ns more, and in return works for any knot vector.
- Building the lookup is cheap: one pass over cells and knots, 193 us for 100,000 knots (a first version
  with one `findSpan` per cell took 7.1 ms, more than the curve table itself).
- Total precomputation (what must be built before the first point; the benchmark prints it, with the
  memory and the break-even point): basis = basis table; curve = curve table (its basis table inside);
  lookup = curve table + lookup table. With 10,000 knots, `p = 3`, `float`: basis 332 us, curve 416 us,
  lookup 435 us; they pay for themselves after 1.26 million, 54,000 and 32,000 points. So the lookup
  costs the most to prepare but pays off soonest, because it saves the most per point.


## Basis table build, CPU vs GPU (`bench_curve_point` with CUDA, since 2026-10-26)

The last section of `bench_curve_point` (only with CUDA) builds the `BasisTable` on the CPU and with
`buildBasisTableOnDevice` (one thread per span, `src/gpu/basis_table_gpu.cu`), best of 5:

- **kernel only**: CUDA events directly around the launch, into an allocated table;
- **allocate + kernel**: host clock around `buildBasisTableOnDevice` and `cudaDeviceSynchronize`;
- **upload, build, download**: the table ends up in host memory, like the CPU's;
- **first call**: the type's first end-to-end build (for float, the program's first CUDA call: it also
  creates the CUDA context).

### 2026-10-26, RTX 2000 Ada (sm_89) and AMD EPYC 7443 (1 thread), g++-14 -O3, nvcc 12.8, p = 3, microseconds, float (double)

Measured with the first version of the section (best of 5 runs, host allocation inside the GPU rows); see
"GPU table builds" below for the current rows and the reasons they changed.

| spans | CPU | GPU kernel only | GPU allocate + kernel | GPU upload, build, download | GPU first call |
|---|---|---|---|---|---|
| 97 | 4.2 (5.0) | 10.1 (25.4) | 14.9 (30.1) | 160 (179) | 245,503 (217) |
| 9,997 | 433 (515) | 38 (88) | 43 (92) | 275 (403) | 156,168 (494) |
| 99,997 | 4,391 (5,168) | 313 (667) | 450 (813) | 1,277 (2,174) | 147,545 (3,132) |

Largest |GPU - CPU| coefficient: 2.4e-7 (float), 3.3e-16 (double), FMA rounding.

Observations:

- With ~100 spans the GPU loses even on the kernel alone (launch latency, ~10 us). From ~10,000 spans the
  kernel is 11-14x faster than one CPU core (float).
- The first CUDA call costs 150-250 ms (context creation). Double's first call only adds the lazy
  loading of its kernel (CUDA 12 loads each kernel at its first launch).
- End to end, the copies dominate: at 100k spans ~830 us of 1,277 go to upload and download (6.4 MB
  from pageable host memory, ~8 GB/s). So later kernels should read the table where it is, on the GPU.
- The kernel is about 10x slower than its memory limit: writing 6.4 MB at ~224 GB/s takes ~30 us, the
  kernel 313 us. Double takes 2x as long as float (the bytes), not ~30x (this GPU's FP64 rate is 1/64), so
  it is limited by memory access, not arithmetic. Suspects: the in-place recurrence works on the table
  in global memory (p read/write passes over each row), and neighbouring threads write rows 64 bytes
  apart (uncoalesced). Next: confirm with `ncu`.

## GPU table builds, CPU vs GPU (`bench_curve_point` with CUDA, since 2026-11-07)

The last part of `bench_curve_point` builds the `BasisTable` and the `CurveTable` on the CPU (1 thread) and on
the GPU (`src/gpu/basis_table_gpu.cu`, `curve_table_gpu.cu`). Each row compares the same work on both:

- **Rebuild**: memory allocated beforehand and reused, as in a program that rebuilds a table whenever the
  curve changes. CPU: `fillBasisTable` / `fillCurveTable` into existing vectors.
- **Build once**: the build allocates its own memory. CPU: the constructors. GPU: `cudaMalloc` and new host vectors.
- GPU **computation only**: the kernel(s) on data already on the GPU (build once: also `cudaMalloc` of the
  table). GPU **incl. copies**: upload the inputs, the kernel(s), download the table into CPU memory.
- **First build in this program**: build once incl. copies, the first time. The program's first CUDA call
  (float basis table) also starts CUDA; every other first build loads its kernel.

Sections: **basis table** (knots -> basis table); **curve table, stage 2 only** (basis table -> curve table,
basis table already built: the work the curve kernel does); **curve table, whole job** (knots + control
points -> curve table). The resulting table is never freed inside the timed region (the caller keeps it),
but a build once's temporaries are, as a one-off build pays for them before it returns: the CPU
`CurveTable(curve)` frees its `BasisTable`, the GPU incl. copies `cudaFree`s all its device buffers (its
result is the downloaded host table). Every time is the best of at least 5 runs and 0.2 s (`bestOf`);
first build is one run.

### 2026-11-07, RTX A4500 (sm_86, 640 GB/s) and AMD EPYC 7352 (1 thread), g++-14 -O3, nvcc 12.8, p = 3, microseconds

Each value is the median of three program runs. Rebuild rows agree within a few percent between runs.
Build once rows include allocation and vary more, up to ~25%, and more for large double tables on the GPU
(`cudaMalloc` of 64-128 MB). First builds are a single run per program run and vary most.

Basis table (knots -> basis table):

| spans | comparison | float CPU | float GPU | GPU is | double CPU | double GPU | GPU is |
|---|---|---|---|---|---|---|---|
| 97 | Rebuild, computation only | 6.3 | 12.6 | 2.0x slower | 6.8 | 29.4 | 4.3x slower |
|  | Rebuild, incl. copies CPU <-> GPU | 6.3 | 27.6 | 4.4x slower | 6.8 | 45.0 | 6.6x slower |
|  | Build once, computation only | 6.5 | 18.5 | 2.8x slower | 7.0 | 35.4 | 5.1x slower |
|  | Build once, incl. copies CPU <-> GPU | 6.5 | 41.4 | 6.4x slower | 7.0 | 59.4 | 8.5x slower |
|  | First build in this program | 6.5 | 169,766 | 26,118x slower | 7.0 | 431.3 | 62x slower |
| 9,997 | Rebuild, computation only | 659.0 | 25.8 | 26x faster | 712.8 | 54.9 | 13x faster |
|  | Rebuild, incl. copies CPU <-> GPU | 659.0 | 113.3 | 5.8x faster | 712.8 | 205.0 | 3.5x faster |
|  | Build once, computation only | 669.6 | 32.2 | 21x faster | 732.7 | 60.8 | 12x faster |
|  | Build once, incl. copies CPU <-> GPU | 669.6 | 142.8 | 4.7x faster | 732.7 | 246.8 | 3.0x faster |
|  | First build in this program | 669.6 | 186,983 | 279x slower | 732.7 | 1,393 | 1.9x slower |
| 99,997 | Rebuild, computation only | 6,643 | 141.3 | 47x faster | 7,171 | 316.6 | 23x faster |
|  | Rebuild, incl. copies CPU <-> GPU | 6,643 | 696.5 | 9.5x faster | 7,171 | 1,370 | 5.2x faster |
|  | Build once, computation only | 6,734 | 328.1 | 21x faster | 7,583 | 489.0 | 16x faster |
|  | Build once, incl. copies CPU <-> GPU | 6,734 | 1,292 | 5.2x faster | 7,583 | 2,636 | 2.9x faster |
|  | First build in this program | 6,734 | 179,216 | 27x slower | 7,583 | 10,108 | 1.3x slower |
| 999,997 | Rebuild, computation only | 68,030 | 1,333 | 51x faster | 71,838 | 2,822 | 25x faster |
|  | Rebuild, incl. copies CPU <-> GPU | 68,030 | 8,278 | 8.2x faster | 71,838 | 16,683 | 4.3x faster |
|  | Build once, computation only | 103,639 | 2,286 | 45x faster | 144,074 | 3,630 | 40x faster |
|  | Build once, incl. copies CPU <-> GPU | 103,639 | 47,683 | 2.2x faster | 144,074 | 90,484 | 1.6x faster |
|  | First build in this program | 103,639 | 291,293 | 2.8x slower | 144,074 | 93,509 | 1.5x faster |

Curve table, stage 2 only (basis table -> curve table; basis table already built):

| spans | comparison | float CPU | float GPU | GPU is | double CPU | double GPU | GPU is |
|---|---|---|---|---|---|---|---|
| 97 | Rebuild, computation only | 1.3 | 8.8 | 6.8x slower | 1.5 | 9.1 | 6.1x slower |
|  | Build once, computation only | 1.5 | 13.8 | 9.2x slower | 1.8 | 13.5 | 7.5x slower |
| 9,997 | Rebuild, computation only | 133.7 | 9.1 | 15x faster | 154.3 | 11.6 | 13x faster |
|  | Build once, computation only | 146.5 | 13.7 | 11x faster | 181.9 | 151.5 | 1.2x faster |
| 99,997 | Rebuild, computation only | 1,387 | 32.9 | 42x faster | 1,784 | 61.2 | 29x faster |
|  | Build once, computation only | 1,559 | 178.2 | 8.7x faster | 2,552 | 391.7 | 6.5x faster |
| 999,997 | Rebuild, computation only | 14,499 | 274.4 | 53x faster | 18,564 | 553.3 | 34x faster |
|  | Build once, computation only | 52,376 | 553.9 | 95x faster | 94,451 | 915.2 | 103x faster |

Curve table, whole job (curve: knots + control points -> curve table):

| spans | comparison | float CPU | float GPU | GPU is | double CPU | double GPU | GPU is |
|---|---|---|---|---|---|---|---|
| 97 | Rebuild, incl. copies CPU <-> GPU | 7.6 | 36.8 | 4.8x slower | 8.3 | 55.8 | 6.7x slower |
|  | Build once, incl. copies CPU <-> GPU | 7.9 | 308.9 | 39x slower | 8.9 | 326.6 | 37x slower |
|  | First build in this program | 7.9 | 546.2 | 69x slower | 8.9 | 438.6 | 49x slower |
| 9,997 | Rebuild, incl. copies CPU <-> GPU | 793.5 | 136.3 | 5.8x faster | 867.5 | 251.1 | 3.5x faster |
|  | Build once, incl. copies CPU <-> GPU | 818.6 | 418.7 | 2.0x faster | 915.2 | 765.2 | 1.2x faster |
|  | First build in this program | 818.6 | 746.0 | 1.1x faster | 915.2 | 880.4 | 1.0x faster |
| 99,997 | Rebuild, incl. copies CPU <-> GPU | 8,035 | 848.0 | 9.5x faster | 8,919 | 1,640 | 5.4x faster |
|  | Build once, incl. copies CPU <-> GPU | 8,484 | 2,339 | 3.6x faster | 18,227 | 4,034 | 4.5x faster |
|  | First build in this program | 8,484 | 6,501 | 1.3x faster | 18,227 | 11,908 | 1.5x faster |
| 999,997 | Rebuild, incl. copies CPU <-> GPU | 81,361 | 9,753 | 8.3x faster | 91,353 | 18,410 | 5.0x faster |
|  | Build once, incl. copies CPU <-> GPU | 163,481 | 53,379 | 3.1x faster | 251,664 | 98,615 | 2.6x faster |
|  | First build in this program | 163,481 | 53,194 | 3.1x faster | 251,664 | 99,699 | 2.5x faster |

Largest |GPU - CPU| coefficient: basis 2.4e-7 (float), 4.4e-16 (double); curve up to 1.8e-2 (float), 4.4e-11
(double) at 1M spans. The curve's are absolute differences on coordinates up to ~8e4, so ~2 ulp (FMA rounding,
see step 2 in `docs/gpu-plan.md`).

Observations:

- Computation only: from 10k spans the curve kernel is 15-53x faster than the CPU (stage 2, float; double
  13-34x), the basis kernel 26-51x. The curve kernel is close to its memory limit: at 1M spans it reads 64 MB of basis
  table and 16 MB of points and writes 64 MB (float), ~225 us at 640 GB/s if each byte moves once, against
  274 us measured. The basis kernel (1,333 us for 64 MB written) is not, as found in step 1.
- Curve table, whole job, rebuild: GPU 6-10x faster than the CPU from 10k spans (float; double 3-6x). The copies dominate: at 1M spans the kernels take ~1.6 ms of 9.7 ms, the rest
  is ~88 MB between pageable host memory and the GPU (~11 GB/s). Pinned host memory would be the next step
  if this matters.
- Curve table, whole job, build once: GPU 2.0-3.6x faster (float). Both sides pay for host memory here, and that is large: at 1M
  spans allocating the curve table's 64 MB (float) costs the CPU ~38 ms (stage 2: 52.4 ms build once vs
  14.5 ms rebuild), and the GPU about as much (53.4 vs 9.8 ms, with the `cudaFree`s of its temporaries). The first write to each new 4 KB page is a
  page fault, ~2 us each.
- How much allocation costs depends on glibc: blocks above its mmap threshold come fresh from the OS (page
  faults again) each time, smaller ones are reused from its heap. Hence double at 100k spans: whole job
  build once 18.2 ms vs rebuild 8.9 ms (two 12.8 MB tables), but float only 8.5 vs 8.0 ms (6.4 MB).
- Allocation on the GPU (measured with a standalone `cudaMalloc`/`cudaFree` loop): the driver hands out
  device memory in 2 MB chunks. An allocation that fits in the free part of an already mapped chunk costs
  ~2 us; otherwise `cudaMalloc` maps a new chunk (~130 us, ~850 us for 32 MB) and `cudaFree` unmaps it
  again (~110 us), every time. Hence double's stage 2, build once, computation only at 10k spans (152 us vs 14 us for float): it
  has ~1.76 MB live (knots, points, basis table), so its 1.36 MB curve table needs a new chunk; float
  (0.9 MB + 0.68 MB) fits. So: reuse device tables (`fill...OnDevice`) or use a memory pool
  (`cudaMallocAsync`) when builds are repeated.
- Freeing the GPU's temporaries is not free either (`cudaFree` is synchronous and unmaps chunks it empties).
  Timing them (old vs new binary, run alternately) added to build once incl. copies: basis table ~7 us at
  97 spans, ~0.15-0.2 ms at 100k; curve table whole job ~120 us at 97 spans, ~0.7 ms (float) and ~1.2 ms
  (double) at 100k, 2-5 ms (5-10%) at 1M.

### How the method got here (2026-11-07)

- The first version of the curve section downloaded into a new `std::vector` inside the timed region, and
  compared with the CPU constructors. Large host allocations made those rows vary between runs and with
  glibc settings (`GLIBC_TUNABLES=glibc.malloc.mmap_threshold=...` moved GPU end-to-end at 100k spans from 3
  to 10 ms) and the first call looked faster than the best run (later runs also freed the previous run's
  64/128 MB result). Hence the split into rebuild and build once, with the CPU `fill` functions so that
  both sides can build into existing memory.
- Best of a fixed 5 runs was not enough for runs under a few ms: in a standalone test, best of 5 basis table
  builds (10k spans, ~0.5 ms each) came out 1.5-2.7x too slow and differed between runs, best of 50 was
  stable (probably the CPU clocking up after waiting for the GPU; not confirmed). Hence `bestOf` runs for at
  least 0.2 s; it also applies to the table build times of the first section.
- Until the free rule above, the GPU build once incl. copies stopped the clock before its device buffers
  were freed, while the CPU `CurveTable(curve)` frees its temporary `BasisTable` inside the timed region.
  The tables were re-measured then (median of three runs). Some build once rows of the earlier tables had
  also gone stale after the measuring order changed (basis table, 97 spans, incl. copies: 172 us in the
  earlier table, 35 us measured with the code just before the change).
