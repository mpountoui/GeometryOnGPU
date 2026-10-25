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

