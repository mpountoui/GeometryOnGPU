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

