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
