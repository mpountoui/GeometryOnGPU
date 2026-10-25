# Precomputed span polynomials (basis and curve)

Code: `src/geom/basis_polynomials.hpp` (step 2, CPU basis table; next: step 3, the kernel).
Tests: `tests/test_basis_polynomials.cpp`, against `basisFunctions` (A2.2); later also
`curvePoint` (A4.1).

Goal: precompute everything that does not depend on `u`, so that a later `C(u)` query at any
parameter is cheaper than A2.1 + A2.2 + A4.1. The precomputation runs once per curve, on the GPU.

## Book reference

- Chapter 6, *B-spline <-> power basis* conversion, book p.265 (PDF p.136). **Not read yet**
  (the PDF is only on the dev Mac): read it and note here how the book's method (expected:
  knot insertion to Bezier segments, then a Bezier-to-power matrix) compares with the direct
  recurrence below.
- The recurrence itself is Eq. (2.5), book p.50. Section 2.5 (pp.67-70) explains why only
  `p + 1` functions are nonzero on a span; see `basis-functions.md`.

## Idea

Each `N_{i,p}(u)` is a piecewise polynomial: on every knot span `[u_j, u_{j+1})` it is an
ordinary polynomial of degree `<= p`, a different one on each span. So "the coefficients of
`N_{i,p}`" means one coefficient set per span. On span `j` only `N_{j-p,p} .. N_{j,p}` are
nonzero, so per span we need `(p + 1)` functions times `(p + 1)` coefficients.

Multiplying them by the control points gives the curve itself as one polynomial per span:

    Cw(u) = sum_{r=0..p} N_{j-p+r,p}(u) * Pw[j-p+r] = sum_{d=0..p} A_j[d] * t^d

`A_j[d]` are 4D (homogeneous) coefficients, so `C(u) = Cw / w` as in A4.1.

## Local parameter

The polynomials are written in a local parameter of the span, not in `u`:

    t = (u - u_j) / h_j,   h_j = u_{j+1} - u_j,   t in [0, 1)

Expanding in powers of `u` is numerically bad: with knots far from 0 the coefficients become
large with alternating signs and cancel (much worse in `float`). With only a shift `u - u_j`, the
coefficients grow like `1 / h^d`, which with thousands of knots (`h ~ 1e-3`) and `p = 9` gets
close to the `float` limit. With `t` in `[0, 1]` the coefficients stay of moderate size.

Cost: one subtraction and one multiplication (store `1 / h_j`) per query.

## Recurrence on polynomials

On span `j`, level `k` (degree `k`) holds the `k + 1` functions `N_{i,k}`, `i = j-k+r`,
`r = 0..k`, each as `k + 1` coefficients in `t`. Level 0 is the constant `1` (only `N_{j,0}` is
nonzero on span `j`).

Eq. (2.5): `N_{i,k} = L * N_{i,k-1} + R * N_{i+1,k-1}`, with the two linear factors written in `t`
(`u = u_j + h t`):

    L = (u - u_i) / (u_{i+k} - u_i)                 = (u_j - u_i) / dL + (h / dL) t
    R = (u_{i+k+1} - u) / (u_{i+k+1} - u_{i+1})     = (u_{i+k+1} - u_j) / dR - (h / dR) t

A factor whose denominator is 0 (repeated knots) is taken as 0, as in the book's 0/0 = 0 rule.

In level `k - 1`, `N_{i,k-1}` has index `r - 1` and `N_{i+1,k-1}` has index `r`; indices outside
`0..k-1` are zero functions. Multiplying a polynomial `c` by `(beta + alpha t)` is

    out[d] = beta * c[d] + alpha * c[d-1]       (c[-1] = c[k] = 0)

so every output coefficient `(r, d)` of level `k` is four multiply-adds from level `k - 1`:

    out(r, d) = betaL * c_{r-1}[d] + alphaL * c_{r-1}[d-1] + betaR * c_r[d] + alphaR * c_r[d-1]

On the CPU this can run **in place** in one `(p + 1) x (p + 1)` array (row `r`, column `d`): go
through `r` and `d` downwards, then `(r, d)` only reads entries with smaller or equal indices,
which still hold level `k - 1`. Row `k` and column `k` are still 0 from the initialization, so
they act as the zero entries `c_k` and `c[k]`. The GPU version uses two buffers instead, because
all threads of a level write at the same time.
All `(k + 1)^2` outputs of a level are independent of each other, which is what the GPU
parallelizes; the levels must run in order.

## Data layout

- One row per span index `j = p .. n` (`n + 1 - p` rows, row = `j - p`), so `findSpan` gives the
  row directly. Zero-length spans (repeated knots) get a row too but are never used, because
  `findSpan` never returns them. This wastes a little memory and keeps indexing simple.
- Per row: `u_j`, `1 / h_j`, and `A_j[0..p]` as `Point4<T>` (aligned like `float4`).
- The basis table `(p + 1) x (p + 1)` per span is an intermediate result. It is kept (and
  downloaded) only for testing against `basisFunctions`.

## GPU mapping (first version)

- Kernel 1, basis table: **one block per span**, one thread per output `(r, d)`, so
  `(p + 1)^2` threads (16 for `p = 3`). Levels live in shared memory, two buffers used in
  turn (read level `k - 1`, write level `k`), with `__syncthreads()` between levels.
  `__syncthreads()` only synchronizes one block, which is enough because spans are independent:
  any number of spans means just more blocks.
- Kernel 2, curve coefficients: one thread per `(span, d)`, `A_j[d] = sum_r coef[r][d] * Pw[j-p+r]`.
  (May be merged into kernel 1 later.)
- Known weakness: 16 threads use half of a 32-thread warp. Later: several spans per block.

## Evaluation (the query)

    j = findSpan(u);  t = (u - u_j) * invH_j;  Cw = Horner(A_j, t);  C = Cw / w

Work per point: binary search (same as now) plus `4p` multiply-adds, instead of A2.2 (`O(p^2)`,
with divisions) plus the weighted sum `4(p + 1)`.

Two ways to run queries, to be measured against each other (and against `curvePoint`):

1. **Host:** download the table once, evaluate on the CPU. No GPU latency per query.
2. **Device:** keep the table on the GPU, upload a batch of `u`, one thread per `u`, download
   the points. Each batch pays a fixed launch + copy latency (microseconds), so this should
   only pay off for large batches. The benchmark should find where the two cross over.

## Benchmarks (planned)

**Table build, CPU vs GPU, against the number of knots** (e.g. 100, 1,000, 10,000, 40,000;
`p = 3` and `p = 9`). Estimate per span: ~250 flops for `p = 3`, ~2,300 for `p = 9`. At 40,000
spans that is ~10 M flops (3-10 ms on one CPU core) for `p = 3`; on the GPU the compute is
microseconds and the cost is the copies: ~1.6 MB up (knots, control points, double) and ~5 MB
down (the table), so ~0.1-0.5 ms. Expectation: the GPU wins from a few thousand spans on.
Measure separately:

- CPU, one thread, and CPU multithreaded (spans are independent; a fair baseline is ~8x faster).
- GPU kernel time only (CUDA events), GPU with upload, GPU with upload and download.
- Exclude the one-time CUDA context creation (~100 ms or more on the first CUDA call): do a
  warm-up call first. A program that builds just one small table will never win it back.

**Queries:** host evaluation from the table vs `curvePoint`, and device evaluation against the
batch size (1, 100, 10^4, 10^6 parameters) to find where it overtakes the host.

## Tests (planned)

- Basis table: evaluate each stored polynomial at sample `t` and compare with `basisFunctions`
  on the same span, for `p = 1, 2, 3, 5, 9`, uniform and non-uniform knots, and with repeated
  interior knots (zero-length spans, 0/0 factors).
- Curve table: Horner evaluation against `curvePoint` on a sweep of `u`, including `u = u_m`.
- `float` and `double`, with a tolerance per type.
- GPU tables (step 3) equal the CPU tables (step 2), up to rounding.
