# Span lookup table (replaces the binary search of A2.1)

Code: `src/geom/span_lookup.hpp` (`SpanLookup`, `findSpanWithLookup`); used by the curve table in
`src/geom/curve_table.hpp`. Tests: `tests/test_span_lookup.cpp`. Benchmark: method `lookup` of
`bench_curve_point`.

Not from the book. A2.1 `FindSpan` (p.68) finds the span with a binary search; this finds the same
span with a precomputed table.

## Why

With the curve table, the binary search became the main cost of a point (measured 2026-10-26, M1,
`double`): 4.2 of 6.4 ns per point with 100 control points, 8.4 of 9.9 ns with 10,000. It takes
`log2(spans)` steps and every step waits for the memory load of the previous one. Finding the span in
O(1) instead brings the whole evaluation to about 2.4 ns per point in both cases.

## Idea

Split the domain `[a, b]` (`a = u_p`, `b = u_{n+1}`) into `cells` equal cells and store, for every cell,
the span that contains its left edge (the first span that overlaps the cell). For a query:

    if u >= b: return n;   if u <= a: return p            (same special cases as findSpan)
    cell = floor((u - a) * cells / (b - a)), clamped to [0, cells - 1]
    span = firstSpans[cell]
    while u >= u_{span+1}: span++                         (usually 0 or 1 step)
    while u <  u_span:     span--                         (only after rounding at a cell edge)

Example, `p = 2`, knots `{0,0,0, 0.1, 0.15, 0.6, 1,1,1}` (spans 2: [0, 0.1), 3: [0.1, 0.15),
4: [0.15, 0.6), 5: [0.6, 1]), 4 cells of width 0.25: `firstSpans = {2, 4, 4, 5}`. `u = 0.12`: cell 0,
start at 2, one step to 3. `u = 0.65`: cell 2, start at 4, one step to 5. `u = 1`: span `n = 5`.

## Correctness

The result is always the same as `findSpan`, for every `u` in the domain `[a, b]`. Outside it, it clamps to
`p` and `n`, as `findSpan` does in release builds (in debug builds `findSpan` asserts that `u` is inside):

- `u >= b` gives `n` and `u <= a` gives `p`, as in `findSpan`. Otherwise `a < u < b`, so the walks stay
  between `p` and `n`.
- After the forward walk, `u < u_{span+1}`. If it moved, also `u_span <= u`. If it did not move and the
  start was too far right (a rounding effect at a cell edge, `u < u_span`), the backward walk moves left
  until `u_span <= u`; every span it passes ends at a knot `> u`, so it stops on a non-empty span.
- Empty spans (repeated knots) are skipped by the forward walk, since their end knot equals their start
  knot `<= u`. So the result is the unique non-empty span with `u_span <= u < u_{span+1}`.

## Cost

- Steps per query: at most the number of knots inside the cell (plus one at a rounding edge). With
  `cellsPerSpan` cells per span (default 2) most cells hold zero or one knot. Clusters of very short
  spans give more steps in their cells; more cells per span fix that.
- Build, once: one `findSpan` per cell. Memory: one 32-bit index per cell (2 x 10,000 spans = 80 KB).
- Uniform knots need no special case: every cell lies within at most two spans.
- GPU: one load from a small table and a few comparisons per thread, instead of a chain of dependent
  loads with divergent branches.

## Tests

- Equal to `findSpan` for the example above (and its table `{2, 4, 4, 5}` with one cell per span), Ex2.3
  (an empty span), a `p`-fold interior knot, a cluster of tiny spans inside one cell, and random knot
  vectors (fixed seed) with repeated knots. Queries at random `u`, exactly at every knot, at `a`, `b`
  and slightly outside `[a, b]`; with 1, 2 and 4 cells per span; `double` and `float`.
- Curve points through the lookup equal `curve(u)`.
