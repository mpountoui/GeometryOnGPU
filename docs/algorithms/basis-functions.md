# B-spline basis functions (A2.1 FindSpan, A2.2 BasisFuns)

Code: `src/geom/basis.hpp`. Tests: `tests/test_basis.cpp`.

## Book reference

- Section 2.5, *Computational Algorithms*, book pp.67-70 (PDF pages 37-39).
- **A2.1 FindSpan** (p.68): binary search for the span index `i` with `u` in `[u_i, u_{i+1})`.
  Special case `u == u_m` returns `n` (the last non-empty span), so the curve end is reachable.
- **A2.2 BasisFuns** (p.70): the `p + 1` nonvanishing functions `N_{i-p,p}(u) .. N_{i,p}(u)`,
  built with the inverted triangle of Eq. (2.5) without division by zero.
- Book example **Ex2.3 / Ex2.4** (pp.68-71): `p = 2`, `U = {0,0,0,1,2,3,4,4,5,5,5}`, `u = 5/2`
  gives span `i = 4` and `N_{2,2}, N_{3,2}, N_{4,2} = 1/8, 6/8, 1/8`.

## Our version

```cpp
std::size_t findSpan(int degree, std::span<T const> knots, T u);
void basisFunctions(std::size_t span, T u, int degree, std::span<T const> knots, std::span<T> values);
```

- `n` is derived from the knots: `n = knots.size() - p - 2` (since `m = n + p + 1`).
- `values` must hold `degree + 1` entries; `values[r]` is `N_{span-p+r,p}(u)`.
- **Deviation from the book:** A2.2 stores `left[j] = u - U[i+1-j]` and `right[j] = U[i+j] - u`
  in two scratch arrays. We recompute them from the knots inside the loop instead. The result is
  the same, the function needs no scratch memory and no maximum degree, which suits GPU threads.
- No allocation and no exceptions, so the same code can later run inside a kernel.

## Preconditions

- The knots come from a valid clamped knot vector (`NurbsCurve` guarantees this).
- `u` lies in the domain `[u_p, u_m]`. Checked with `assert` in debug builds. In release builds an
  out-of-range `u` is clamped to the first or last span instead of looping forever (a hang on the
  GPU is worse than a wrong value).

## Tests

- Ex2.3: span index and the three basis values.
- Span search at both ends, at an interior knot, and next to the double knot `u = 4`.
- Partition of unity (P2.4) and nonnegativity (P2.3) on a sweep of `u`, for `p = 2` and `p = 3`.
- End values (P2.8): at `u = a` only the first function is 1, at `u = b` only the last.
- `float` instantiation.

## GPU notes (for later)

- Natural mapping: one thread per parameter value `u`. Work is `O(log m)` for the search and
  `O(p^2)` for the basis, with `p + 1` values kept in registers.
- The device version will take raw pointers (or a non-owning view of the curve) instead of
  `std::span`, unless `std::span` works in device code with the chosen `nvcc` flags.
