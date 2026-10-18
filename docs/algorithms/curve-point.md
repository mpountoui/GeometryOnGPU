# Point on a NURBS curve (A4.1 CurvePoint)

Code: `src/geom/curve_point.hpp` and `NurbsCurve::operator()`. Tests: `tests/test_curve_point.cpp`.

## Book reference

- Section 4.2, book p.124 (PDF page 66): **A4.1 CurvePoint**, based on Eq. (4.5).
- It works on the weighted control points `Pw_i = (w_i x_i, w_i y_i, w_i z_i, w_i)`:
  1. `span = FindSpan(u)` (A2.1) and `N = BasisFuns(span, u)` (A2.2);
  2. `Cw = sum_{j=0..p} N[j] * Pw[span - p + j]`, a point in 4D;
  3. `C = Cw / w`, the projection back to 3D.
- Book example **Ex4.1** (pp.122-124): `p = 2`, `U = {0,0,0,1,2,3,3,3}`, weights `{1,4,1,1,1}`,
  points `{(0,0),(1,1),(3,2),(4,1),(5,-1)}`. At `u = 1`: `Cw = (7/2, 3, 5/2)` and `C(1) = (7/5, 6/5)`.

## Our version

```cpp
Point3<T> curvePoint(int degree, std::span<const T> knots, std::span<const Point4<T>> weightedPoints, T u);
Point3<T> NurbsCurve<T>::operator()(T u) const;   // curve(u), calls curvePoint with the curve's data
```

- The basis values live in a local `std::array<T, kMaxDegree + 1>`, so evaluation allocates
  nothing (on the GPU they would sit in registers). `kMaxDegree` is defined in `basis.hpp`, and
  `NurbsCurve` rejects higher degrees, so a valid curve can always be evaluated.
- `Get3DPoint(Pw)` (in `point.hpp`) is the book's projection `C = Cw / w`; `NurbsCurve::controlPoint` uses it too.
- `curve_point.hpp` does not depend on `NurbsCurve`; `nurbs_curve.hpp` includes it. That way
  `curve(u)` works with only `nurbs_curve.hpp` included, and kernels can call `curvePoint` on raw data.
- Same preconditions as `findSpan`: `u` in `[u_p, u_m]` (asserted in debug, clamped in release).

## Tests

- Ex4.1: `C(1) = (7/5, 6/5, 0)`.
- End points (P4.8): `C(a) = P_0` and `C(b) = P_n`, on the rational Figure 4.1 curve.
- Non-rational quadratic Bezier `(0,0), (1,1), (2,0)`: `C(1/2) = (1, 1/2)`.
- Quarter circle as a rational quadratic Bezier, `P = (1,0), (1,1), (0,1)`, `w = (1, sqrt(2)/2, 1)`:
  every point has radius 1 (a classic result, book Chapter 7).
- Scaling all weights by the same factor does not move the curve (P4.7).
- `float` instantiation.

## GPU notes (for later)

- One thread per `u`. Work per thread: `O(log m)` search, `O(p^2)` basis, `O(p)` sum.
- Neighbouring threads with nearby `u` read the same `p + 1` control points, so the reads are
  cache friendly. `Point4<float>` matches `float4`, one 16-byte load per control point.
