# Newton-Raphson root finding

Code: `src/geom/newton_raphson.hpp`. Plot: `src/viz/newton_plot.hpp`. Tests: `tests/test_newton_raphson.cpp`.

## The method

Find `x` with `f(x) = 0`, given `f` and its derivative `f'`. From a start value `x_0`, replace `f` near
`x_i` by its tangent line and take the point where that line crosses zero:

    x_{i+1} = x_i - f(x_i) / f'(x_i)

Near a simple root the convergence is quadratic (the number of correct digits roughly doubles per
step). It can fail when `f'(x_i)` is close to zero (the tangent is almost horizontal) or when `x_0`
is far from the root, so the iteration needs a limit and a good start value.

Not a numbered algorithm in *The NURBS Book*, but the book uses it: Section 6.1 (p.230, Eq. 6.3)
projects a point onto a curve by solving `f(u) = C'(u) . (C(u) - P) = 0` with Newton iteration.
That is the planned first use of this solver.

## Our version

```cpp
NewtonResult<T> newtonRaphson(F f, DF df, T x0, NewtonOptions<T> options = {});
NewtonResult<T> newtonRaphson(F f, DF df, T x0, NewtonOptions<T> options, OnStep onStep);
```

- `f` and `df` are any callables `T -> T` (lambdas, functions, function objects).
- Stops with `Converged` when `|f(x)| <= valueTolerance`, or when the step is tiny:
  `|x_{i+1} - x_i| <= stepTolerance * max(1, |x_{i+1}|)` (relative for large `x`, absolute near 0).
- Stops with `ZeroDerivative` if `f'(x_i) == 0`, `NotFinite` if a value becomes NaN or infinite,
  `MaxIterations` after `maxIterations` steps. The result always holds the last `x` and `f(x)`.
- Defaults: `valueTolerance = epsilon` of `T`, `stepTolerance = 100 * epsilon` (relative). The value
  tolerance bounds `f`, not `x`: the error in `x` is about `|f| / |f'|`. A first version used
  `100 * epsilon` for it too, and for `float` `x^2 - 2` stopped one step early with `x` off by 2e-6
  (`|f| = 6e-6 < 1.2e-5`); one more step gives full precision. So the default accepts only an
  (almost) exact zero, and the step test does the stopping.
- `onStep(i, x_i, f(x_i), f'(x_i))` is called before every step; the plot uses it to record the path.
- No allocation, no exceptions, marked `GEOM_HD`: it can run inside a kernel when `f` and `df` can.

## Tests

- `x^2 - 2` from 1: the root is sqrt(2), and quadratic convergence (5 steps or fewer).
- Newton's own example `x^3 - 2x - 5` (root 2.0945514815423265), and `cos(x) - x` (0.7390851332151607).
- A start value that already is the root: 0 iterations.
- `x^2 - 1` from 0: `f'(0) = 0`, so `ZeroDerivative`. `x^2 + 1` (no real root): `MaxIterations`.
- The callback sees every step; `float` works.

## Plot

`addNewtonPlot(svg, f, df, x0, a, b)` draws `f` on `[a, b]` (widened to include every iterate), the
x-axis, and for each step the tangent from `(x_i, f(x_i))` down to `(x_{i+1}, 0)` plus a dashed line
up to `(x_{i+1}, f(x_{i+1}))`. Iterates are red dots, the final root green. x and y are scaled
separately (`SvgDocument::setKeepAspectRatio(false)`), since function plots rarely have equal ranges.
