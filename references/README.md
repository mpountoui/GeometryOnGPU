# References

PDFs here are gitignored (`references/*.pdf`) and stay on the local machine only.

## The NURBS Book, 2nd ed. (Les Piegl, Wayne Tiller; Springer)

- File: `references/TheNURBSBook.pdf` (327 PDF pages, scanned, no text layer: read pages as images).
- Used as the specification for NURBS algorithms. Cite section, algorithm number (A2.1 etc.)
  and book page in `docs/algorithms/<name>.md`.
- The book prints pseudocode as `ALGORITHM A<chapter>.<n>`; each PDF page is a two-page spread.

### Page mapping

Each PDF page holds two book pages (a spread), so for book page `b` (b >= 1):

    PDF page = floor(b / 2) + 4

Examples: book p.1 -> PDF 4, p.141 -> PDF 74, p.641 (Index) -> PDF 324.
Read at most ~6 PDF pages per request. The contents are on PDF pages 3-4 (book pages x-xiv).

### Chapter index (book page -> PDF page)

| Ch | Title | Book p. | PDF p. |
|----|-------|---------|--------|
| 1 | Curve and Surface Basics (power basis, Bezier, rational Bezier, tensor product surfaces) | 1 | 4 |
| 2 | B-spline Basis Functions (definition, derivatives, computational algorithms 2.5 @ p.67) | 47 | 27 |
| 3 | B-spline Curves and Surfaces (definition, derivatives) | 81 | 44 |
| 4 | Rational B-spline Curves and Surfaces (NURBS definition, derivatives) | 117 | 62 |
| 5 | Fundamental Geometric Algorithms (knot insertion p.141, refinement 162, removal 179, degree elevation 188, reduction 212) | 141 | 74 |
| 6 | Advanced Geometric Algorithms (point inversion/projection 229, transforms 236, reparameterization 241, reversal 263, B-spline <-> power basis 265) | 229 | 118 |
| 7 | Conics and Circles | 281 | 144 |
| 8 | Construction of Common Surfaces (cylinder 334, ruled 337, revolution 340) | 333 | 170 |
| 9 | Curve and Surface Fitting (global interp. 364, local interp. 382, global approx. 405, local approx. 437) | 361 | 184 |
| 10 | Advanced Surface Construction (swung 455, skinned 457, swept 472, Coons 496) | 455 | 231 |
| 11 | Shape Modification Tools (control points 511, weights 518, warping 533, constraints 555) | 509 | 258 |
| 12 | Standards and Data Exchange (IGES, STEP, PHIGS) | 571 | 289 |
| 13 | B-spline Programming Concepts (data structures 596, memory 601, utilities 613) | 593 | 300 |
| - | References / Index | 629 / 641 | 318 / 324 |

Spot-checked: PDF 74 = book p.141 (chapter 5 opener); PDF 5-10 = book pp.2-13.
Section-level pages are in the book's contents (PDF 3-4); add them here as they become relevant.
