#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "geom/basis.hpp"
#include "geom/point.hpp"

namespace geom
{
	// A pth-degree NURBS curve, The NURBS Book Eq. (4.1), p.117:
	//
	//            sum_{i=0..n} N_{i,p}(u) w_i P_i
	//   C(u) = -----------------------------------
	//            sum_{i=0..n} N_{i,p}(u) w_i
	//
	// Stores what the book's curve object stores (Section 13.3, p.596): the degree p, the knots
	// u_0..u_m, and the weighted control points Pw_i = (w_i x_i, w_i y_i, w_i z_i, w_i), which most
	// of the book's algorithms operate on directly. The book's index names are kept:
	// n = number of control points - 1, m = number of knots - 1, and m = n + p + 1.
	//
	// The curve is immutable and validated on construction, so every NurbsCurve object is a
	// valid curve. Algorithms that change a curve (knot insertion, degree elevation, ...)
	// will return a new one.
	//
	// Host-side only (it owns std::vectors). This header must stay C++20 because .cu files
	// will include it; see CLAUDE.md.
	template <std::floating_point T>
	class NurbsCurve
	{
	public:
		// Rational curve from Euclidean control points and their weights.
		// Throws std::invalid_argument if the data does not define a valid curve.
		NurbsCurve(int degree, std::vector<T> knots, std::span<const Point3<T>> points, std::span<const T> weights);
		
		// Non-rational B-spline curve: every weight is 1.
		NurbsCurve(int degree, std::vector<T> knots, std::span<const Point3<T>> points);
		
	public:
		int degree() const noexcept;
		std::size_t n() const noexcept;
		std::size_t m() const noexcept;
		std::size_t numControlPoints() const noexcept;
		
		std::span<const T> knots() const noexcept;
		std::span<const Point4<T>> weightedControlPoints() const noexcept;
		
		// Euclidean control point P_i and weight w_i. Throw std::out_of_range if i > n.
		Point3<T> controlPoint(std::size_t i) const;
		T weight(std::size_t i) const;
		
		// Parameter range [a, b]. The knot vector is clamped, so this is [u_0, u_m].
		std::pair<T, T> domain() const noexcept;
		
		// False when all weights are equal: the curve is then a plain B-spline curve (P4.7, p.118).
		bool isRational() const noexcept;
		
		Point3<T> u_to_xyz(T u) const;

	private:
		[[noreturn]] static void fail(const std::string& message);
		
		// Builds Pw_i = (w_i x_i, w_i y_i, w_i z_i, w_i). Book assumption, p.118: w_i > 0.
		static std::vector<Point4<T>> makeWeightedPoints(std::span<const Point3<T>> points, std::span<const T> weights);
		
		void validate() const;
		void checkDegree() const;
		void checkCounts() const;
		void checkKnotValues() const;
		void checkClamped() const;
		void checkNonEmptyDomain() const;
		void checkInteriorMultiplicities() const;
		
	private:
		int m_degree;
		std::vector<T> m_knots;
		std::vector<Point4<T>> m_controlPointsW;
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	NurbsCurve<T>::NurbsCurve(int degree, std::vector<T> knots, std::span<const Point3<T>> points, std::span<const T> weights)
		: m_degree(degree),
		m_knots(std::move(knots)),
		m_controlPointsW(makeWeightedPoints(points, weights))
	{
		validate();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	NurbsCurve<T>::NurbsCurve(int degree, std::vector<T> knots, std::span<const Point3<T>> points)
		: NurbsCurve(degree, std::move(knots), points, std::vector<T>(points.size(), T{1}))
	{}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	int NurbsCurve<T>::degree() const noexcept
	{
		return m_degree;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::size_t NurbsCurve<T>::n() const noexcept
	{
		return m_controlPointsW.size() - 1;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::size_t NurbsCurve<T>::m() const noexcept
	{
		return m_knots.size() - 1;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::size_t NurbsCurve<T>::numControlPoints() const noexcept
	{
		return m_controlPointsW.size();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::span<const T> NurbsCurve<T>::knots() const noexcept
	{
		return m_knots;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::span<const Point4<T>> NurbsCurve<T>::weightedControlPoints() const noexcept
	{
		return m_controlPointsW;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	Point3<T> NurbsCurve<T>::controlPoint(std::size_t i) const
	{
		return Get3DPoint(m_controlPointsW.at(i));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	T NurbsCurve<T>::weight(std::size_t i) const
	{
		return m_controlPointsW.at(i).w;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::pair<T, T> NurbsCurve<T>::domain() const noexcept
	{
		return {m_knots.front(), m_knots.back()};
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	bool NurbsCurve<T>::isRational() const noexcept
	{
		const T w0 = m_controlPointsW.front().w;
		return std::ranges::any_of(m_controlPointsW, [w0](const Point4<T>& pw) { return pw.w != w0; });
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void NurbsCurve<T>::fail(const std::string& message)
	{
		throw std::invalid_argument("NurbsCurve: " + message);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	std::vector<Point4<T>> NurbsCurve<T>::makeWeightedPoints(std::span<const Point3<T>> points, std::span<const T> weights)
	{
		if (weights.size() != points.size())
		{
			fail(std::to_string(points.size()) + " control points but " + std::to_string(weights.size()) + " weights");
		}
		std::vector<Point4<T>> result;
		result.reserve(points.size());
		for (std::size_t i = 0; i < points.size(); ++i)
		{
			const T w = weights[i];
			if (!(w > T{0}) || !std::isfinite(w))
			{
				fail("weights must be positive and finite");
			}
			result.push_back({w * points[i].x, w * points[i].y, w * points[i].z, w});
		}
		return result;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void NurbsCurve<T>::validate() const
	{
		checkDegree();
		checkCounts();
		checkKnotValues();
		checkClamped();
		checkNonEmptyDomain();
		checkInteriorMultiplicities();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void NurbsCurve<T>::checkDegree() const
	{
		if (m_degree < 1)
		{
			fail("degree must be at least 1");
		}
		if (m_degree > kMaxDegree)
		{
			fail("degree must be at most " + std::to_string(kMaxDegree) + " (kMaxDegree in basis.hpp)");
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void NurbsCurve<T>::checkCounts() const
	{
		const auto p = static_cast<std::size_t>(m_degree);
		const std::size_t numPoints = m_controlPointsW.size();
		if (numPoints < p + 1)
		{
			fail("need at least degree + 1 control points");
		}
		if (m_knots.size() != numPoints + p + 1)
		{
			fail("expected " + std::to_string(numPoints + p + 1) + " knots (control points + degree + 1), got " + std::to_string(m_knots.size()));
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void NurbsCurve<T>::checkKnotValues() const
	{
		if (!std::ranges::all_of(m_knots, [](T u) { return std::isfinite(u); }))
		{
			fail("knots must be finite");
		}
		if (!std::ranges::is_sorted(m_knots))
		{
			fail("knots must be non-decreasing");
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Clamped ("nonperiodic") knot vector, p.117: the first and last knots are repeated exactly
	// p + 1 times, so the curve starts at P_0 and ends at P_n (P4.8). The "exactly" is checked
	// in checkNonEmptyDomain.
	template <std::floating_point T>
	void NurbsCurve<T>::checkClamped() const
	{
		const auto p = static_cast<std::size_t>(m_degree);
		const std::size_t last = m_knots.size() - 1;
		for (std::size_t i = 1; i <= p; ++i)
		{
			if (m_knots[i] != m_knots[0])
			{
				fail("the first degree + 1 knots must be equal (clamped)");
			}
			if (m_knots[last - i] != m_knots[last])
			{
				fail("the last degree + 1 knots must be equal (clamped)");
			}
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	void NurbsCurve<T>::checkNonEmptyDomain() const
	{
		const auto p = static_cast<std::size_t>(m_degree);
		const std::size_t last = m_knots.size() - 1;
		if (!(m_knots[p] < m_knots[p + 1]))
		{
			fail("the first knot is repeated more than degree + 1 times");
		}
		if (!(m_knots[last - p - 1] < m_knots[last - p]))
		{
			fail("the last knot is repeated more than degree + 1 times");
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Interior knots: multiplicity k <= p. The curve is C^(p-k) at a knot (P4.11), so k = p + 1
	// would split it into disconnected pieces.
	template <std::floating_point T>
	void NurbsCurve<T>::checkInteriorMultiplicities() const
	{
		const auto p = static_cast<std::size_t>(m_degree);
		const std::size_t last = m_knots.size() - 1;
		std::size_t multiplicity = 1;
		for (std::size_t i = p + 2; i + p + 1 <= last; ++i)
		{
			multiplicity = (m_knots[i] == m_knots[i - 1]) ? multiplicity + 1 : 1;
			if (multiplicity > p)
			{
				fail("an interior knot is repeated more than degree times");
			}
		}
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	Point3<T> NurbsCurve<T>::u_to_xyz(T u) const
	{
		return curvePoint(*this, u);
	}
}  // namespace geom
