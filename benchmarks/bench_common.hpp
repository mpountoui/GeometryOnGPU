#pragma once

#include <charconv>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <format>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "geom/basis_table.hpp"
#include "geom/curve_table.hpp"
#include "geom/inverse_span_lengths.hpp"
#include "geom/nurbs_curve.hpp"
#include "geom/point.hpp"
#include "geom/span_lookup.hpp"

// What the benchmark executables share: the benchmark curve, the timing method and number formatting.

namespace bench
{
	using Clock = std::chrono::steady_clock;
	
	constexpr int kRuns = 5;
	constexpr double kMinSeconds = 0.2;   // bestOf runs at least kRuns times and at least this long in total
	
	// Written by timeCpuBuildOnce so the compiler cannot drop the table builds it times.
	inline double volatile checksumSink = 0.0;
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Uniform clamped knots: degree + 1 zeros, evenly spaced interior knots, degree + 1 ones.
	template <std::floating_point T>
	std::vector<T> uniformClampedKnots(int degree, std::size_t numPoints)
	{
		auto const p = static_cast<std::size_t>(degree);
		std::size_t const numInterior = numPoints - p - 1;
		std::vector<T> knots(p + 1, T{0});
		for (std::size_t k = 1; k <= numInterior; ++k)
		{
			knots.push_back(static_cast<T>(k) / static_cast<T>(numInterior + 1));
		}
		knots.insert(knots.end(), p + 1, T{1});
		return knots;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A rational 3D spiral with numKnots uniform clamped knots: control points on a widening spiral
	// in xy that rises in z (about six turns per 100 points), weights between 0.5 and 1.5.
	// Precondition: numKnots >= 2 * (degree + 1).
	template <std::floating_point T>
	geom::NurbsCurve<T> makeBenchmarkCurve(int degree, std::size_t numKnots)
	{
		std::size_t const numControlPoints = numKnots - static_cast<std::size_t>(degree) - 1;
		std::vector<geom::Point3<T>> points{};
		std::vector<T> weights{};
		for (std::size_t i = 0; i < numControlPoints; ++i)
		{
			double const t = static_cast<double>(i);
			double const radius = 1.0 + 0.08 * t;
			points.push_back({static_cast<T>(radius * std::cos(0.4 * t)), static_cast<T>(radius * std::sin(0.4 * t)), static_cast<T>(0.05 * t)});
			weights.push_back(static_cast<T>(1.0 + 0.5 * std::sin(0.7 * t)));
		}
		return geom::NurbsCurve<T>(degree, uniformClampedKnots<T>(degree, numControlPoints), points, weights);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	inline double secondsSince(Clock::time_point start)
	{
		return std::chrono::duration<double>(Clock::now() - start).count();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// The smallest result of measure(), which returns seconds, over at least kRuns runs and kMinSeconds in
	// total. A fixed few runs are not enough for short measurements: best of 5 builds of ~0.5 ms came out
	// 1.5-2.7x too slow and differed between runs (the CPU had probably not reached full speed yet).
	template <typename Measure>
	double bestOf(Measure const& measure)
	{
		Clock::time_point const start = Clock::now();
		double best = measure();
		for (int run = 1; run < kRuns || secondsSince(start) < kMinSeconds; ++run)
		{
			best = std::min(best, measure());
		}
		return best;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// 1000000 -> "1,000,000".
	inline std::string withThousands(std::size_t value)
	{
		std::string digits = std::to_string(value);
		for (std::size_t i = digits.size(); i > 3; i -= 3)
		{
			digits.insert(i - 3, ",");
		}
		return digits;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Microseconds: one decimal below 1000, else whole microseconds with thousands separators.
	inline std::string formatMicroseconds(double seconds)
	{
		double const microseconds = seconds * 1e6;
		if (microseconds < 1000.0)
		{
			return std::format("{:.1f}", microseconds);
		}
		return withThousands(static_cast<std::size_t>(std::llround(microseconds)));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A positive integer, or 0 if the text is not one.
	inline std::size_t parseNumber(std::string_view text)
	{
		std::size_t value = 0;
		auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
		if (error != std::errc{} || end != text.data() + text.size())
		{
			return 0;
		}
		return value;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// How long seconds took compared with reference, in words: "14x faster" or "1.9x slower".
	inline std::string speedInWords(double reference, double seconds)
	{
		bool const faster = seconds <= reference;
		double const ratio = faster ? reference / seconds : seconds / reference;
		std::string const number = (ratio < 10.0) ? std::format("{:.1f}", ratio) : withThousands(static_cast<std::size_t>(std::llround(ratio)));
		return number + (faster ? "x faster" : "x slower");
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A value that depends on the whole build, so the compiler cannot drop the builds timeCpuBuildOnce times.
	template <std::floating_point T>
	double sinkValue(geom::BasisTable<T> const& table)
	{
		return static_cast<double>(table.coefficients().back());
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	double sinkValue(geom::CurveTable<T> const& table)
	{
		return static_cast<double>(table.coefficients().back().w);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <std::floating_point T>
	double sinkValue(geom::SpanLookup<T> const& lookup)
	{
		return static_cast<double>(lookup.firstSpans().back());
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A build once of a table (BasisTable, CurveTable or SpanLookup) from the curve: its constructor,
	// allocation included.
	template <typename Table, std::floating_point T>
	double timeCpuBuildOnce(geom::NurbsCurve<T> const& curve)
	{
		Clock::time_point const start = Clock::now();
		Table const table(curve);
		double const seconds = secondsSince(start);
		checksumSink = checksumSink + sinkValue(table);
		return seconds;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A build once of the 1 / h array, allocation included.
	template <std::floating_point T>
	double timeCpuInverseSpanLengthsBuildOnce(geom::NurbsCurve<T> const& curve)
	{
		Clock::time_point const start = Clock::now();
		std::vector<T> const inverseSpanLengths = geom::buildInverseSpanLengths<T>(curve.degree(), curve.knots());
		double const seconds = secondsSince(start);
		checksumSink = checksumSink + static_cast<double>(inverseSpanLengths.back());
		return seconds;
	}
}  // namespace bench
