#pragma once

#include <concepts>
#include <cstddef>

#include "geom/point.hpp"
#include "gpu/basis_table_gpu.hpp"
#include "gpu/device_buffer.hpp"

namespace gpu
{
	// A geom::CurveTable built on the GPU and kept there: same layout and values (see
	// src/geom/curve_table.hpp), but in device memory. To look at it on the CPU, download the buffers.
	template <std::floating_point T>
	struct DeviceCurveTable
	{
		int degree{};
		std::size_t numSpans{};                             // n + 1 - p rows, one per span j = p..n
		DeviceBuffer<geom::Point4<T>> coefficients{};       // numSpans rows of p + 1 weighted points A_j[0..p]
		DeviceBuffer<T> inverseSpanLengths{};               // numSpans values 1 / h_j, copied from the basis table
	};
	
	// The functions below are defined in curve_table_gpu.cu for float and double only.
	
	// Device memory for the curve table of a curve with numSpans spans, not yet filled (uninitialized).
	template <std::floating_point T>
	DeviceCurveTable<T> allocateCurveTableOnDevice(int degree, std::size_t numSpans);
	
	// Fills an allocated table from the curve's basis table and its numSpans + p weighted control points,
	// both already in device memory: one thread per coefficient A_j[d], then a device-to-device copy of
	// the 1 / h values. The basis table is not changed, so it can serve other curves with the same knots.
	// Like fillBasisTableOnDevice it only queues work on the GPU and returns; the next call that waits for
	// the GPU also reports errors. Throws std::invalid_argument if degree or sizes do not match.
	template <std::floating_point T>
	void fillCurveTableOnDevice(DeviceBasisTable<T> const& basis, DeviceBuffer<geom::Point4<T>> const& weightedPoints, DeviceCurveTable<T>& table);
	
	// allocateCurveTableOnDevice, then fillCurveTableOnDevice.
	template <std::floating_point T>
	DeviceCurveTable<T> buildCurveTableOnDevice(DeviceBasisTable<T> const& basis, DeviceBuffer<geom::Point4<T>> const& weightedPoints);
}  // namespace gpu
