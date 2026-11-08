#pragma once

#include <concepts>

#include "geom/point.hpp"
#include "gpu/curve_table_gpu.hpp"
#include "gpu/device_buffer.hpp"
#include "gpu/span_lookup_gpu.hpp"

namespace gpu
{
	// Evaluates C(u) for one u on the GPU from tables already in device memory: the curve table and the span
	// lookup, as curvePointFromTable(curve, table, invH, lookup, u) does on the CPU. u is passed as a kernel
	// argument, so it needs no buffer. Writes result[0]. Only launches the kernel, with one thread: download
	// result to wait for it and get the point. Defined in curve_points_gpu.cu for float and double only.
	// Throws std::invalid_argument if the arrays do not belong to the same curve or result is empty.
	template <std::floating_point T>
	void evaluateCurvePointOnDevice(DeviceBuffer<T> const& knots, DeviceBuffer<T> const& inverseSpanLengths, DeviceCurveTable<T> const& table, DeviceSpanLookup<T> const& lookup, T u, DeviceBuffer<geom::Point3<T>>& result);
}  // namespace gpu
