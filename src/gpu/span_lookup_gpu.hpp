#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>

#include "gpu/device_buffer.hpp"

namespace gpu
{
	// A geom::SpanLookup built on the GPU and kept there: same values (see src/geom/span_lookup.hpp), but
	// firstSpans is in device memory, so later kernels can read it without a copy.
	template <std::floating_point T>
	struct DeviceSpanLookup
	{
		int degree{};
		std::size_t numKnots{};                     // of the knot vector it was allocated for
		T start{};                                  // a = u_p, the start of the domain
		T cellsPerUnit{};                           // cells / (b - a)
		DeviceBuffer<std::uint32_t> firstSpans{};   // one span index per cell
	};
	
	// The functions below are defined in span_lookup_gpu.cu for float and double only.
	
	// Device memory for the lookup table of numKnots knots with domain [start, end], not yet filled
	// (uninitialized). The caller passes the domain because the host cannot read knots in device memory.
	template <std::floating_point T>
	DeviceSpanLookup<T> allocateSpanLookupOnDevice(int degree, std::size_t numKnots, T start, T end, std::size_t cellsPerSpan = 2);
	
	// Fills an allocated table from a valid clamped knot vector (already in device memory) with the domain
	// given at allocation, one thread per cell. Only launches the kernel, no allocation, so it can reuse a
	// table and be timed alone. Returns once the kernel is launched: the next call that waits for the GPU
	// also reports its errors. Throws std::invalid_argument if the table was allocated for a different number
	// of knots.
	template <std::floating_point T>
	void fillSpanLookupOnDevice(DeviceBuffer<T> const& knots, DeviceSpanLookup<T>& lookup);
	
	// allocateSpanLookupOnDevice, then fillSpanLookupOnDevice.
	template <std::floating_point T>
	DeviceSpanLookup<T> buildSpanLookupOnDevice(int degree, DeviceBuffer<T> const& knots, T start, T end, std::size_t cellsPerSpan = 2);
}  // namespace gpu
