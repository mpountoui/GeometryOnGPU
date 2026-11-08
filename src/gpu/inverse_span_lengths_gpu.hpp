#pragma once

#include <concepts>

#include "gpu/device_buffer.hpp"

namespace gpu
{
	// The array of src/geom/inverse_span_lengths.hpp, computed on the GPU and kept there. It is one
	// plain buffer that the caller owns: every table built from the same knots uses this one instance.
	// The functions below are defined in inverse_span_lengths_gpu.cu for float and double only.
	
	// Fills knots.size() - 2p - 1 values from a valid clamped knot vector (already in device memory), one
	// thread per span. Only launches the kernel, no allocation, so it can reuse a buffer and be timed alone.
	// Returns once the kernel is launched: the next call that waits for the GPU also reports its errors.
	// Throws std::invalid_argument if the buffer has the wrong size for these knots.
	template <std::floating_point T>
	void fillInverseSpanLengthsOnDevice(int degree, DeviceBuffer<T> const& knots, DeviceBuffer<T>& inverseSpanLengths);
	
	// Allocates the buffer, then fillInverseSpanLengthsOnDevice.
	template <std::floating_point T>
	DeviceBuffer<T> buildInverseSpanLengthsOnDevice(int degree, DeviceBuffer<T> const& knots);
}  // namespace gpu
