#pragma once

#include <concepts>
#include <cstddef>

#include "gpu/device_buffer.hpp"

namespace gpu
{
	// A geom::BasisTable built on the GPU and kept there: same layout and values (see
	// src/geom/basis_table.hpp), but in device memory, so later kernels can read it without a copy.
	// To look at it on the CPU, download the buffers: table.coefficients.download().
	template <std::floating_point T>
	struct DeviceBasisTable
	{
		int degree{};
		std::size_t numSpans{};                  // n + 1 - p rows, one per span j = p..n
		DeviceBuffer<T> coefficients{};          // numSpans rows of (p + 1)^2 values
		DeviceBuffer<T> inverseSpanLengths{};    // numSpans values 1 / h_j (0 for empty spans)
	};
	
	// The functions below are defined in basis_table_gpu.cu for float and double only.
	
	// Device memory for the basis table of numKnots knots, not yet filled (uninitialized).
	template <std::floating_point T>
	DeviceBasisTable<T> allocateBasisTableOnDevice(int degree, std::size_t numKnots);
	
	// Fills an allocated table from a valid clamped knot vector (already in device memory), one thread
	// per span. Only launches the kernel, no allocation, so it can reuse a table and be timed alone.
	// Returns once the kernel is launched: the GPU may still be working, and the next call that waits
	// for it (e.g. a download) also reports errors from the kernel. Throws std::invalid_argument if the
	// table was allocated for a different number of knots.
	template <std::floating_point T>
	void fillBasisTableOnDevice(DeviceBuffer<T> const& knots, DeviceBasisTable<T>& table);
	
	// allocateBasisTableOnDevice, then fillBasisTableOnDevice.
	template <std::floating_point T>
	DeviceBasisTable<T> buildBasisTableOnDevice(int degree, DeviceBuffer<T> const& knots);
}  // namespace gpu
