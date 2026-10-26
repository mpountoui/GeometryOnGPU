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
	
	// Builds the basis table of a valid clamped knot vector (already in device memory) on the GPU,
	// one thread per span. Returns once the kernel is launched: the GPU may still be working, and the
	// next call that waits for it (e.g. a download) also reports errors from the kernel.
	// Defined in basis_table_gpu.cu for float and double only.
	template <std::floating_point T>
	DeviceBasisTable<T> buildBasisTableOnDevice(int degree, DeviceBuffer<T> const& knots);
}  // namespace gpu
