#pragma once

#include <source_location>

#include <cuda_runtime.h>

namespace gpu
{
	// Error checking for the CUDA runtime API.
	//
	// Every CUDA API function returns a cudaError_t; cudaSuccess means it worked. Ignoring it hides
	// bugs: a failed cudaMalloc leaves a null pointer, and a failed kernel only shows up later as
	// garbage results. So every call goes through checkCuda:
	//
	//   gpu::checkCuda(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
	//
	// The default argument std::source_location::current() is evaluated at the call site, so the
	// error message names the file and line of the failing call without needing a macro.
	//
	// Host-only. Compiled in cuda_check.cpp (the gpu library).

	// Throws std::runtime_error ("file:line: CUDA error <name>: <description>") if error is not cudaSuccess.
	void checkCuda(cudaError_t error, std::source_location location = std::source_location::current());

	// Call right after a kernel launch (kernel<<<...>>>(...)), which itself returns nothing.
	// Catches launch errors such as an invalid block size. Errors while the kernel runs are only
	// reported by the next call that waits for the GPU (e.g. cudaMemcpy, cudaDeviceSynchronize).
	void checkLaunch(std::source_location location = std::source_location::current());
}  // namespace gpu
