#pragma once

#include <cstddef>
#include <span>

#include <cuda_runtime.h>

#include "check.hpp"
#include "gpu/cuda_check.hpp"
#include "gpu/device_buffer.hpp"

// Test support shared by the GPU test executables (only built with CUDA).

namespace test
{
	// Sets every byte of a freed device block of this size to 0xFF (NaN as float and double). The next
	// cudaMalloc of the same size usually gets this block back, so a value a kernel forgets to write
	// shows up as NaN instead of a lucky 0.
	inline void dirtyDeviceMemory(std::size_t bytes);
	
	// Same number of values, and each gpu[i] within tolerance of cpu[i] (NaN never is).
	template <typename T>
	void checkWithin(std::span<T const> cpu, std::span<T const> gpu, T tolerance);
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	inline void dirtyDeviceMemory(std::size_t bytes)
	{
		gpu::DeviceBuffer<unsigned char> buffer(bytes);
		gpu::checkCuda(cudaMemset(buffer.data(), 0xFF, bytes));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <typename T>
	void checkWithin(std::span<T const> cpu, std::span<T const> gpu, T tolerance)
	{
		CHECK(gpu.size() == cpu.size());
		for (std::size_t i = 0; i < cpu.size() && i < gpu.size(); ++i)
		{
			CHECK(near(gpu[i], cpu[i], tolerance));
		}
	}
}  // namespace test
