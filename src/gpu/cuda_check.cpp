#include "gpu/cuda_check.hpp"

#include <format>
#include <stdexcept>

namespace gpu
{
	void checkCuda(cudaError_t error, std::source_location location)
	{
		if (error == cudaSuccess)
		{
			return;
		}
		throw std::runtime_error(std::format("{}:{}: CUDA error {}: {}", location.file_name(), location.line(), cudaGetErrorName(error), cudaGetErrorString(error)));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void checkLaunch(std::source_location location)
	{
		checkCuda(cudaGetLastError(), location);
	}
}  // namespace gpu
