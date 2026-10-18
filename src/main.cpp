#include <iostream>

#ifdef GEOM_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace
{
#ifdef GEOM_HAS_CUDA
	// Returns false (after printing the error) if a CUDA call failed.
	bool checkCuda(cudaError_t err)
	{
		if (err == cudaSuccess)
		{
			return true;
		}
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << '\n';
		return false;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	bool printCudaDevice(int index)
	{
		cudaDeviceProp prop{};
		if (!checkCuda(cudaGetDeviceProperties(&prop, index)))
		{
			return false;
		}
		std::cout << "CUDA device " << index << ": " << prop.name << " (sm_" << prop.major << prop.minor << ")\n";
		return true;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	bool printCudaDevices()
	{
		int deviceCount = 0;
		if (!checkCuda(cudaGetDeviceCount(&deviceCount)))
		{
			return false;
		}
		for (int i = 0; i < deviceCount; ++i)
		{
			if (!printCudaDevice(i))
			{
				return false;
			}
		}
		if (deviceCount == 0)
		{
			std::cout << "No CUDA devices found\n";
		}
		return true;
	}
#else
	bool printCudaDevices()
	{
		std::cout << "Built without CUDA\n";
		return true;
	}
#endif
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main()
{
	std::cout << "GeometryOnGPU\n";
	return printCudaDevices() ? 0 : 1;
}
