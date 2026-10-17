#include <iostream>

#ifdef GEOM_HAS_CUDA
#include <cuda_runtime.h>
#endif

int main() {
  std::cout << "GeometryOnGPU\n";

#ifdef GEOM_HAS_CUDA
  int deviceCount = 0;
  const cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << '\n';
    return 1;
  }
  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp prop{};
    const cudaError_t perr = cudaGetDeviceProperties(&prop, i);
    if (perr != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(perr) << '\n';
      return 1;
    }
    std::cout << "CUDA device " << i << ": " << prop.name << " (sm_" << prop.major << prop.minor
              << ")\n";
  }
  if (deviceCount == 0) std::cout << "No CUDA devices found\n";
#else
  std::cout << "Built without CUDA\n";
#endif
  return 0;
}
