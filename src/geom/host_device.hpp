#pragma once

// GEOM_HD marks a function as callable from both CPU (host) and GPU (device) code.
//
// nvcc compiles such a function twice: once for the CPU and once for the GPU. nvcc defines
// __CUDACC__; a plain C++ compiler (g++, clang) does not know __host__ / __device__, so for it
// the macro expands to nothing and the function is ordinary C++.
#ifdef __CUDACC__
#define GEOM_HD __host__ __device__
#else
#define GEOM_HD
#endif
