# CMake is a build-system generator: this file describes WHAT to build, and CMake
# generates the actual build files (Makefiles, Ninja, Xcode, ...) for your machine.
#
# Typical use, run from the repo root:
#   cmake -S . -B build        # "configure": read this file, generate build files in ./build
#   cmake --build build        # "build": compile and link
# Options are passed at configure time with -D, e.g.  cmake -S . -B build -DGEOM_ENABLE_CUDA=OFF

# The oldest CMake version this file works with. CMake refuses to run if it is older.
# 3.24 is needed for CMAKE_CUDA_ARCHITECTURES=native (use the GPU installed in this machine).
cmake_minimum_required(VERSION 3.24)

# Names the project and enables the languages. Only C++ is enabled here; CUDA is switched
# on further down, and only if a CUDA compiler exists.
project(GeometryOnGPU LANGUAGES CXX)

# ---- C++ language settings ---------------------------------------------------------------
# These are plain variables. CMake reads them when it creates each target below.
set(CMAKE_CXX_STANDARD 23)            # compile with -std=c++23
set(CMAKE_CXX_STANDARD_REQUIRED ON)   # error out instead of silently falling back to older C++
set(CMAKE_CXX_EXTENSIONS OFF)         # use -std=c++23, not the compiler-specific -std=gnu++23

# Writes build/compile_commands.json, which editors (VS Code clangd, etc.) use for
# code completion and error squiggles.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# If the user did not choose a build type, use Release (optimized, no debug info).
# Other values: Debug (no optimization, debug symbols), RelWithDebInfo.
# The condition skips this for IDE-style generators that choose the type at build time.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Optional CUDA support ---------------------------------------------------------------
# option(NAME "description" DEFAULT) creates a true/false setting the user can change with
# -DNAME=ON/OFF. It lets us turn CUDA off even on a machine that has it.
option(GEOM_ENABLE_CUDA "Build CUDA code when a CUDA compiler is available" ON)

# Our own flag, set to ON only if CUDA is really usable below.
set(GEOM_HAS_CUDA OFF)

if(GEOM_ENABLE_CUDA)
	# check_language(CUDA) looks for nvcc and sets CMAKE_CUDA_COMPILER if it finds one.
	# It does not fail when nothing is found, which is why we use it instead of enable_language.
	include(CheckLanguage)
	check_language(CUDA)
	
	if(CMAKE_CUDA_COMPILER)
		# GPU architectures to generate code for. "native" = the GPU in this machine.
		# Override with e.g. -DCMAKE_CUDA_ARCHITECTURES="80;86" to target other GPUs.
		if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
			set(CMAKE_CUDA_ARCHITECTURES native)
		endif()
		
		# nvcc does not support C++23, so .cu files use C++20. Keep C++23-only features
		# (std::print, std::expected, ...) in .cpp files, never in .cu files or in headers
		# that .cu files include.
		set(CMAKE_CUDA_STANDARD 20)
		set(CMAKE_CUDA_STANDARD_REQUIRED ON)
		
		enable_language(CUDA)         # now .cu files can be compiled with nvcc
		
		# Locates the CUDA runtime libraries and exposes them as targets such as CUDA::cudart.
		find_package(CUDAToolkit REQUIRED)
		
		set(GEOM_HAS_CUDA ON)
	else()
		# message(STATUS ...) prints a note during configure; the build continues without CUDA.
		message(STATUS "No CUDA compiler found: building without CUDA")
	endif()
endif()

# ---- Helpers -----------------------------------------------------------------------------
# A function bundles commands we need for several targets. ${target} is its argument.
# Compiler warnings: MSVC (Visual Studio) uses /W4; GCC and Clang use the -W flags.
# A warning often points at a real bug, so we keep them on.
function(geom_enable_warnings target)
	if(MSVC)
		target_compile_options(${target} PRIVATE /W4)
	else()
		target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
	endif()
endfunction()

# ---- The geometry library ----------------------------------------------------------------
# The headers in src/geom (NurbsCurve, Point3, ...) are templates, so there is nothing to
# compile on their own. An INTERFACE library is a target with no compiled code: it only
# carries settings to whatever links to it. Here that is the include directory, so any
# target that links "geom" can write #include "geom/nurbs_curve.hpp".
add_library(geom INTERFACE)
target_include_directories(geom INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# .cu files will include these headers and nvcc has no C++23, so the headers need only C++20.
target_compile_features(geom INTERFACE cxx_std_20)

# ---- The program -------------------------------------------------------------------------
# A "target" is something to build. add_executable(<name> <sources...>) creates an
# executable called geometry_on_gpu from the listed source files. New .cpp/.cu files
# for the program get added to this list.
add_executable(geometry_on_gpu src/main.cpp)

# Requires C++23 for this target (the global setting above already asks for it).
# PRIVATE means the requirement applies to this target only, not to anything that links to it.
target_compile_features(geometry_on_gpu PRIVATE cxx_std_23)
target_link_libraries(geometry_on_gpu PRIVATE geom)
geom_enable_warnings(geometry_on_gpu)

if(GEOM_HAS_CUDA)
	# Adds -DGEOM_HAS_CUDA=1 to every compile command, so C++ code can use
	# "#ifdef GEOM_HAS_CUDA" to include its CUDA parts only when CUDA is available.
	target_compile_definitions(geometry_on_gpu PRIVATE GEOM_HAS_CUDA=1)
	
	# Link against the CUDA runtime library. CUDA::cudart also adds the CUDA include
	# directory, so "#include <cuda_runtime.h>" works in main.cpp.
	target_link_libraries(geometry_on_gpu PRIVATE CUDA::cudart)
endif()

# ---- Tests -------------------------------------------------------------------------------
# enable_testing() turns on CTest, CMake's test runner. add_test registers a program as a
# test; it passes if the program exits with 0. Run all tests with:
#   ctest --test-dir build --output-on-failure
enable_testing()

add_executable(test_nurbs_curve tests/test_nurbs_curve.cpp)
target_compile_features(test_nurbs_curve PRIVATE cxx_std_23)
target_link_libraries(test_nurbs_curve PRIVATE geom)
geom_enable_warnings(test_nurbs_curve)
add_test(NAME nurbs_curve COMMAND test_nurbs_curve)
