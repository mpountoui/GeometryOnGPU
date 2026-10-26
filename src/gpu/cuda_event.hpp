#pragma once

#include <cuda_runtime.h>

namespace gpu
{
	// A CUDA event: a marker that the CPU puts into the GPU's work queue (record()). The GPU stamps it
	// with its own clock when it reaches the marker, i.e. once all work queued before it is done. So the
	// time between two events is what the GPU spent on the work between them, without the CPU-side
	// launch overhead that a host clock would include:
	//
	//   gpu::CudaEvent start{};
	//   gpu::CudaEvent stop{};
	//   start.record();
	//   kernel<<<blocks, threads>>>(...);
	//   stop.record();
	//   double const seconds = gpu::elapsedSeconds(start, stop);   // waits for stop
	//
	// Owns the cudaEvent_t like DeviceBuffer owns its memory: created in the constructor, destroyed in
	// the destructor, move-only. Host-only. Compiled in cuda_event.cpp (the gpu library).
	class CudaEvent
	{
	public:
		CudaEvent();
		
		CudaEvent(CudaEvent&& other) noexcept;
		CudaEvent& operator=(CudaEvent&& other) noexcept;
		CudaEvent(CudaEvent const&) = delete;
		CudaEvent& operator=(CudaEvent const&) = delete;
		~CudaEvent();
		
	public:
		// Queues the marker behind all work already queued (default stream). Returns at once.
		void record();
		
		// Waits until the GPU has reached the marker.
		void synchronize() const;
		
		cudaEvent_t get() const noexcept;
		
	private:
		// Destroys the event (if any) and leaves this object empty.
		void release() noexcept;
		
	private:
		cudaEvent_t m_event{};
	};
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Seconds between two recorded events, start before stop. Waits until the GPU has reached stop.
	// Resolution about 0.5 microseconds.
	double elapsedSeconds(CudaEvent const& start, CudaEvent const& stop);
}  // namespace gpu
