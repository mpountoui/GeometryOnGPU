#include "gpu/cuda_event.hpp"

#include <utility>

#include "gpu/cuda_check.hpp"

namespace gpu
{
	CudaEvent::CudaEvent()
	{
		checkCuda(cudaEventCreate(&m_event));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Takes over other's event and leaves other empty, so only one destructor destroys it.
	CudaEvent::CudaEvent(CudaEvent&& other) noexcept
		: m_event(std::exchange(other.m_event, nullptr))
	{
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept
	{
		if (this != &other)
		{
			release();
			m_event = std::exchange(other.m_event, nullptr);
		}
		return *this;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	CudaEvent::~CudaEvent()
	{
		release();
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void CudaEvent::record()
	{
		checkCuda(cudaEventRecord(m_event));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	void CudaEvent::synchronize() const
	{
		checkCuda(cudaEventSynchronize(m_event));
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	cudaEvent_t CudaEvent::get() const noexcept
	{
		return m_event;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// As in DeviceBuffer::release, an error here is ignored: a destructor must not throw.
	void CudaEvent::release() noexcept
	{
		if (m_event != nullptr)
		{
			cudaEventDestroy(m_event);
		}
		m_event = nullptr;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// cudaEventElapsedTime gives milliseconds as a float.
	double elapsedSeconds(CudaEvent const& start, CudaEvent const& stop)
	{
		stop.synchronize();
		float milliseconds = 0.0f;
		checkCuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()));
		return static_cast<double>(milliseconds) * 1e-3;
	}
}  // namespace gpu
