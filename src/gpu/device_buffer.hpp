#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/cuda_check.hpp"

namespace gpu
{
	// An array of count T's in GPU (device) memory, owned like a std::vector owns host memory:
	// allocated in the constructor (cudaMalloc), freed in the destructor (cudaFree).
	//
	// The CPU cannot dereference data(): it is an address in GPU memory. Pass it to a kernel, and
	// move data with upload() (host -> device) and download() (device -> host). Both copies are
	// synchronous: they return once the data has arrived.
	//
	// Move-only: copying would mean two owners freeing the same memory. T must be trivially
	// copyable because the copies are raw bytes (no constructors run on the GPU side).
	template <typename T>
	requires std::is_trivially_copyable_v<T>
	class DeviceBuffer
	{
	public:
		DeviceBuffer() = default;

		// Uninitialized device memory for count elements.
		explicit DeviceBuffer(std::size_t count);

		// Device memory with a copy of host.
		explicit DeviceBuffer(std::span<T const> host);

		DeviceBuffer(DeviceBuffer&& other) noexcept;
		DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
		DeviceBuffer(DeviceBuffer const&) = delete;
		DeviceBuffer& operator=(DeviceBuffer const&) = delete;
		~DeviceBuffer();

	public:
		T* data() noexcept;
		T const* data() const noexcept;
		std::size_t size() const noexcept;
		std::size_t sizeBytes() const noexcept;

		// Copies host into the buffer. Throws std::invalid_argument if the sizes differ.
		void upload(std::span<T const> host);

		// Copies the buffer into host. Throws std::invalid_argument if the sizes differ.
		void download(std::span<T> host) const;

		// Copies the buffer into a new std::vector.
		std::vector<T> download() const;

	private:
		// Frees the memory (if any) and leaves the buffer empty.
		void release() noexcept;

	private:
		T* m_data{};
		std::size_t m_size{};

	private:
		static void checkSameSize(std::size_t bufferSize, std::size_t hostSize);
	};

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	template <typename T>
	requires std::is_trivially_copyable_v<T>
	DeviceBuffer<T>::DeviceBuffer(std::size_t count)
		: m_size(count)
	{
		if (count > 0)
		{
			checkCuda(cudaMalloc(reinterpret_cast<void**>(&m_data), count * sizeof(T)));
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	template <typename T>
	requires std::is_trivially_copyable_v<T>
	DeviceBuffer<T>::DeviceBuffer(std::span<T const> host)
		: DeviceBuffer(host.size())
	{
		upload(host);
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	// Takes over other's memory and leaves other empty, so only one destructor frees it.
	template <typename T>
	requires std::is_trivially_copyable_v<T>
	DeviceBuffer<T>::DeviceBuffer(DeviceBuffer&& other) noexcept
		: m_data(std::exchange(other.m_data, nullptr)),
		m_size(std::exchange(other.m_size, 0))
	{
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	template <typename T>
	requires std::is_trivially_copyable_v<T>
	DeviceBuffer<T>& DeviceBuffer<T>::operator=(DeviceBuffer&& other) noexcept
	{
		if (this != &other)
		{
			release();
			m_data = std::exchange(other.m_data, nullptr);
			m_size = std::exchange(other.m_size, 0);
		}
		return *this;
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	template <typename T>
	requires std::is_trivially_copyable_v<T>
	DeviceBuffer<T>::~DeviceBuffer()
	{
		release();
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	template <typename T>
	requires std::is_trivially_copyable_v<T>
	T* DeviceBuffer<T>::data() noexcept
	{
		return m_data;
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	template <typename T>
	requires std::is_trivially_copyable_v<T>
	T const* DeviceBuffer<T>::data() const noexcept
	{
		return m_data;
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	template <typename T>
	requires std::is_trivially_copyable_v<T>
	std::size_t DeviceBuffer<T>::size() const noexcept
	{
		return m_size;
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	template <typename T>
	requires std::is_trivially_copyable_v<T>
	std::size_t DeviceBuffer<T>::sizeBytes() const noexcept
	{
		return m_size * sizeof(T);
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	template <typename T>
	requires std::is_trivially_copyable_v<T>
	void DeviceBuffer<T>::upload(std::span<T const> host)
	{
		checkSameSize(m_size, host.size());
		if (m_size > 0)
		{
			checkCuda(cudaMemcpy(m_data, host.data(), sizeBytes(), cudaMemcpyHostToDevice));
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	template <typename T>
	requires std::is_trivially_copyable_v<T>
	void DeviceBuffer<T>::download(std::span<T> host) const
	{
		checkSameSize(m_size, host.size());
		if (m_size > 0)
		{
			checkCuda(cudaMemcpy(host.data(), m_data, sizeBytes(), cudaMemcpyDeviceToHost));
		}
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	template <typename T>
	requires std::is_trivially_copyable_v<T>
	std::vector<T> DeviceBuffer<T>::download() const
	{
		std::vector<T> host(m_size);
		download(std::span<T>(host));
		return host;
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	// A destructor must not throw, so a cudaFree error is ignored here. It can only fail if the
	// GPU is already in an error state, which an earlier checkCuda call will have reported.
	template <typename T>
	requires std::is_trivially_copyable_v<T>
	void DeviceBuffer<T>::release() noexcept
	{
		if (m_data != nullptr)
		{
			cudaFree(m_data);
		}
		m_data = nullptr;
		m_size = 0;
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	template <typename T>
	requires std::is_trivially_copyable_v<T>
	void DeviceBuffer<T>::checkSameSize(std::size_t bufferSize, std::size_t hostSize)
	{
		if (bufferSize != hostSize)
		{
			throw std::invalid_argument("DeviceBuffer: host and device sizes differ");
		}
	}
}  // namespace gpu
