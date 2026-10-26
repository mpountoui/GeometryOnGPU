#include <cstddef>
#include <utility>
#include <vector>

#include "check.hpp"
#include "gpu/cuda_event.hpp"
#include "gpu/device_buffer.hpp"

using gpu::CudaEvent;

namespace
{
	// Two events with nothing between them: a tiny non-negative time.
	void testEmptyInterval()
	{
		CudaEvent start{};
		CudaEvent stop{};
		start.record();
		stop.record();
		double const seconds = gpu::elapsedSeconds(start, stop);
		CHECK(seconds >= 0.0 && seconds < 1e-3);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// Copying 64 MB to the GPU takes measurable time (milliseconds over PCIe).
	void testMeasuresWork()
	{
		std::vector<float> const host(std::size_t{16} << 20, 1.0f);
		gpu::DeviceBuffer<float> buffer(host.size());
		CudaEvent start{};
		CudaEvent stop{};
		start.record();
		buffer.upload(host);
		stop.record();
		CHECK(gpu::elapsedSeconds(start, stop) > 1e-5);
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	// A move hands the event over and leaves the source empty.
	void testMove()
	{
		CudaEvent a{};
		cudaEvent_t const handle = a.get();
		CudaEvent b(std::move(a));
		CHECK(b.get() == handle);
		CHECK(a.get() == nullptr);
		CudaEvent c{};
		c = std::move(b);
		CHECK(c.get() == handle);
		CHECK(b.get() == nullptr);
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main()
{
	testEmptyInterval();
	testMeasuresWork();
	testMove();
	return test::report("cuda_event");
}
