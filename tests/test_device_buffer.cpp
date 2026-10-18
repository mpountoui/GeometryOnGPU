#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "geom/point.hpp"
#include "gpu/cuda_check.hpp"
#include "gpu/device_buffer.hpp"

using gpu::DeviceBuffer;

namespace
{
	// Host -> device -> host gives back the same values.
	void testRoundTrip()
	{
		std::vector<double> const host{1.5, -2.0, 3.25, 1e-300};
		DeviceBuffer<double> const buffer{std::span<double const>(host)};
		CHECK(buffer.size() == host.size());
		CHECK(buffer.sizeBytes() == host.size() * sizeof(double));
		CHECK(buffer.data() != nullptr);
		CHECK(buffer.download() == host);
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	// The control point type: 16-byte aligned Point4<float> survives the copy unchanged.
	void testRoundTripPoint4()
	{
		std::vector<geom::Point4<float>> const host{{1, 2, 3, 1}, {2, 4, 6, 2}, {0.5f, 0, 0, 0.5f}};
		DeviceBuffer<geom::Point4<float>> buffer(host.size());
		buffer.upload(host);
		std::vector<geom::Point4<float>> back(host.size());
		buffer.download(back);
		CHECK(back == host);
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	// Moving transfers ownership: the source is left empty, so its destructor frees nothing.
	void testMove()
	{
		std::vector<int> const host{1, 2, 3};
		DeviceBuffer<int> a{std::span<int const>(host)};
		int const* const address = a.data();
		DeviceBuffer<int> b(std::move(a));
		CHECK(a.data() == nullptr && a.size() == 0);
		CHECK(b.data() == address && b.download() == host);
		DeviceBuffer<int> c{};
		c = std::move(b);
		CHECK(b.data() == nullptr && c.data() == address);
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	void testEmpty()
	{
		DeviceBuffer<float> const buffer(0);
		CHECK(buffer.data() == nullptr && buffer.size() == 0);
		CHECK(buffer.download().empty());
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	void testSizeMismatchThrows()
	{
		DeviceBuffer<float> buffer(3);
		std::vector<float> const tooShort(2);
		bool thrown = false;
		try
		{
			buffer.upload(tooShort);
		}
		catch (std::invalid_argument const&)
		{
			thrown = true;
		}
		CHECK(thrown);
	}

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

	// checkCuda passes cudaSuccess through and throws on an error, naming this file.
	void testCheckCuda()
	{
		gpu::checkCuda(cudaSuccess);
		std::string message{};
		try
		{
			gpu::checkCuda(cudaErrorInvalidValue);
		}
		catch (std::runtime_error const& error)
		{
			message = error.what();
		}
		CHECK(message.find("test_device_buffer.cpp") != std::string::npos);
		CHECK(message.find("cudaErrorInvalidValue") != std::string::npos);
	}
}  // namespace

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

int main()
{
	testRoundTrip();
	testRoundTripPoint4();
	testMove();
	testEmpty();
	testSizeMismatchThrows();
	testCheckCuda();
	return test::report("device_buffer");
}
