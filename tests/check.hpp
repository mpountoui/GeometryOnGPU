#pragma once

#include <cmath>
#include <print>

// Minimal test support shared by the test executables: CHECK records a failure and keeps going,
// test::report prints the summary and gives the process exit code.

namespace test
{
	inline int failures = 0;
	
	// Exit code for main(): 0 if every CHECK passed.
	inline int report(char const* suiteName);
	
	// |a - b| <= tolerance.
	template <typename T>
	bool near(T a, T b, T tolerance);
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	inline int report(char const* suiteName)
	{
		if (failures > 0)
		{
			std::println(stderr, "{}: {} check(s) failed", suiteName, failures);
			return 1;
		}
		std::println("All {} tests passed", suiteName);
		return 0;
	}
	
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
	
	template <typename T>
	bool near(T a, T b, T tolerance)
	{
		return std::abs(a - b) <= tolerance;
	}
}  // namespace test

/*---------------------------------------------------------------------------------------------------------------------------------------------*/

#define CHECK(cond) \
	do \
	{ \
		if (!(cond)) \
		{ \
			std::println(stderr, "{}:{}: CHECK failed: {}", __FILE__, __LINE__, #cond); \
			++test::failures; \
		} \
	} while (false)
