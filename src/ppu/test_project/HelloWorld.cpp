#include <s2pp.h>

template<int N>
class HelloWorld
{
public:
	HelloWorld() {}

private:
	int member = N;
};

extern "C" {
void
start(void)
{
	vector uint8_t lhs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
}
}
