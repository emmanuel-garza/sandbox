#include <omp.h>
#include <iostream>
#include "/usr/include/c++/8/complex"

using namespace std;
using namespace std::complex_literals;

int main()
{
#pragma omp parallel
    {

        std::cout << "Hello from thread " << omp_get_thread_num() << std::endl;
    }

    cout << 1i << endl;

    return 0;
}