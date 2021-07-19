#include <iostream>
#include <vector>
#include <omp.h>

#include <stdio.h>

__global__ void testKernel()
{
    const int ind_th = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Hello from thread %d\n", ind_th);
}

int main()
{
    std::cout << "Hello World!" << std::endl;

    int n_blocks = 10;
    int n_th_per_block = 25;

    testKernel<<<n_blocks, n_th_per_block>>>();

    cudaDeviceSynchronize();

    return 0;
}