#include <stdio.h>
#include <thrust/device_vector.h>


//64
#define N 64
//32 // Threads per block
#define TPB 32

__device__ float scale(int i, int n)
{
    return ((float)i) / (n - 1);
}

__device__ float distance(float x1, float x2)
{
    return sqrt((x2 - x1) * (x2 - x1));
}

// __global__ void distanceKernel( float *d_out, float ref, int len )
// {
//   const int i = blockIdx.x*blockDim.x + threadIdx.x;
//   const float x = scale( i, len );
//   d_out[i] = distance( x, ref );
//   printf( "i = %2d: dist from %f to %f is %f.\n", i, ref, x, d_out[i] );
// }

__global__ void distanceKernel(float *d_out, float ref, int len)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float x = scale(i, len);

    printf("Hello from block %2d (%2d), thread %2d\n", blockIdx.x, blockDim.x, threadIdx.x);

    // printf( "asdf" );

    for (int j = 0; j < 1000000; ++j)
    {
        d_out[i] = distance(x, ref);
        // printf( "i = %2d: dist from %f to %f is %f.\n", i, ref, x, d_out[i] );
    }
}

int main()
{
    const float ref = 0.5f;
    const int repeat = 1;

    printf("Hello World\n");

    // Pointer for an array of floats (initizlied to zero - null)
    float *d_out = 0;

    // Allocate device memory to store the output array
    cudaMalloc(&d_out, N * sizeof(float));

    // Launch kernel to compute and store distance values
    for (int i = 0; i < repeat; ++i)
    {
        // distanceKernel<<<N/TPB, TPB>>>( d_out, ref, N );
        distanceKernel<<<2, 32>>>(d_out, ref, N);
    }

    // Free Memory
    cudaFree(d_out);

    printf("Bye\n");

    return 0;
}
