#include <iostream>
#include <vector>
#include <omp.h>
// #include <cmath>

// For double precision
using precision = double;
#define wavenum 1.0
#define PI_R 3.14159265358979323846
#define cu_sqrt sqrt
#define cu_cos cos

// // For single precision
// using precision = float;
// #define wavenum 1.0f
// #define PI_R 3.14159265358979323846f
// #define cu_sqrt sqrtf
// #define cu_cos cosf


// Literals for constants, need one for the host and one for the device
__host__ precision operator"" _d(long double v) { return (precision)v; }
__device__ precision operator"" _dd(long double v) { return (precision)v; }


// Compute distance in the GPU
__device__ precision distance(
    precision x0_src, precision x1_src, precision x2_src,
    precision x0_trg, precision x1_trg, precision x2_trg)
{
    return cu_sqrt(
        (x0_src - x0_trg) * (x0_src - x0_trg) +
        (x1_src - x1_trg) * (x1_src - x1_trg) +
        (x2_src - x2_trg) * (x2_src - x2_trg));
        
}

// Launch Kernel for the GPU
__global__ void integralKernel(
    precision *integrals_out,
    precision *x0_src, precision *x1_src, precision *x2_src,
    precision *x0_trg, precision *x1_trg, precision *x2_trg,
    int len_src)
{
    // Get the index that corresponds to this thread/block pair
    const int ind_trg = blockIdx.x * blockDim.x + threadIdx.x;

    integrals_out[ind_trg] = 0.0_dd;

    for (int ind_src = 0; ind_src < len_src; ind_src++)
    {
        precision r = distance(
            x0_src[ind_src], x1_src[ind_src], x2_src[ind_src],
            x0_trg[ind_trg], x1_trg[ind_trg], x2_trg[ind_trg]);

        integrals_out[ind_trg] += cu_cos(wavenum * r) / (4.0_dd * PI_R * r);
    }

    return;
}


int main()
{
    double t1, t2;
    
    // Lets initialize a vector for the coordinates
    int nu = 1000;
    int nv = 512*2;

    int n = nu * nv;
    
    std::cout << "Total Number of Points = " << n << std::endl;

    int ind_sample = n-1;

    // Allocate the host vectors
    std::vector<precision> x0_src(n), x1_src(n), x2_src(n);
    std::vector<precision> x0_trg(n), x1_trg(n), x2_trg(n);
    std::vector<precision> integrals(n);

    int k = -1;
    for (int i = 0; i < nu; i++)
    {
        for (int j = 0; j < nv; j++)
        {
            k++;

            x0_src[k] = 1.0_d * i / (nu - 1);
            x1_src[k] = 1.0_d * j / (nv - 1);
            x2_src[k] = 1.0_d;

            x0_trg[k] = 1.0_d * i / (nu - 1);
            x1_trg[k] = 1.0_d * j / (nv - 1);
            x2_trg[k] = 0.0_d;
        }
    }


    // Computation in the GPU

    t1 = omp_get_wtime();

    precision *integrals_cu = 0;

    precision *x0_src_cu = 0;
    precision *x1_src_cu = 0;
    precision *x2_src_cu = 0;

    precision *x0_trg_cu = 0;
    precision *x1_trg_cu = 0;
    precision *x2_trg_cu = 0;

    // Allocate memory in the GPU
    cudaMalloc(&integrals_cu, n * sizeof(precision));

    cudaMalloc(&x0_src_cu, n * sizeof(precision));
    cudaMalloc(&x1_src_cu, n * sizeof(precision));
    cudaMalloc(&x2_src_cu, n * sizeof(precision));

    cudaMalloc(&x0_trg_cu, n * sizeof(precision));
    cudaMalloc(&x1_trg_cu, n * sizeof(precision));
    cudaMalloc(&x2_trg_cu, n * sizeof(precision));

    // Copy from the device to the GPU
    cudaMemcpy(x0_src_cu, &(x0_src[0]), n*sizeof(precision), cudaMemcpyHostToDevice);
    cudaMemcpy(x1_src_cu, &(x1_src[0]), n*sizeof(precision), cudaMemcpyHostToDevice);
    cudaMemcpy(x2_src_cu, &(x2_src[0]), n*sizeof(precision), cudaMemcpyHostToDevice);

    cudaMemcpy(x0_trg_cu, &(x0_trg[0]), n*sizeof(precision), cudaMemcpyHostToDevice);
    cudaMemcpy(x1_trg_cu, &(x1_trg[0]), n*sizeof(precision), cudaMemcpyHostToDevice);
    cudaMemcpy(x2_trg_cu, &(x2_trg[0]), n*sizeof(precision), cudaMemcpyHostToDevice);

    //Launch Cuda Kernel
    integralKernel<<<nu, nv>>>(
        integrals_cu,
        x0_src_cu, x1_src_cu, x2_src_cu,
        x0_trg_cu, x1_trg_cu, x2_trg_cu,
        n);

    // Copy to device
    cudaMemcpy(&(integrals[0]), integrals_cu, n*sizeof(precision), cudaMemcpyDeviceToHost);

    t2 = omp_get_wtime();
    std::cout << integrals[ind_sample] << std::endl;
    std::cout << "Time GPU = " << t2 - t1 << " seconds" << std::endl;

    // Free Device Memory
    cudaFree(integrals_cu);

    cudaFree(x0_src_cu);
    cudaFree(x1_src_cu);
    cudaFree(x2_src_cu);

    cudaFree(x0_trg_cu);
    cudaFree(x1_trg_cu);
    cudaFree(x2_trg_cu);

    return 0;

    // Now we do it in the CPU
    t1 = omp_get_wtime();

#pragma omp parallel
    {
#pragma omp for schedule(guided, 1)
        for(int ind_trg = 0; ind_trg < n; ind_trg++)
        {
            integrals[ind_trg] = 0.0_d;
            for(int ind_src = 0; ind_src < n; ind_src++)
            {
                precision r = std::sqrt(
                    std::pow(x0_src[ind_src] - x0_trg[ind_trg], 2) +
                    std::pow(x1_src[ind_src] - x1_trg[ind_trg], 2) +
                    std::pow(x2_src[ind_src] - x2_trg[ind_trg], 2));

                integrals[ind_trg] += std::cos(wavenum * r) / (4.0_d * PI_R * r);
            }
        }
    }

    t2 = omp_get_wtime();
    std::cout << integrals[ind_sample] << std::endl;
    std::cout << "Time CPU = " << t2 - t1 << " seconds" << std::endl;

    return 0;
}