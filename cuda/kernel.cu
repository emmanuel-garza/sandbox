#include <iostream>

// To use containers like the STL but for cuda
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

#include <vector>
#include <omp.h>
#include <cmath>

using precision = float;
precision operator"" _d(long double v) { return (precision)v; }

// const precision wavenum = 1.0_d;
// const precision PI_R = (precision)M_PI;

#define wavenum 1.0f
#define PI_R 3.14159265358979323846f

// // The Green's function for real
// __device__ precision greens_function_re(precision r)
// {
//     return;
// }

// __device__ precision greens_function_re(
//     precision x0_src, precision x1_src, precision x2_src,
//     precision x0_trg, precision x1_trg, precision x2_trg)
// {
//     precision r = sqrtf(
//         powf(x0_src - x0_trg, 2) +
//         powf(x1_src - x1_trg, 2) +
//         powf(x2_src - x2_trg, 2) +);

//     return;
// }

__device__ precision distance(
    precision x0_src, precision x1_src, precision x2_src,
    precision x0_trg, precision x1_trg, precision x2_trg)
{
    return sqrtf(
        (x0_src - x0_trg) * (x0_src - x0_trg) +
        (x1_src - x1_trg) * (x1_src - x1_trg) +
        (x2_src - x2_trg) * (x2_src - x2_trg));
        
}

__global__ void integralKernel(
    precision *integrals_out,
    precision *x0_src, precision *x1_src, precision *x2_src,
    precision *x0_trg, precision *x1_trg, precision *x2_trg,
    int len_src)
{
    // Get the index that corresponds to this thread/block pair
    const int ind_trg = blockIdx.x * blockDim.x + threadIdx.x;

    integrals_out[ind_trg] = 0.0f;

    for (int ind_src = 0; ind_src < len_src; ind_src++)
    {
        precision r = distance(
            x0_src[ind_src], x1_src[ind_src], x2_src[ind_src],
            x0_trg[ind_trg], x1_trg[ind_trg], x2_trg[ind_trg]);

        integrals_out[ind_trg] += cosf(wavenum * r) / (4.0f * PI_R * r);
    }

    return;
}


int main()
{
    std::cout << "Hello!" << std::endl;

    double t1, t2;
    
    // Lets initialize a vector for the coordinates
    int nu = 100;
    int nv = 512;

    int n = nu * nv;

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


    // Allocate device memory to store and pass the arrays

    t1 = omp_get_wtime();

    float *integrals_cu = 0;

    float *x0_src_cu = 0;
    float *x1_src_cu = 0;
    float *x2_src_cu = 0;

    float *x0_trg_cu = 0;
    float *x1_trg_cu = 0;
    float *x2_trg_cu = 0;



    cudaMalloc(&integrals_cu, n * sizeof(float));

    cudaMalloc(&x0_src_cu, n * sizeof(float));
    cudaMalloc(&x1_src_cu, n * sizeof(float));
    cudaMalloc(&x2_src_cu, n * sizeof(float));

    cudaMalloc(&x0_trg_cu, n * sizeof(float));
    cudaMalloc(&x1_trg_cu, n * sizeof(float));
    cudaMalloc(&x2_trg_cu, n * sizeof(float));


    // Copy from the device to the GPU
    cudaMemcpy(x0_src_cu, &(x0_src[0]), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x1_src_cu, &(x1_src[0]), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x2_src_cu, &(x2_src[0]), n*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(x0_trg_cu, &(x0_trg[0]), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x1_trg_cu, &(x1_trg[0]), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x2_trg_cu, &(x2_trg[0]), n*sizeof(float), cudaMemcpyHostToDevice);


    //Launch Cuda Kernel
    integralKernel<<<nu, nv>>>(
        integrals_cu,
        x0_src_cu, x1_src_cu, x2_src_cu,
        x0_trg_cu, x1_trg_cu, x2_trg_cu,
        n);

    // Copy to device
    cudaMemcpy(&(integrals[0]), integrals_cu, n*sizeof(float), cudaMemcpyDeviceToHost);

    
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


    t1 = omp_get_wtime();

    // Now we do it in the host
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