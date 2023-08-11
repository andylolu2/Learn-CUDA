#include <stdint.h>
#include <stdio.h>

#include <iostream>

#include "lib/utils.cuh"
#define N (1 << 10)

__global__ void kernel(float* array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    array[idx] = 1000 * blockIdx.x + threadIdx.x;
}

int main(void) {
    // allocate device memory
    float* a_device;
    checkCudaStatus(cudaMalloc(&a_device, N * sizeof(float)));

    kernel<<<1, N>>>(a_device);

    // allocate host memory
    float* a_host = (float*)malloc(N * sizeof(float));

    // do the work
    checkCudaStatus(cudaMemcpy(a_host, a_device, N * sizeof(float), cudaMemcpyDeviceToHost));

    // deallocate
    checkCudaStatus(cudaFree(a_device));

    return 0;
}
