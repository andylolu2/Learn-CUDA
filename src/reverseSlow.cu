#include <cassert>
#include <iostream>

#include "lib/utils.cuh"

#define N (1 << 28)
#define N_THREADS 256

__global__ void kernel(float *a, float *b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        b[N - i - 1] = a[i];
    }
}

int main(void) {
    // create timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // create array
    float *a_host = (float *)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        a_host[i] = i;
    }

    // allocate device memory
    float *a_device, *b_device;
    checkCudaStatus(cudaMalloc(&a_device, N * sizeof(float)));
    checkCudaStatus(cudaMalloc(&b_device, N * sizeof(float)));

    // copy initial array to device
    checkCudaStatus(cudaMemcpy(a_device, a_host, N * sizeof(float), cudaMemcpyHostToDevice));

    // reverse array
    cudaEventRecord(start);
    kernel<<<((N + N_THREADS - 1) / N_THREADS), N_THREADS>>>(a_device, b_device);
    cudaEventRecord(stop);

    // copy result back
    float *b_host = (float *)malloc(N * sizeof(float));
    checkCudaStatus(cudaMemcpy(b_host, b_device, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // deallocate device memory
    checkCudaStatus(cudaFree(a_device));
    checkCudaStatus(cudaFree(b_device));

    // assert result
    for (size_t i = 0; i < N; i++) {
        assert(a_host[i] == b_host[N - i - 1]);
    }

    printf("Time elapsed: %fms\n", milliseconds);

    return 0;
}
