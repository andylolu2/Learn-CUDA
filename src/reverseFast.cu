#include <cassert>
#include <iostream>

#include "lib/utils.cuh"

#define N (1 << 28)
#define N_THREADS 256

__global__ void kernel(float *a, float *b) {
    size_t offset = threadIdx.x;
    size_t start = blockIdx.x * blockDim.x;
    size_t idx = start + offset;

    __shared__ float s[N_THREADS];

    // read from global memory (in linear order) to shared memory (in reverse order)
    if (idx < N) {
        s[N_THREADS - offset - 1] = a[idx];
    }

    __syncthreads();

    // write back to global memory (in linear order for coalescing)
    size_t rStart = (gridDim.x - blockIdx.x - 1) * blockDim.x;
    size_t dstIdx = rStart + offset;
    if (dstIdx < N) {
        size_t srcIdx;
        if (N - dstIdx >= N_THREADS) {
            srcIdx = offset;
        } else {
            // account for padding if last block
            size_t padding = (N_THREADS - (N % N_THREADS)) % N_THREADS;
            srcIdx = padding + offset;
        }
        b[dstIdx] = s[srcIdx];
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
