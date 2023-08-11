#include <stdio.h>

#include "lib/utils.cuh"

int main(void) {
    // allocate host value
    float n = 1234.56f;
    float *a_host = &n;

    // allocate device memory
    float *a_device, *b_device;
    checkCudaStatus(cudaMalloc(&a_device, sizeof(float)));
    checkCudaStatus(cudaMalloc(&b_device, sizeof(float)));

    // shuffle things around
    checkCudaStatus(cudaMemcpy(a_device, a_host, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemcpy(b_device, a_device, sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaStatus(cudaMemcpy(a_host, b_device, sizeof(float), cudaMemcpyDeviceToHost));

    // free memory
    checkCudaStatus(cudaFree(a_device));
    checkCudaStatus(cudaFree(b_device));

    printf("%f\n", *a_host);

    return 0;
}
