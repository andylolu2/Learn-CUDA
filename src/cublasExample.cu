// Example 2. Application Using C and cuBLAS: 0-based indexing
//-----------------------------------------------------------
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cublas_v2.h"
#include "lib/utils.cuh"

#define SIZES_START 8
#define SIZES_END 8
#define TIMES 50000
#define WARMUP 5000

int main(void) {
    int i, j, N;

    FILE* fptr;
    fptr = fopen("timings.txt", "w");
    if (fptr == NULL) {
        printf("Error!");
        exit(1);
    }

    // For loop over sizes
    for (N = SIZES_START; N <= SIZES_END; N++) {
        float* a = (float*)malloc(N * N * sizeof(float));
        float* b = (float*)malloc(N * N * sizeof(float));
        float* c = (float*)malloc(N * N * sizeof(float));
        // float* d = (float*)malloc(N * N * sizeof(float));

        // initialize matrix a
        for (j = 0; j < N; j++) {
            for (i = 0; i < N; i++) {
                a[IDX2C(i, j, N)] = ((float)(i * N + j + 1)) / ((float)(N * N));
            }
        }
        // print2DArray(a, N, N);

        float* A;
        checkCudaStatus(cudaMalloc((void**)&A, N * N * sizeof(*a)));
        float* B;
        checkCudaStatus(cudaMalloc((void**)&B, N * N * sizeof(*b)));
        float* C;
        checkCudaStatus(cudaMalloc((void**)&C, N * N * sizeof(*c)));
        // float* D;
        // checkCudaStatus(cudaMalloc((void**)&D, N * N * sizeof(*d)));

        // create cublasLt handle
        cublasHandle_t handle;
        checkCublasStatus(cublasCreate(&handle));

        checkCublasStatus(cublasSetMatrix(N, N, sizeof(*a), a, N, A, N));

        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        // Perform GEMM AA^T + 0
        float alpha = 1.0f;
        float beta = 0.0f;
        for (int i = 0; i < WARMUP; i++) {
            checkCublasStatus(
                cublasSgemm(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    N,
                    N,
                    N,
                    &alpha,
                    A,
                    N,
                    A,
                    N,
                    &beta,
                    C,
                    N));
        }
        cudaEventRecord(start);
        for (int i = 0; i < TIMES; i++) {
            checkCublasStatus(
                cublasSgemm(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    N,
                    N,
                    N,
                    &alpha,
                    A,
                    N,
                    A,
                    N,
                    &beta,
                    C,
                    N));
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, end);

        printf("N = %d, %.4f ops/ms\n", N, TIMES / milliseconds);
        fprintf(fptr, "%d,%f\n", N, TIMES / milliseconds);
        fflush(fptr);

        // Deallocate
        checkCublasStatus(cublasDestroy(handle));

        checkCublasStatus(cublasGetMatrix(N, N, sizeof(*a), C, N, c, N));
        print2DArray(a, N, N);

        checkCudaStatus(cudaFree(A));
        checkCudaStatus(cudaFree(B));
        checkCudaStatus(cudaFree(C));
        free(a);
        free(b);
        free(c);
    }

    fclose(fptr);
    return EXIT_SUCCESS;
}