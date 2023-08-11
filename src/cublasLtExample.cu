// Example 2. Application Using C and cuBLAS: 0-based indexing
//-----------------------------------------------------------
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cublasLt.h"
#include "lib/utils.cuh"

#define SIZES_START 2048
#define SIZES_END 2048
#define TIMES 10000
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

        // initialize matrix a
        for (j = 0; j < N; j++) {
            for (i = 0; i < N; i++) {
                a[IDX2C(i, j, N)] = ((float)(i * N + j + 1)) / ((float)(N * N));
            }
        }

        float* A;
        checkCudaStatus(cudaMalloc((void**)&A, N * N * sizeof(*a)));
        float* B;
        checkCudaStatus(cudaMalloc((void**)&B, N * N * sizeof(*b)));
        float* C;
        checkCudaStatus(cudaMalloc((void**)&C, N * N * sizeof(*c)));

        // create cublasLt handle
        cublasLtHandle_t handle;
        checkCublasStatus(cublasLtCreate(&handle));

        // create operation desciriptor
        cublasLtMatmulDesc_t operationDesc;
        checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t transa = CUBLAS_OP_N;
        checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        cublasOperation_t transb = CUBLAS_OP_T;
        checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

        checkCublasStatus(cublasSetMatrix(N, N, sizeof(*a), a, N, A, N));

        // create (empty) preference for heuristics
        cublasLtMatmulPreference_t preference;
        checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));

        // create heuristic
        int returnedResults = 0;
        cublasLtMatmulHeuristicResult_t heuristicResult = {};
        cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
        // create matrix descriptors, we are good with the details here so no need to set any extra attributes
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, N, N, N));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, N, N, N));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, N, N, N));

        checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
        if (returnedResults == 0) {
            checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
        }

        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        // Perform GEMM AA^T + 0
        float alpha = 1.0f;
        float beta = 0.0f;
        for (int i = 0; i < WARMUP; i++) {
            checkCublasStatus(
                cublasLtMatmul(
                    handle,
                    operationDesc,
                    &alpha,
                    A,
                    Adesc,
                    B,
                    Bdesc,
                    &beta,
                    C,
                    Cdesc,
                    C,
                    Cdesc,
                    &heuristicResult.algo,
                    nullptr,
                    0,
                    0));
        }
        cudaEventRecord(start);
        for (int i = 0; i < TIMES; i++) {
            checkCublasStatus(
                cublasLtMatmul(
                    handle,
                    operationDesc,
                    &alpha,
                    A,
                    Adesc,
                    B,
                    Bdesc,
                    &beta,
                    C,
                    Cdesc,
                    C,
                    Cdesc,
                    &heuristicResult.algo,
                    nullptr,
                    0,
                    0));
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, end);

        printf("N = %d, %.4f ops/ms\n", N, TIMES / milliseconds);
        fprintf(fptr, "%d,%f\n", N, TIMES / milliseconds);
        fflush(fptr);

        // Deallocate
        checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
        checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
        checkCublasStatus(cublasLtDestroy(handle));

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