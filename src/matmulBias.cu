// Example 2. Application Using C and cuBLAS: 0-based indexing
//-----------------------------------------------------------
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>

#include "cublasLt.h"
#include "lib/utils.cuh"

int main(void) {
    int M = 16;
    int N = 16;
    int K = 16;

    float* a = (float*)malloc(K * M * sizeof(float));
    float* b = (float*)malloc(K * N * sizeof(float));
    float* c = (float*)malloc(N * M * sizeof(float));
    float* bias = (float*)malloc(N * sizeof(float));

    // initialize matrix a and b
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            a[i * K + j] = static_cast<float>(i * K + j);  // / static_cast<float>(M * K);
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            b[i * K + j] = static_cast<float>(i * K + j);  // / static_cast<float>(N * K);
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            c[i * K + j] = 0;
        }
    }
    for (int i = 0; i < N; i++) {
        bias[i] = static_cast<float>(i) / 10;  // / static_cast<float>(N * K);
    }

    float* A;
    float* B;
    float* C;
    float* Bias;
    checkCudaStatus(cudaMalloc(&A, M * K * sizeof(float)));
    checkCudaStatus(cudaMalloc(&B, N * K * sizeof(float)));
    checkCudaStatus(cudaMalloc(&C, M * N * sizeof(float)));
    checkCudaStatus(cudaMalloc(&Bias, N * sizeof(float)));
    cudaMemcpy(A, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, b, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C, c, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bias, bias, N * sizeof(float), cudaMemcpyHostToDevice);

    // create cublasLt handle
    cublasLtHandle_t handle;
    checkCublasStatus(cublasLtCreate(&handle));

    // create operation desciriptor
    cublasLtMatmulDesc_t operationDesc;
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t transa = CUBLAS_OP_N;
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    cublasOperation_t transb = CUBLAS_OP_N;
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create (empty) preference for heuristics
    cublasLtMatmulPreference_t preference;
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));

    // create heuristic
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult;
    cublasLtMatrixLayout_t Adesc;
    cublasLtMatrixLayout_t Bdesc;
    cublasLtMatrixLayout_t Biasdesc;
    cublasLtMatrixLayout_t Cdesc;
    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, M));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, N, K, N));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Biasdesc, CUDA_R_32F, M, N, 0));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, M));

    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Biasdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    float alpha = 1.0f;
    float beta = 1.0f;
    checkCublasStatus(cublasLtMatmul(
        handle,
        operationDesc,
        &alpha,
        A,
        Adesc,
        B,
        Bdesc,
        &beta,
        Bias,
        Biasdesc,
        C,
        Cdesc,
        &heuristicResult.algo,
        nullptr,
        0,
        0));

    cudaMemcpy(c, C, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "A:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << a[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "B:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << b[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "bias:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << bias[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "C:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setprecision(8) << c[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

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

    return 0;
}