#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <cublas_v2.h>
#include <stdio.h>

#include <stdexcept>

#define checkCudaStatus(result) \
    { cudaAssert((result), __FILE__, __LINE__); }
inline void cudaAssert(cudaError err, const char *file, int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA assert: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) {
            exit(err);
        }
    }
}

#define checkCublasStatus(result) \
    { cublasAssert((result), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t err, const char *file, int line, bool abort = true) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS assert: %d %s %d\n", err, file, line);
        if (abort) {
            exit(err);
        }
    }
}

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

void print2DArray(float *arr, int rwos, int cols);

#endif
