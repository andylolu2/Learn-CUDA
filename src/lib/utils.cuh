#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <cublas_v2.h>
#include <cudnn.h>
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

#define checkCudnnErr(...) \
    { checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); }
inline void checkCudnnError(cudnnStatus_t code, const char *expr, const char *file, int line) {
    if (code) {
        printf("CUDNN error at %s:%d, code=%d (%s) in '%s'\n", file, line, (int)code, cudnnGetErrorString(code), expr);
        exit(1);
    }
}

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

void print2DArray(float *arr, int rwos, int cols);

#endif
