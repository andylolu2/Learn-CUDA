#include <string.h>

#include <iostream>

#include "cuda_runtime.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/tensor_fill.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s N TIMES\n", argv[0]);
        return 0;
    }

    int N = atoi(argv[1]);
    int TIMES = atoi(argv[2]);
    printf("N = %d, TIMES = %d\n", N, TIMES);

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,               // ElementA
        cutlass::layout::ColumnMajor,  // LayoutA
        cutlass::half_t,               // ElementB
        cutlass::layout::ColumnMajor,  // LayoutB
        cutlass::half_t,               // ElementOutput
        cutlass::layout::ColumnMajor,  // LayoutOutput
        cutlass::half_t,               // ElementAccumulator
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm75>;
    Gemm gemm_op;

    // Allocate device memory
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({N, N});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({N, N});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({N, N});
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> D({N, N});

    cutlass::reference::device::TensorFill(A.device_view(), 1.0_hf);
    cutlass::reference::device::TensorFill(B.device_view(), 1.0_hf);
    cutlass::reference::device::TensorFill(C.device_view(), 0.5_hf);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int i = 0; i < TIMES; i++) {
        cutlass::Status status = gemm_op({
            {N, N, N},
            A.device_ref(),   // TensorRef to A device tensor
            B.device_ref(),   // TensorRef to B device tensor
            C.device_ref(),   // TensorRef to C device tensor
            D.device_ref(),   // TensorRef to D device tensor - may be the same as C
            {1.0_hf, 1.0_hf}  // epilogue operation arguments
        });
        // if (status != cutlass::Status::kSuccess) {
        //     std::cerr << "Error running Gemm: " << std::endl;
        //     return -1;
        // }
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);

    double ops = 2 * (double)TIMES * (double)N * (double)N * (double)N / ((double)milliseconds / 1000.0);
    printf("N = %d, %.4f ops/ms, %.4f TFLOPS\n", N, TIMES / milliseconds, ops / 1e12);

    return 0;
}