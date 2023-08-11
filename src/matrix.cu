#include <assert.h>

#include <array>

#define N (1 << 6)

__host__ __global__ size_t idx2D(size_t i, size_t j, size_t nrow, size_t ncol) {
    // Return the element at the i-th row, j-th column, assuming row-major layout.
    assert(0 <= i && i < nrow && 0 <= j && j < ncol);
    return i * ncol + j;
}

__global__ void kernel(float* A, float* B, float* C, size_t n, size_t m, size_t p) {
    /**
     * Precondition:
     * A: size (n, m)
     * B: size (m, p)
     * C: size (n, p)
     * Matrices are row-major.
     */
    size_t x = threadIdx.x;
    size_t y = threadIdx.y;
    size_t cellWidth = blockDim.x;
    size_t cellHeight = blockDim.y;
    size_t cellX = blockIdx.x;
    size_t cellY = blockIdx.y;
    size_t i = cellX * cellWidth + x;
    size_t j = cellY * cellHeight + y;

    // compute C[i, j] as dot(A[i, :], B[:, j])
    float acc = 0;
    for (size_t k = 0; k < m; k++) {
        acc += A[idx2D(i, k, n, m)] * B[idx2D(k, j, m, p)];
    }
    C[i * n + j] = acc;
}

int main(void) {
    std::array<float, N> mat1 = {0};
    std::array<float, N> mat2 = {0};
    return 0;
}
