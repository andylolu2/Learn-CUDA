#include "utils.cuh"

void print2DArray(float *arr, int rows, int cols) {
    printf("[\n");
    for (int i = 0; i < rows; i++) {
        printf("[");
        for (int j = 0; j < cols; j++) {
            printf("%7.2f", arr[IDX2C(i, j, rows)]);
        }
        printf("]\n");
    }
    printf("]\n");
}