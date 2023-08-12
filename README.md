## Traps

### THE COMPILER
There are about 1000000 (undocumented) flags that you have to pass to `nvcc` to make things work properly. If something doesn't compile / function properly, check the compiler flags first.

Flags that I missed:
- `-I` / `--include`: To add a directory to find header files. E.g. `-I=src/cutlass/include,src/cutlass/tools/util/include`.
- `-l` / `--library`: To include cuda libraries like `CuBLAS` / `CUTLASS`. E.g. `-l=cublas,cublasLt`.
- `-gencode`: Specify which device you are compiling for. Needed to get things like Tensor Core GEMM working. E.g. `-gencode=arch=compute_75,code=sm_75`.

### Hardward support

RTX 2060 only supports Tensor Core MatMuls with FP16 inputs and FP32 compute dtype. Using either FP16 / FP32 for both inputs and compute dtype will fail.

## Discoveries

### TFLOPS
The `TFLOPS` estimation on canonical sources (e.g. https://www.techpowerup.com/gpu-specs) is not an accurate estimation of the theoretical max FLOPs. It probably takes into account for all types of operations (e.g. FP32,FP16,INT8,...). We can actually get a **much** higher TFLOPS if we are just using the Tensor Cores.

For example, the RTX 2060 MAX-Q is rated to have a max TFLOPS of 9.101 (FP16). However, 
```
$ bin/cutlassExample 8192 1000 tmp.txt
N = 8192, TIMES = 1000, FILENAME = tmp.txt
N = 8192, 0.0366 ops/ms, 40.1975 TFLOPS
```

This number comes from the fact that RTX 2060 has 30 SMs, each with 8 Tensor Cores (240 in total). Each Tensor Core is capable of 64 fused multiply-add (128 FLOPs) each clock cycle. While running the program, the clock speed was observed to be ~1300-1400 MHz. In summary, this gives `240 * 128  * 1350e6 / 1e12 = 41.472` TFLOPs!
