#include <cudnn_frontend.h>

#include <array>
#include <iostream>

#include "lib/utils.cuh"

bool allowAll(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

cudnn_frontend::ExecutionPlan
get_plan_from_heuristics(cudnn_frontend::OperationGraph &opGraph, cudnnHandle_t handle) {
    auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                          .setOperationGraph(opGraph)
                          .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                          .build();

    auto &engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

    auto plan_builder = [&]() -> cudnn_frontend::ExecutionPlan {
        for (auto &ecfg : engine_config) {
            try {
                auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                .setHandle(handle)
                                .setEngineConfig(ecfg, opGraph.getTag())
                                .build();
                return plan;
            } catch (cudnn_frontend::cudnnException &e) {
                continue;
            }
        }
        return cudnn_frontend::ExecutionPlanBuilder()
            .setHandle(handle)
            .setEngineConfig(engine_config[0], opGraph.getTag())
            .build();
    };

    return plan_builder();
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s N TIMES\n", argv[0]);
        return 0;
    }

    int N = atoi(argv[1]);
    int TIMES = atoi(argv[2]);
    printf("N = %d, TIMES = %d\n", N, TIMES);

    printf("CUDNN VERSION FROM cudnnGetVersion(): %zu\n", cudnnGetVersion());
    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    int64_t dims[3] = {1, N, N};
    int64_t stride[3] = {N * N, N, 1};

    auto xTensor = cudnn_frontend::TensorBuilder()
                       .setDim(3, dims)
                       .setStride(3, stride)
                       .setId('x')
                       .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                       .setDataType(CUDNN_DATA_HALF)
                       .build();

    auto yTensor = cudnn_frontend::TensorBuilder()
                       .setDim(3, dims)
                       .setStride(3, stride)
                       .setId('y')
                       .setAlignment(16)
                       .setDataType(CUDNN_DATA_HALF)
                       .build();

    auto cTensor = cudnn_frontend::TensorBuilder()
                       .setDim(3, dims)
                       .setStride(3, stride)
                       .setId('c')
                       .setAlignment(16)
                       .setDataType(CUDNN_DATA_HALF)
                       .build();

    std::cout << xTensor.describe() << std::endl;
    std::cout << yTensor.describe() << std::endl;
    std::cout << cTensor.describe() << std::endl;

    auto matmulDesc = cudnn_frontend::MatMulDescBuilder()
                          .setComputeType(CUDNN_DATA_FLOAT)
                          .build();
    std::cout << matmulDesc.describe() << std::endl;

    auto matmulOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                        .setaMatDesc(xTensor)
                        .setbMatDesc(yTensor)
                        .setcMatDesc(cTensor)
                        .setmatmulDesc(matmulDesc)
                        .build();
    std::cout << matmulOp.describe() << std::endl;

    std::array<cudnn_frontend::Operation const *, 1> ops = {&matmulOp};

    auto opGraph = cudnn_frontend::OperationGraphBuilder()
                       .setHandle(handle)
                       .setOperationGraph(ops.size(), ops.data())
                       .build();
    std::cout << opGraph.describe() << std::endl;

    auto plan = get_plan_from_heuristics(opGraph, handle);
    std::shared_ptr<cudnn_frontend::ExecutionPlan> inputProjLayerPlan = std::make_shared<cudnn_frontend::ExecutionPlan>(std::move(plan));

    std::cout << "[INFO] Execution Plan tag for input projection layer: " << inputProjLayerPlan->getTag() << std::endl;

    auto workspace_size = plan.getWorkspaceSize();
    std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

    void *workspace_ptr = nullptr;
    if (workspace_size > 0) {
        checkCudaStatus(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
    }

    size_t xSize = N * N * sizeof(half);
    size_t ySize = N * N * sizeof(half);
    size_t cSize = N * N * sizeof(half);
    printf("xSize = %zu, ySize = %zu, cSize = %zu\n", xSize, ySize, cSize);
    void *x_ptr, *y_ptr, *c_ptr;
    checkCudaStatus(cudaMalloc(&x_ptr, xSize));
    checkCudaStatus(cudaMalloc(&y_ptr, ySize));
    checkCudaStatus(cudaMalloc(&c_ptr, cSize));

    void *data_ptrs[] = {x_ptr, y_ptr, c_ptr};
    int64_t uids[] = {'x', 'y', 'c'};

    auto variantPack = cudnn_frontend::VariantPackBuilder()
                           .setWorkspacePointer(workspace_ptr)
                           .setDataPointers(3, data_ptrs)
                           .setUids(3, uids)
                           .build();
    std::cout << "variantPack " << variantPack.describe() << std::endl;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    for (int i = 0; i < 100; i++) {
        checkCudnnErr(
            cudnnBackendExecute(
                handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
    }

    cudaEventRecord(start);
    for (int i = 0; i < TIMES; i++) {
        checkCudnnErr(
            cudnnBackendExecute(
                handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    double nOps = 2 * (double)TIMES * (double)N * (double)N * (double)N / ((double)milliseconds / 1000.0);
    printf("N = %d, %.4f ops/ms, %.4f TFLOPS\n", N, TIMES / milliseconds, nOps / 1e12);

    checkCudaStatus(cudaFree(x_ptr));
    checkCudaStatus(cudaFree(y_ptr));
    checkCudaStatus(cudaFree(c_ptr));
    checkCudaStatus(cudaFree(workspace_ptr));
    checkCudnnErr(cudnnDestroy(handle));
    return 0;
}
