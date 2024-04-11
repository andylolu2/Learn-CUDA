DIR=src
BIN=bin
PROFILE=prof

all: saxpy copy launchKernel reverse

.PRECIOUS: $(BIN)/%

$(BIN)/%: $(DIR)/%.cu $(wildcard $(DIR)/lib/*.cu) $(wildcard $(DIR)/lib/*.cuh)
	nvcc --std=c++17 \
	-I=src/cudnn-frontend/include \
	-I=src/cutlass/include,src/cutlass/tools/util/include \
	-l=cudnn,cublas,cublasLt \
	-gencode=arch=compute_75,code=sm_75 \
	$(DIR)/lib/*.cu $< \
	-o $@

%: $(BIN)/%
	# ncu -o $(PROFILE)/$@ -f $(BIN)/$@
	$(BIN)/$@

	# nvcc -l=cublas,cublasLt $(DIR)/lib/*.cu $< -o $@

	# nvcc --std=c++17 \
	# -I=src/cutlass/include,src/cutlass/tools/util/include,src/cudnn-frontend/include \
	# -l=cudnn \
	# -gencode=arch=compute_75,code=sm_75 \
	# $(DIR)/lib/*.cu $< \
	# -o $@