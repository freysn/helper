

NVCC := /usr/local/cuda/bin/nvcc

NVCC_FLAGS := -O3
NVCC_FLAGS += -std=c++11

INCLUDES :=
#NVCC_FLAGS := -arch compute_12
#NVCC_FLAGS :=

OS := $(shell uname)

ifeq ($(OS),Darwin)
NVCC_FLAGS += -gencode arch=compute_12,code=sm_12
INCLUDES += -I/opt/local/include
INCLUDES += -I/Developer/NVIDIA/CUDA-6.5/samples/common/inc
INCLUDES += -I/Users/freysn/dev/reyars
else
NVCC_FLAGS += -gencode arch=compute_52,code=sm_52
INCLUDES += -I/usr/local/cuda-6.5/samples/common/inc
INCLUDES += -I/home/freysn/dev/reyars
endif

NVCC_FLAGS += -DNDEBUG_CUDA

all: cuda
#all: cuda_est_quant

cuda:
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) mlzss.cu -o mlzss_cuda -DMLZSS_WITH_MAIN -L/opt/local/lib -lboost_system-mt -lboost_filesystem-mt

cuda_est_quant:
	$(NVCC) -DMLZSS_EST_QUANT_MAIN $(NVCC_FLAGS) $(INCLUDES) mlzss_est_quant.cu -o mlzss_cuda_est_quant

host:
	g++ -std=c++11 -g mlzss.cpp -o mlzss_cpu

clean:
	rm -rf *~ mlzss_cpu mlzss_cuda *.dSYM 
