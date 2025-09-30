# ====== Config ======
NP ?= 1

NVCC=/usr/local/cuda/bin/nvcc
HPCX_HOME ?= /opt/hpcx

# MPI wrappers
MPICXX ?= /usr/mpi/gcc/openmpi-4.1.7a1/bin/mpicxx
MPIRUN ?= /usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun
CUDA_HOME ?= /usr/local/cuda

# ====== Arch / Flags ======
GENCODE_SM90    := -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90
GENCODE_FLAGS	:= $(GENCODE_SM90)

ifdef DISABLE_CUB
	NVCC_FLAGS = -Xptxas --optimize-float-atomics
else
	NVCC_FLAGS = -DHAVE_CUB
endif
ifdef SKIP_CUDA_AWARENESS_CHECK
	MPICXX_FLAGS = -DSKIP_CUDA_AWARENESS_CHECK
endif

NVCC_FLAGS     += -lineinfo $(GENCODE_FLAGS) -std=c++17
MPICXX_FLAGS   += -DUSE_NVTX -I$(CUDA_HOME)/include -std=c++17
LD_FLAGS        = -L$(CUDA_HOME)/lib64 -lcudart -lcuda -ldl -lm

# ====== Select source/exe at make time ======
# 默认源文件和可执行文件；可在命令行覆盖：make EXE=foo SRC=foo.cpp
SRC ?= jacobi.cpp
EXE ?= $(basename $(SRC))

# 若源文件不存在，给出清晰报错
ifeq ($(wildcard $(SRC)),)
$(error Source file '$(SRC)' not found)
endif

# ====== Targets ======
# 默认目标
all: $(EXE)

# 可执行文件由指定的源 + 通用 kernels.o 链接
$(EXE): $(SRC) jacobi_kernels.o
	$(MPICXX) $(MPICXX_FLAGS) $(SRC) jacobi_kernels.o $(LD_FLAGS) -o $@

# 正确的 jacobi_kernels.o 规则：把 Makefile 设为 order-only 依赖，避免 nvcc 误把它当源码
jacobi_kernels.o: jacobi_kernels.cu | Makefile
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

.PHONY: clean sanitize run profile

clean:
	rm -f jacobi* *_kernels.o jacobi_kernels.o *.nsys-rep jacobi.*.compute-sanitizer.log

# 计算器工具（对当前 EXE 生效）
sanitize: $(EXE)
	$(MPIRUN) -np $(NP) compute-sanitizer --log-file $(EXE).%q{OMPI_COMM_WORLD_RANK}.compute-sanitizer.log ./$(EXE) -niter 10

# 运行（使用你现有的 MPI 参数），对当前 EXE 生效
run: $(EXE)
	$(MPIRUN) --allow-run-as-root -np $(NP) --mca pml ob1 --mca btl self,vader,tcp --mca coll ^hcoll --mca pml_ucx_enable 0 ./$(EXE)

# 性能分析（对当前 EXE）
profile: $(EXE)
	$(MPIRUN) -np $(NP) nsys profile --trace=mpi,cuda,nvtx -o $(EXE).%q{OMPI_COMM_WORLD_RANK} ./$(EXE) -niter 10
