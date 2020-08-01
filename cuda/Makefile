NVCC_FLAGS = -g -G -Xcompiler "-fopenmp -Wall"

all: main.exe
	export OMP_NUM_THREADS=6; \
	./main.exe

main.exe: kernel.cu
	nvcc $(NVCC_FLAGS) $^ -o $@
