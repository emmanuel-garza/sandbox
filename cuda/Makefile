NVCC_FLAGS =-Xcompiler "-Ofast -fopenmp -Wall"

all: main.exe
	export OMP_NUM_THREADS=6; \
	./main.exe

# main.exe: kernel.cu
main.exe: test_class.cu
	nvcc $(NVCC_FLAGS) $^ -o $@

clean:
	rm *.exe
