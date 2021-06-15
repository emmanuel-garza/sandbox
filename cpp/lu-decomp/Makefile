
OMP_THREADS=8


INC_PATH=./:$(CONDA_PREFIX)/include:$(CONDA_PREFIX)/include/eigen3
LIB_PATH=./:$(CONDA_PREFIX)/lib


COMPILE_ENV_VARS=export CPATH=$(INC_PATH):$(CPATH); \
	export LIBRARY_PATH=$(LIB_PATH):$(LIBRARY_PATH); \
	export LD_LIBRARY_PATH=$(LIB_PATH):$(LD_LIBRARY_PATH);

EXEC_ENV_VARS=export OMP_NUM_THREADS=$(OMP_THREADS); \
	export LD_LIBRARY_PATH=$(LIB_PATH):$(LD_LIBRARY_PATH); 

# -DEIGEN_USE_MKL_ALL

a.out: main_eigen.cpp
	$(COMPILE_ENV_VARS) \
	g++ -Ofast -DEIGEN_USE_MKL_ALL -DMKL_LP64  main_eigen.cpp -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -fopenmp

# run: a.out
# 	./a.out

# clean:
# 	rm -f *.o *.so *.out




# g++ -DUSE_FLOAT=1 main.cpp

# a.out: main.cpp
# 	gcc -Ofast  main.cpp -L/home/emmanuel/Desktop/SuperLU_MT_3.1/lib -lsuperlu_mt_PTHREAD -lf77blas -lpthread -lm
	

run: a.out
	$(COMPILE_ENV_VARS) \
	$(EXEC_ENV_VARS) \
	./a.out

clean:
	rm -f *.o *.so *.out