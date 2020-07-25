
a.out: main.cpp parametrization.cpp
	export CPATH=$(CONDA_PREFIX)/include:$(CPATH); \
	export LIBRARY_PATH=$(CONDA_PREFIX)/lib:$(LIBRARY_PATH); \
	export LD_LIBRARY_PATH=$(CONDA_PREFIX)/lib:$(LD_LIBRARY_PATH); \
	g++ -Ofast main.cpp -fopenmp
	

run: a.out
	./a.out

clean:
	rm -f *.o *.so *.out