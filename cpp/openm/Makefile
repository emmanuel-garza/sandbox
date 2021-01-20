
a.out: main.cpp
	g++  -std=c++14 -O3 -DNDEBUG -fopenmp main.cpp 

run: a.out
	export OMP_NUM_THREADS=8; \
	./a.out

clean:
	rm -f *.o *.so *.out