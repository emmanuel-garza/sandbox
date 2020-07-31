
# g++ -DUSE_FLOAT=1 main.cpp

a.out: main.cpp
	g++ -Ofast main.cpp -fopenmp -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
	

run: a.out
	./a.out

clean:
	rm -f *.o *.so *.out