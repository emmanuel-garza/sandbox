
a.out: main.cpp parametrization.cpp
	g++ -Ofast -march=native main.cpp -fopenmp

# -mlong-double-64 
# -m128bit-long-double 

run: a.out
	./a.out

clean:
	rm -f *.o *.so *.out