
# g++ -DUSE_FLOAT=1 main.cpp

a.out: main.cpp
	g++ -Ofast main.cpp -fopenmp
	

run: a.out
	./a.out

clean:
	rm -f *.o *.so *.out