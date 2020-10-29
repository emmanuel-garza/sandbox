
a.out: main.cpp
	export LIBRARY_PATH=/home/emmanuel/Software/gmsh/build/:$(LIBRARY_PATH); \
	g++ -Ofast -march=native main.cpp -lgmsh -fopenmp 

run: a.out
	export LD_LIBRARY_PATH=/home/emmanuel/Software/gmsh/build/:$(LD_LIBRARY_PATH); \
	./a.out

clean:
	rm -f *.o *.so *.out