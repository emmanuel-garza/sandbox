
a.out: main.cpp
	g++ -I/home/emmanuel/Software/opennurbs-7 -Ofast main.cpp -fopenmp -L /home/emmanuel/Software/opennurbs-7 -lopennurbs_public -luuid
	
run: a.out
	./a.out

clean:
	rm -f *.o *.so *.out