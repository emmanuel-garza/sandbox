a.out: main.c
	gcc -c main.c 
	gcc main.o -L/home/emmanuel/ownCloud/Programming/github-projects/sandbox/py-cpp/ -ltest

lib: test.c
	gcc -c -Wall -Werror -fPIC test.c
	gcc -shared -o libtest.so test.o

run: a.out
	time ./a.out

clean:
	rm -f *.o *.so *.out