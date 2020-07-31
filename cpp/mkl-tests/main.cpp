
#include <vector>
#include <iostream>
#include <omp.h>
#include <cmath>
#include "mkl.h"

using namespace std;

int main()
{
    cout << "Hello World!" << endl;

    int n = 100;
    int n_repeat = 1000;

    vector<double> vec(n), s(n), c(n);

    double t1, t2;

    t1 = omp_get_wtime();
    for (int i = 0; i < n; i++)
        vec[i] = 2.0 * M_PI * i / ((double)(n));

    t2 = omp_get_wtime();
    cout << "Time to initialize: " << (t2 - t1) << " sec" << endl;

    // Direct computation
    t1 = omp_get_wtime();

    for (int j = 0; j < n_repeat; j++)
    {
        for (int i = 0; i < n; i++)
        {
            s[i] = std::sin(vec[i]);
            c[i] = std::cos(vec[i]);
        }
    }

    t2 = omp_get_wtime();
    cout << s[n / 4] << " " << c[n / 4] << endl;
    cout << "Time using STL sin/cos: " << (t2 - t1) << " sec" << endl;

    // Using sincos
    t1 = omp_get_wtime();

    for (int j = 0; j < n_repeat; j++)
    {
        for (int i = 0; i < n; i++)
        {
            // double *sptr, *cptr;
            sincos(vec[i], &(s[i]), &(c[i]));
            // s[i] = sptr;
            // c[i] = *cptr;
        }
    }

    t2 = omp_get_wtime();
    cout << s[n / 4] << " " << c[n / 4] << endl;
    cout << "Time using sincos: " << (t2 - t1) << " sec" << endl;

    // Using MLK
    t1 = omp_get_wtime();

    for (int j = 0; j < n_repeat; j++)
        vdSinCos(n, &(vec[0]), &(s[0]), &(c[0]));

    t2 = omp_get_wtime();
    cout << s[n / 4] << " " << c[n / 4] << endl;
    cout << "Time using MKL: " << (t2 - t1) << " sec" << endl;

    return 0;
}