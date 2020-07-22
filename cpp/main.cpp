#include "/home/emmanuel/miniconda3/envs/solver-cpp/include/eigen3/Eigen/Dense"

#ifdef USE_FLOAT
typedef float myprecision;
typedef Eigen::ArrayXf myarray;
// float operator "" _mp(double d)
// {
//     return (float) d;

// };
#else
typedef double myprecision;
typedef Eigen::ArrayXd myarray;
// double operator "" _mp(double d)
// {
//     return d;
// };
#endif

#include <vector>
#include <iostream>
#include <omp.h>

using namespace std;

void FejerQuadrature1(myarray &x_k, myarray &w_k)
{

    int n = x_k.size();

    x_k.setConstant(0.0);
    w_k.setConstant(0.0);

    int ind_max;
    if (n % 2 == 0)
    {
        ind_max = n / 2;
    }
    else
    {
        ind_max = (n - 1) / 2;
    }

    for (int k = 0; k < n; k++)
    {
        myprecision theta = M_PI * (2.0 * k + 1.0) / n;

        x_k[k] = cos(theta / 2.0);

        for (int j = 1; j <= ind_max; j++)
        {
            w_k[k] = w_k[k] + cos(theta * j) / (4.0 * pow(j, 2) - 1);
        }

        w_k[k] = (2.0 / n) * (1.0 - 2.0 * w_k[k]);
    }

    return;
}

int main()
{
    cout << "Hello World!" << endl;

    double t1 = omp_get_wtime();

    int n = 50000;

    myarray x_k(n), w_k(n);

    FejerQuadrature1( x_k, w_k );

    cout << w_k.sum() << endl;


    // vector<myprecision> myvector;

    // myvector.assign(n, 0.2);

    // myprecision sum_res = 0.0;

    // for (int i = 0; i < n; i++)
    //     for (int j = 0; j < n; j++)
    //         sum_res += myvector[i];

    // cout << typeid(sum_res).name() << endl;

    double t2 = omp_get_wtime();

    cout << typeid(x_k(0)).name() << " Time = " << (t2 - t1) << " seconds" << endl;

    

    return 0;
}