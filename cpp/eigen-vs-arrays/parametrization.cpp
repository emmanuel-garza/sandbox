#include <iostream>
#include "/home/emmanuel/miniconda3/envs/solver-cpp/include/eigen3/Eigen/Dense"
#include <omp.h>

#include "/home/emmanuel/miniconda3/envs/solver-fortran/include/boost/multi_array.hpp"
#include <cassert>

using namespace Eigen;
using namespace std;

typedef boost::multi_array<double, 2> array_type;

void ParametrizationTest0(ArrayXXd &u, ArrayXXd &v, ArrayXXd &x0, ArrayXXd &x1, ArrayXXd &x2)
{

    for (int j = 0; j < u.cols(); j++)
    {
        for (int i = 0; i < u.rows(); i++)
        {

            x2(i, j) = 1.0 / sqrt(1.0 + pow(u(i, j), 2) + pow(v(i, j), 2));
            x0(i, j) = u(i, j) * x2(i, j);
            x1(i, j) = v(i, j) * x2(i, j);
        }
    }

    return;
}

void ParametrizationTest(ArrayXXd &u, ArrayXXd &v, array<ArrayXXd, 3> &x)
{
    // x[2] = 1.0 / sqrt(1.0 + pow(u, 2) + pow(v, 2));
    // x[0] = u * x[2];
    // x[1] = v * x[2];

    for (int j = 0; j < u.cols(); j++)
    {
        for (int i = 0; i < u.rows(); i++)
        {

            x[2](i, j) = 1.0 / sqrt(1.0 + pow(u(i, j), 2) + pow(v(i, j), 2));
            x[0](i, j) = u(i, j) * x[2](i, j);
            x[1](i, j) = v(i, j) * x[2](i, j);
        }
    }

    return;
}

void ParametrizationTest2(int n, double **u, double **v, double **x0, double **x1, double **x2)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            x2[i][j] = 1.0 / sqrt(1.0 + pow(u[i][j], 2) + pow(v[i][j], 2));
            x0[i][j] = u[i][j] * x2[i][j];
            x1[i][j] = v[i][j] * x2[i][j];
        }
    }

    return;
}

void ParametrizationTest3(int n, array_type &u, array_type &v, array_type &x0, array_type &x1, array_type &x2)
{
    typedef array_type::index index;
    for (index i = 0; i != n; ++i)
    {
        for (index j = 0; j != n; ++j)
        {
            x2[i][j] = 1.0 / sqrt(1.0 + pow(u[i][j], 2) + pow(v[i][j], 2));
            x0[i][j] = u[i][j] * x2[i][j];
            x1[i][j] = v[i][j] * x2[i][j];
        }
    }

    return;
}

void ParametrizationTest4(int n, vector<vector<double>>  &u, vector<vector<double>>  &v, vector<vector<double>>  &x0, vector<vector<double>>  &x1, vector<vector<double>>  &x2)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            x2[i][j] = 1.0 / sqrt(1.0 + pow(u[i][j], 2) + pow(v[i][j], 2));
            x0[i][j] = u[i][j] * x2[i][j];
            x1[i][j] = v[i][j] * x2[i][j];
        }
    }

    return;
}


void ParametrizationTest4(int n, vector<double>  &u, vector<double>  &v, array<vector<double>,3>  &x)
{
    int c = -1;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            c++;
            x[2][c] = 1.0 / sqrt(1.0 + pow(u[c], 2) + pow(v[c], 2));
            x[0][c] = u[c] * x[2][c];
            x[1][c] = v[c] * x[2][c];
        }
    }

    return;
}
