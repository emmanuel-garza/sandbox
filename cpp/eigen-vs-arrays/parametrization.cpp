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

    // for (int i = 0; i < u.rows(); i++)
    // {
    //     for (int j = 0; j < u.cols(); j++)
    //     {
    //         x[2](i, j) = 1.0 / sqrt(1.0 + pow(u(i, j), 2) + pow(v(i, j), 2));
    //         x[0](i, j) = u(i, j) * x[2](i, j);
    //         x[1](i, j) = v(i, j) * x[2](i, j);
    //     }
    // }

    int c = -1;

    double *ptr0 = x[2].data();

    for (int j = 0; j < u.cols(); j++)
    {
        for (int i = 0; i < u.rows(); i++)
        {

            c++;
            x[2](c) = 1.0 / sqrt(1.0 + pow(u(c), 2) + pow(v(c), 2));
            x[0](c) = u(c) * x[2](c);
            x[1](c) = v(c) * x[2](c);
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

void ParametrizationTest4(int n, vector<vector<double>> &u, vector<vector<double>> &v, vector<vector<double>> &x0, vector<vector<double>> &x1, vector<vector<double>> &x2)
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

void ParametrizationTest4(int n, vector<double> &u, vector<double> &v, array<vector<double>, 3> &x)
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

template <class T>
class MyArray
{
private:
    T *ptr = nullptr;

public:
    std::vector<T> data;

    int rows;
    int cols;

    void resize(int rows, int cols)
    {
        this->rows = rows;
        this->cols = cols;
        this->data.resize(rows * cols);

        this->ptr = &(this->data[0]);
    }

    int rows()
    {
        this->rows;
    }

    int cols()
    {
        this->cols();
    }


    MyArray()
    {
    }

    MyArray(int rows, int cols)
    {
        resize(rows, cols);
    }

    // T operator()(int i, int j)
    // {
    //     return data[j + i * this->cols];
    // }

    T &operator()(int i, int j)
    {
        return *(&data[0] + (j + i * this->cols));
        // return *ptr;
        // return *(ptr + (j + i * this->cols));
    }

    // T &operator[](int ind)
    // {
    //     return
    // }

    void set(int i, int j, T d)
    {
        data[j + i * this->cols] = d;
        return;
    }

    int index(int i, int j)
    {
        return j + i * this->cols;
    }
};

// // Overloading the product
// MyArray &operator*(MyArray)
// {
// }

void ParametrizationTest5(MyArray<double> &u, MyArray<double> &v, array<MyArray<double>, 3> &x)
{

    for (int i = 0; i < u.rows; ++i)
    {
        for (int j = 0; j < u.cols; ++j)
        {
            // x[2].set(i, j, 1.0 / sqrt(1.0 + pow(u(i, j), 2) + pow(v(i, j), 2)));
            // x[0].set(i, j, u(i, j) * x[2](i, j));
            // x[1].set(i, j, v(i, j) * x[2](i, j));

            x[2](i, j) = 1.0 / sqrt(1.0 + pow(u(i, j), 2) + pow(v(i, j), 2));
            x[0](i, j) = u(i, j) * x[2](i, j);
            x[1](i, j) = v(i, j) * x[2](i, j);
        }
    }

    return;
}

void ParametrizationTest6(MyArray<double> &u, MyArray<double> &v, array<MyArray<double>, 3> &x)
{
    int cols = u.cols;
    int rows = u.rows;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            // x[2].data[j + i * cols] = 1.0 / sqrt(1.0 + pow(u.data[j + i * cols], 2) + pow(v.data[j + i * cols], 2));
            // x[0].data[j + i * cols] = u.data[j + i * cols] * x[2].data[j + i * cols];
            // x[1].data[j + i * cols] = v.data[j + i * cols] * x[2].data[j + i * cols];

            x[2].data[u.index(i, j)] = 1.0 / sqrt(1.0 + pow(u.data[u.index(i, j)], 2) + pow(v.data[u.index(i, j)], 2));
            x[0].data[u.index(i, j)] = u.data[u.index(i, j)] * x[2].data[u.index(i, j)];
            x[1].data[u.index(i, j)] = v.data[u.index(i, j)] * x[2].data[u.index(i, j)];
        }
    }

    return;
}