#include <iostream>
#include <omp.h>
#include <cassert>
#include <vector>
#include <array>
#include <math.h>

using namespace std;

using precision = float;
using precision_hd = double;
// long double;



template <class T>
class Array2D
{
private:
    T *ptr = nullptr;

public:
    std::vector<T> data;

    int n_rows;
    int n_cols;
    int n;

    void resize(int rows, int cols)
    {
        this->n_rows = rows;
        this->n_cols = cols;
        this->n = rows * cols;
        this->data.resize(rows * cols);

        this->ptr = &(this->data[0]);
    }

    int rows()
    {
        this->n_rows;
    }

    int cols()
    {
        this->n_cols;
    }

    Array2D()
    {
    }

    Array2D(int rows, int cols)
    {
        this->resize(rows, cols);
    }

    T &operator()(int i, int j)
    {
        return *(&data[0] + (j + i * this->n_cols));
    }

    T &operator[](int ind)
    {
        return *(&data[0] + ind);
    }

    void set(int i, int j, T d)
    {
        data[j + i * this->n_cols] = d;
        return;
    }

    int index(int i, int j)
    {
        return j + i * this->n_cols;
    }

    int size()
    {
        return this->n;
    }
};



//
// Sphere Template
//
template <typename T>
void SphereTemplate(
    int ind_patch,
    Array2D<T> &u,
    Array2D<T> &v,
    std::vector<precision> &geometry_param,
    bool normal_flag,
    bool tangential_flag,
    std::array<Array2D<T>, 3> &x,
    std::array<Array2D<T>, 3> &n,
    std::array<Array2D<T>, 3> &eu,
    std::array<Array2D<T>, 3> &ev,
    int &normal_orientation)
{
    // Parameters of the sphere
    std::array<T, 3> xc; // Center

    xc[0] = (T)geometry_param[0];
    xc[1] = (T)geometry_param[1];
    xc[2] = (T)geometry_param[2];

    T radius = (T)geometry_param[3];

    normal_orientation = 1;



    // We always compute the coordinates
    switch (ind_patch)
    {
    case 0:


        for (unsigned int i = 0; i < u.size(); i++)
        {

            x[2].data[i] = ((T)1.0L) / std::sqrt(((T)1.0L) + u.data[i] * u.data[i] + v.data[i] * v.data[i]);
            x[0].data[i] = u.data[i] * x[2].data[i];
            x[1].data[i] = v.data[i] * x[2].data[i];
        }

        break;


    case 1:

        for (unsigned int i = 0; i < u.size(); i++)
        {
            x[1].data[i] = ((T)1.0) / std::sqrt(((T)1.0) + u.data[i] * u.data[i] + v.data[i] * v.data[i]);
            x[0].data[i] = u.data[i] * x[1].data[i];
            x[2].data[i] = v.data[i] * x[1].data[i];
        }
        break;

    case 2:

        for (unsigned int i = 0; i < u.size(); i++)
        {
            x[0].data[i] = ((T)1.0) / std::sqrt(((T)1.0) + u.data[i] * u.data[i] + v.data[i] * v.data[i]);
            x[1].data[i] = v.data[i] * x[0].data[i];
            x[2].data[i] = u.data[i] * x[0].data[i];
        }
        break;

    case 3:

        for (unsigned int i = 0; i < u.size(); i++)
        {
            x[2].data[i] = -((T)1.0) / std::sqrt(((T)1.0) + u.data[i] * u.data[i] + v.data[i] * v.data[i]);
            x[0].data[i] = u.data[i] * (-x[2].data[i]);
            x[1].data[i] = v.data[i] * (-x[2].data[i]);
        }
        break;

    case 4:

        for (unsigned int i = 0; i < u.size(); i++)
        {
            x[1].data[i] = -((T)1.0) / std::sqrt(((T)1.0) + u.data[i] * u.data[i] + v.data[i] * v.data[i]);
            x[0].data[i] = u.data[i] * (-x[1].data[i]);
            x[2].data[i] = v.data[i] * (-x[1].data[i]);
        }
        break;

    case 5:

        for (unsigned int i = 0; i < u.size(); i++)
        {
            x[0].data[i] = -((T)1.0) / std::sqrt(((T)1.0) + u.data[i] * u.data[i] + v.data[i] * v.data[i]);
            x[1].data[i] = v.data[i] * (-x[0].data[i]);
            x[2].data[i] = u.data[i] * (-x[0].data[i]);
        }
        break;
    }


    // Check if we need the normals
    if (normal_flag)
    {
        n = x;
    }

    (void) eu;
    (void) ev;

    return;

}