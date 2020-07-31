#include "parametrization.cpp"

int main()
{
    int n = 100;
    int n_repeat = 100000;

    {
        double t1 = omp_get_wtime();

        MyArray<double> u(n, n), v(n, n);

        std::array<MyArray<double>, 3> x;
        x[0].resize(n,n);
        x[1].resize(n,n);
        x[2].resize(n,n);

        // Set the uv parameters
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                // u.data[u.index(i, j)] = -1.0 + 2.0 * i / (n - 1);
                // v.data[v.index(i, j)] = -1.0 + 2.0 * j / (n - 1);

                // u.set(i, j, -1.0 + 2.0 * i / (n - 1));
                // v.set(i, j, -1.0 + 2.0 * j / (n - 1));

                u(i, j) = -1.0 + 2.0 * i / (n - 1);
                v(i, j) = -1.0 + 2.0 * j / (n - 1);
            }
        }

        for (int r = 0; r < n_repeat; r++)
        {
            ParametrizationTest5(u, v, x);
            // ParametrizationTest6(u, v, x);
        }

        // x[0](1,4) = 10.0;

        cout << x[0](1,4) << endl;
        double t2 = omp_get_wtime();

        cout << "Time Using MyArray: " << t2 - t1 << " seconds" << endl;

    }

    // return 0;

    // // Using Boost
    // {
    //     double t1 = omp_get_wtime();

    //     typedef array_type::index index;

    //     array_type u(boost::extents[n][n]);
    //     array_type v(boost::extents[n][n]);

    //     array_type x0(boost::extents[n][n]);
    //     array_type x1(boost::extents[n][n]);
    //     array_type x2(boost::extents[n][n]);

    //     for (index i = 0; i != n; ++i)
    //     {
    //         for (index j = 0; j != n; ++j)
    //         {
    //             u[i][j] = -1.0 + 2.0 * i / (n - 1);
    //             v[i][j] = -1.0 + 2.0 * j / (n - 1);
    //         }
    //     }

    //     for (int r = 0; r < n_repeat; r++)
    //     {
    //         ParametrizationTest3(n, u, v, x0, x1, x2);
    //     }

    //     // cout << v[5][2] << endl;

    //     cout << x0[1][4] << endl;
    //     double t2 = omp_get_wtime();

    //     cout << "Time Using Boost: " << t2 - t1 << " seconds" << endl;

    //     // for
    // }

    // Using Eigen
    {
        double t1 = omp_get_wtime();

        ArrayXXd u(n, n), v(n, n);
        array<ArrayXXd, 3> x;

        x[0].resize(n, n);
        x[1].resize(n, n);
        x[2].resize(n, n);

        // Set the uv parameters
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                u(i, j) = -1.0 + 2.0 * i / (n - 1);
                v(i, j) = -1.0 + 2.0 * j / (n - 1);
            }
        }

        for (int r = 0; r < n_repeat; r++)
        {
            ParametrizationTest(u, v, x);
        }

        cout << x[0](1, 4) << endl;

        double t2 = omp_get_wtime();

        cout << "Time Using Eigen: " << t2 - t1 << " seconds" << endl;
    }

    // // Using Eigen but not vector
    // {
    //     double t1 = omp_get_wtime();

    //     ArrayXXd u(n, n), v(n, n), x0(n, n), x1(n, n), x2(n, n);

    //     // Set the uv parameters
    //     for (int i = 0; i < n; i++)
    //     {
    //         for (int j = 0; j < n; j++)
    //         {
    //             u(i, j) = -1.0 + 2.0 * i / (n - 1);
    //             v(i, j) = -1.0 + 2.0 * j / (n - 1);
    //         }
    //     }

    //     for (int r = 0; r < n_repeat; r++)
    //     {
    //         ParametrizationTest0(u, v, x0, x1, x2);
    //     }

    //     cout << x0(1, 4) << endl;

    //     double t2 = omp_get_wtime();

    //     cout << "Time Using Eigen (but not vector): " << t2 - t1 << " seconds" << endl;
    // }

    // Using Malloc
    {
        double t1 = omp_get_wtime();

        int r = n, c = n;

        // dynamically create array of pointers of size r
        double **u = new double *[r];
        double **v = new double *[r];

        double **x0 = new double *[r];
        double **x1 = new double *[r];
        double **x2 = new double *[r];

        // dynamically allocate memory of size c for each row
        for (int i = 0; i < r; i++)
        {
            u[i] = new double[c];
            v[i] = new double[c];

            x0[i] = new double[c];
            x1[i] = new double[c];
            x2[i] = new double[c];
        }

        // assign values to allocated memory
        double **u_bak = u;
        double **v_bak = v;

        double **x0_bak = x0;
        double **x1_bak = x1;
        double **x2_bak = x2;

        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                u[i][j] = -1.0 + 2.0 * i / (n - 1);
                v[i][j] = -1.0 + 2.0 * j / (n - 1);
            }
        }

        for (int r = 0; r < n_repeat; r++)
        {
            ParametrizationTest2(n, u, v, x0, x1, x2);
        }

        //     u = u_bak;
        //     v = v_bak;

        //     x0 = x0_bak;
        //     x1 = x1_bak;
        //     x2 = x2_bak;
        // }

        cout << x0[1][4] << endl;

        // deallocate memory using delete[] operator
        for (int i = 0; i < r; i++)
        {
            delete[] u[i];
            delete[] v[i];

            delete[] x0[i];
            delete[] x1[i];
            delete[] x2[i];
        }

        delete[] u, v, x0, x1, x2;

        double t2 = omp_get_wtime();

        cout << "Time Using Pointers: " << t2 - t1 << " seconds" << endl;
    }

    // Using std::vector<vector<double>>
    {
        double t1 = omp_get_wtime();

        vector<vector<double>>
            u(n, vector<double>(n, 0.0)),
            v(n, vector<double>(n, 0.0)),
            x0(n, vector<double>(n, 0.0)),
            x1(n, vector<double>(n, 0.0)),
            x2(n, vector<double>(n, 0.0));

        // Set the uv parameters
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                u[i][j] = -1.0 + 2.0 * i / (n - 1);
                v[i][j] = -1.0 + 2.0 * j / (n - 1);
            }
        }

        for (int r = 0; r < n_repeat; r++)
        {
            ParametrizationTest4(n, u, v, x0, x1, x2);
        }

        cout << x0[1][4] << endl;

        double t2 = omp_get_wtime();

        cout << "Time Using vector<vector<double>>: " << t2 - t1 << " seconds" << endl;
    }

    // Using vector<double
    {
        double t1 = omp_get_wtime();

        vector<double> u, v;
        array<vector<double>, 3> x;

        u.resize(n * n);
        v.resize(n * n);

        x[0].resize(n * n);
        x[1].resize(n * n);
        x[2].resize(n * n);

        int c = -1;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                c++;
                u[c] = -1.0 + 2.0 * i / (n - 1);
                v[c] = -1.0 + 2.0 * j / (n - 1);
            }
        }

        for (int r = 0; r < n_repeat; r++)
        {
            ParametrizationTest4(n, u, v, x);
        }

        double t2 = omp_get_wtime();

        cout << "Time Using vector<double>: " << t2 - t1 << " seconds" << endl;
    }

    return 0;
}