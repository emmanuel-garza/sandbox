
#include <vector>
#include <iostream>
#include <omp.h>

#include <Eigen/Sparse>
// #include <unsupported/Eigen/SparseExtra>
#include <Eigen/PardisoSupport>

using namespace std;
using namespace Eigen;
typedef Eigen::Triplet<complex<double>> T;
typedef Eigen::SparseMatrix<complex<double>> SpMat;
typedef Eigen::PardisoLU<SpMat> Solver;

// void SmallTest()
// {
//     int m = 5;     // number of rows;
//     int n = 5;     // number of cols;
//     SpMat A(m, n); // declare sparse matrix A

//     std::vector<T> triplet_list(12);

//     double ss = 19.0;
//     double uu = 21.0;
//     double pp = 16.0;
//     double ee = 5.0;
//     double rr = 18.0;
//     double ll = 12.0;

//     triplet_list[0] = T(0, 0, ss);
//     triplet_list[1] = T(1, 0, ll);
//     triplet_list[2] = T(4, 0, ll);

//     triplet_list[3] = T(1, 1, uu);
//     triplet_list[4] = T(2, 1, ll);
//     triplet_list[5] = T(4, 1, ll);

//     triplet_list[6] = T(0, 2, uu);
//     triplet_list[7] = T(2, 2, pp);

//     triplet_list[8] = T(0, 3, uu);
//     triplet_list[9] = T(3, 3, ee);

//     triplet_list[10] = T(3, 4, uu);
//     triplet_list[11] = T(4, 4, rr);

//     //
//     // make tripletlist
//     //
//     A.setFromTriplets(triplet_list.begin(), triplet_list.end());

//     //
//     // Right-hand side
//     //
//     Eigen::VectorXd f(m);

//     f(0) = 1.0;
//     f(1) = 1.0;
//     f(2) = 1.0;
//     f(3) = 1.0;
//     f(4) = 1.0;

//     Solver solver;
//     solver.analyzePattern(A);
//     solver.factorize(A);

//     Eigen::VectorXd u = solver.solve(f);

//     for (int i = 0; i < m; i++)
//         cout << u(i) << endl;

//     return;
// }

//
// Test using a banded matrix
//
void BandedTest(int n, int n_band_left, int n_band_right)
{
    SpMat A(n, n); // declare sparse matrix A
    std::vector<T> triplet_list;

    double t1 = omp_get_wtime();

    int c = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = std::max(0, i - n_band_left); j < std::min(n, i + n_band_right + 1); j++)
        {
            // triplet_list.push_back(T(i, j, (double)++c));
            triplet_list.push_back(T(i, j, 1.1));

            c++;

            // if (i > n - 10)
            //     cout << triplet_list[c - 1].row() + 1 << " " << triplet_list[c - 1].col() + 1 << " " << triplet_list[c - 1].value() << endl;
        }
    }
    double t2 = omp_get_wtime();

    cout << "Time to set triplet list = " << t2 - t1 << " sec." << endl;

    //
    // Set sparse matrix
    //
    t1 = omp_get_wtime();

    A.setFromTriplets(triplet_list.begin(), triplet_list.end());

    t2 = omp_get_wtime();
    cout << "Time to set sparse matrix = " << t2 - t1 << " sec." << endl;

    //
    // Right-hand side
    //
    Eigen::VectorXcd f(n);

    f.setConstant(complex<double> (1.0,1.0));

    Solver solver;


    solver.pardisoParameterArray()[2-1] = 0; // Fill-In reduction: 0 = minimum degree

    solver.pardisoParameterArray()[3-1] = 8; // OpenMP
    solver.pardisoParameterArray()[24-1] = 1; // OpenMP
    solver.pardisoParameterArray()[25-1] = 1; // OpenMP


    solver.pardisoParameterArray()[11-1] = 1; // Improved accuracy seems to be needed
    solver.pardisoParameterArray()[13-1] = 2; // Improved accuracy seems to be needed
    // solver.pardisoParameterArray()[3-1] = 10;


    // Analyze pattern
    t1 = omp_get_wtime();
    solver.analyzePattern(A);
    t2 = omp_get_wtime();
    cout << "Time to analyze pattern = " << t2 - t1 << " sec." << endl;


    // Factorize
    t1 = omp_get_wtime();
    solver.factorize(A);
    t2 = omp_get_wtime();
    cout << "Time to factorize = " << t2 - t1 << " sec." << endl;

    // Solve
    t1 = omp_get_wtime();
    Eigen::VectorXcd u = solver.solve(f);
    t2 = omp_get_wtime();
    cout << "Time to solve = " << t2 - t1 << " sec." << endl;

    cout << solver.info() << endl;

    for (int i = 0; i < 10; i++)
        cout << u(i) << endl;

    // for (int i = 0; i < 10; i++)
    //     cout << triplet_list[100]. << endl;

    return;
}

int main()
{

    // SmallTest();

    BandedTest(100000, 20, 20);

    return 0;
}