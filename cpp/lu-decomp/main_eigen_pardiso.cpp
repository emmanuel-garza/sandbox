
#include <vector>
#include <iostream>
#include <omp.h>

#include <Eigen/Sparse>
#include <Eigen/PardisoSupport>

using namespace std;
using namespace Eigen;
typedef Eigen::Triplet<complex<double>> T;
typedef Eigen::SparseMatrix<complex<double>> SpMat;

typedef Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> SpMatRow;
typedef Eigen::PardisoLU<SpMatRow> Solver;

//
// Construct 4x4 block where each of the 16 blocks has the same sparcity pattern
//
void Block4x4Test()
{
    int n = 4;

    vector<double> v = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    vector<int> col_ind = {0, 1, 3, 0, 2, 1, 2, 0, 3};
    vector<int> row_ind = {0, 3, 5, 7, 9}; // 4+1

    const int nnz = v.size();

    typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatRow;

    Eigen::Map<SpMatRow> mat(n, n, nnz, &(row_ind[0]), &(col_ind[0]), &(v[0]));

    // Display the matrix
    cout << endl;
    cout << "Original Matrix:" << endl;
    cout << mat << endl;

    //
    // Create the larger 4x4 sparse matrix using the original sparcity pattern
    //
    vector<double> v_4x4(16 * nnz, 1.0);
    vector<int> col_ind_4x4(16 * nnz);
    vector<int> row_ind_4x4(4 * n + 1);

    //
    // Fill the matrix values
    //
    int c = 0;
    for (int row = 0; row < n; row++)
        for (int k = 0; k < 4; k++)
            for (int i = row_ind[row]; i < row_ind[row + 1]; i++)
                v_4x4[c++] = v[i];

    for (int i = 4 * nnz; i < 16 * nnz; i++)
        v_4x4[i] = v_4x4[i - 4 * nnz];

    //
    // Fill the row indices
    //
    row_ind_4x4[0] = 0;
    for (int i = 1; i <= n; i++)
    {
        int n_per_row = (row_ind[i] - row_ind[i - 1]);
        row_ind_4x4[i] = row_ind_4x4[i - 1] + 4 * n_per_row;
    }

    for (int i = n + 1; i <= 4 * n; i++)
        row_ind_4x4[i] = row_ind_4x4[i - n] + 4 * nnz;

    cout << row_ind_4x4[4 * n] << endl;

    //
    // Fill the column indices
    //
    c = 0;
    for (int row = 0; row < n; row++)
        for (int k = 0; k < 4; k++)
            for (int i = row_ind[row]; i < row_ind[row + 1]; i++)
                col_ind_4x4[c++] = col_ind[i] + k * n;

    for (int i = 4 * nnz; i < 16 * nnz; i++)
        col_ind_4x4[i] = col_ind_4x4[i - 4 * nnz];

    //
    // Construct the Eigen Matrix
    //
    Eigen::Map<SpMatRow> mat_4x4(4 * n,
                                 4 * n,
                                 16 * nnz,
                                 &(row_ind_4x4[0]),
                                 &(col_ind_4x4[0]),
                                 &(v_4x4[0]));

    // Display the matrix
    cout << endl;
    cout << "Block Matrix:" << endl;
    cout << mat_4x4 << endl;

    return;
}

//
// Test to check using Eigen Map
//
void NoTripletTest()
{

    Solver solver;

    int n = 4;
    double t1, t2;

    {
        vector<complex<double>> v = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        vector<int> col_ind = {0, 1, 3, 0, 2, 1, 2, 0, 3};
        vector<int> row_ind = {0, 3, 5, 7, 9}; // 4+1

        const int nnz = v.size();

        Eigen::Map<SpMatRow> mat(n, n, nnz, &(row_ind[0]), &(col_ind[0]), &(v[0]));

        // Display the matrix
        cout << endl;
        cout << "Matrix:" << endl;
        cout << mat << endl;

        // return;

        solver.pardisoParameterArray()[2 - 1] = 0; // Fill-In reduction: 0 = minimum degree

        solver.pardisoParameterArray()[3 - 1] = 8;  // OpenMP
        solver.pardisoParameterArray()[24 - 1] = 1; // OpenMP
        solver.pardisoParameterArray()[25 - 1] = 1; // OpenMP

        // solver.pardisoParameterArray()[28-1] = 1; // OpenMP

        solver.pardisoParameterArray()[11 - 1] = 1; // Improved accuracy seems to be needed
        solver.pardisoParameterArray()[13 - 1] = 2; // Improved accuracy seems to be needed
        // solver.pardisoParameterArray()[3-1] = 10;

        // Analyze pattern
        t1 = omp_get_wtime();
        solver.analyzePattern(mat);
        t2 = omp_get_wtime();
        cout << "Time to analyze pattern = " << t2 - t1 << " sec." << endl;

        // Factorize
        t1 = omp_get_wtime();
        solver.factorize(mat);
        t2 = omp_get_wtime();
        cout << "Time to factorize = " << t2 - t1 << " sec." << endl;
    }
    // Solve
    t1 = omp_get_wtime();
    Eigen::VectorXcd f(n);

    f(0) = 1.0;
    f(1) = 2.0;
    f(2) = 3.0;
    f(3) = 4.0;

    Eigen::VectorXcd u = solver.solve(f);
    t2 = omp_get_wtime();
    cout << "Time to solve = " << t2 - t1 << " sec." << endl;

    cout << u << endl;

    cout << solver.info() << endl;

    return;
}

//
// Test using a banded matrix
//
// void BandedTest(int n, int n_band_left, int n_band_right)
// {
//     SpMat A(n, n); // declare sparse matrix A
//     std::vector<T> triplet_list;

//     double t1 = omp_get_wtime();

//     int c = 0;
//     for (int i = 0; i < n; i++)
//     {
//         for (int j = std::max(0, i - n_band_left); j < std::min(n, i + n_band_right + 1); j++)
//         {
//             // triplet_list.push_back(T(i, j, (double)++c));
//             triplet_list.push_back(T(i, j, 1.1));

//             c++;

//             // if (i > n - 10)
//             //     cout << triplet_list[c - 1].row() + 1 << " " << triplet_list[c - 1].col() + 1 << " " << triplet_list[c - 1].value() << endl;
//         }
//     }
//     double t2 = omp_get_wtime();

//     cout << "Time to set triplet list = " << t2 - t1 << " sec." << endl;

//     //
//     // Set sparse matrix
//     //
//     t1 = omp_get_wtime();

//     A.setFromTriplets(triplet_list.begin(), triplet_list.end());

//     t2 = omp_get_wtime();
//     cout << "Time to set sparse matrix = " << t2 - t1 << " sec." << endl;

//     Solver solver;

//     solver.pardisoParameterArray()[2 - 1] = 0; // Fill-In reduction: 0 = minimum degree

//     solver.pardisoParameterArray()[3 - 1] = 8;  // OpenMP
//     solver.pardisoParameterArray()[24 - 1] = 1; // OpenMP
//     solver.pardisoParameterArray()[25 - 1] = 1; // OpenMP

//     // solver.pardisoParameterArray()[28-1] = 1; // OpenMP

//     solver.pardisoParameterArray()[11 - 1] = 1; // Improved accuracy seems to be needed
//     solver.pardisoParameterArray()[13 - 1] = 2; // Improved accuracy seems to be needed
//     // solver.pardisoParameterArray()[3-1] = 10;

//     // Analyze pattern
//     t1 = omp_get_wtime();
//     solver.analyzePattern(A);
//     t2 = omp_get_wtime();
//     cout << "Time to analyze pattern = " << t2 - t1 << " sec." << endl;

//     // Factorize
//     t1 = omp_get_wtime();
//     solver.factorize(A);
//     t2 = omp_get_wtime();
//     cout << "Time to factorize = " << t2 - t1 << " sec." << endl;

//     // Solve
//     t1 = omp_get_wtime();
//     Eigen::VectorXcd f(n); // RHS
//     f.setConstant(complex<double>(1.0, 1.0));

//     Eigen::VectorXcd u = solver.solve(f);
//     t2 = omp_get_wtime();
//     cout << "Time to solve = " << t2 - t1 << " sec." << endl;

//     cout << solver.info() << endl;

//     for (int i = 0; i < 10; i++)
//         cout << u(i) << endl;

//     // for (int i = 0; i < 10; i++)
//     //     cout << triplet_list[100]. << endl;

//     return;
// }

int main()
{

    // SmallTest();

    // BandedTest(5000, 50, 50);

    NoTripletTest();

    // Block4x4Test();

    return 0;
}