
#include <vector>
#include <iostream>
#include <omp.h>

#include <Eigen/Sparse>
#include <Eigen/PardisoSupport>

using namespace std;
using namespace Eigen;
typedef Eigen::Triplet<complex<double>> T;
typedef Eigen::SparseMatrix<complex<double>> SpMat;
typedef Eigen::PardisoLU<SpMat> Solver;


//
// Test to check using Eigen "reserve" instead of triplets
//
void NoTripletTest()
{
    int n = 4;
    // SpMat mat(n, n);

    vector<complex<double>> v = {10.0, 20.0, 30.0, 50.0, 40.0, 60.0, 70.0};
    vector<int> row_index = {0, 0, 1, 2, 1, 2, 3};
    vector<int> col_index = {0, 1, 3, 4, 7}; // 4+1

    int nnz = v.size();


    Eigen::Map<SpMat> mat(n, n, nnz, &(col_index[0]), &(row_index[0]), &(v[0]));




    cout << mat << endl;



    Solver solver;

    double t1, t2;

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

    Solver solver;

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
    Eigen::VectorXcd f(n); // RHS
    f.setConstant(complex<double>(1.0, 1.0));

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

    // BandedTest(5000, 50, 50);

    NoTripletTest();

    return 0;
}