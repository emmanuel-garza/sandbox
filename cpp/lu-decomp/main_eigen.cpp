#include <vector>
#include <iostream>
#include <omp.h>

#include <Eigen/Dense>

using namespace std;

int main()
{
    Eigen::MatrixXcd A(2, 2);

    A << std::complex<double>(2.0, 1.0), -1, 1, 3;
    cout << "Here is the input matrix A before decomposition:\n"
         << A << endl;

    Eigen::PartialPivLU<Eigen::Ref<Eigen::MatrixXcd>> lu(A);
    cout << "Here is the input matrix A after decomposition:\n"
         << A << endl;

         
    return 0;
}