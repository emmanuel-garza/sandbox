
#include <vector>
#include <iostream>
#include <omp.h>
#include <string>

using namespace std;



int main()
{
    vector<int> a = {1, 2};
    vector<int> b = {2, 1};

    cout << (a == b) << endl;

    return 0;
}