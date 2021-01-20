
#include <vector>
#include <iostream>
#include <omp.h>

using namespace std;


int main()
{
    cout << "Hello World!" << endl;

    vector<double> vec;

    vec = {1.0, 2.0, 3.0};

    // vec = {1.0};
    
    int ind = 0;
    for (int i = 0; i < vec.size(); i++)
        cout << vec[ind++] << endl;

    return 0;
}