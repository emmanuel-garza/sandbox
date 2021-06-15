
#include <vector>
#include <iostream>
#include <omp.h>

using namespace std;

int main()
{
    cout << "Hello World!" << endl;

    vector<double> vec;

    // vec = {1.0, 2.0, 3.0};

    std::cout << vec.size() << std::endl;

    vec.push_back(1.0);
    vec.push_back(2.0);

    // vec = {1.0};

    int ind = 0;
    // for (int i = 0; i < vec.size(); i++)
    //     vec[ind++] = i;

    vector<int> a = {1, 2, 4};
    if (std::find(a.begin(), a.end(), 3) == a.end())
        std::cout << "Not Found!" << std::endl;
    else
        std::cout << "Found!" << std::endl;

    // ind = 0;
    // for (int i = 0; i < vec.size(); i++)
    //     cout << ind << ", " << vec[ind++] << endl;

    return 0;
}