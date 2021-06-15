
#include <vector>
#include <iostream>
#include <omp.h>
#include <string>

#include "parametrization.h"

using namespace std;

int main()
{
    // string filename = "/home/emmanuel/Software/opennurbs-7/example_files/V7/v7_rhino_logo_nurbs.3dm";
    string filename = "~/Desktop/sphere.3dm";

    RhinoParametrization rhino_param(filename);

    double x, y, z;
    double nx, ny, nz;
    double du_x, du_y, du_z;
    double dv_x, dv_y, dv_z;

    omp_set_num_threads(6);

    cout << "Max threads = " << omp_get_max_threads() << endl;

    double t1 = omp_get_wtime();

#pragma omp parallel for
    for (int i = 0; i < 1000000; i++)
    {
        // rhino_param.EvPoint(1, 0.0, 0.0, x, y, z);
        rhino_param.EvAll(0, 0.123, 0.123, x, y, z, nx, ny, nz, du_x, du_y, du_z, dv_x, dv_y, dv_z);
    }
    double t2 = omp_get_wtime();

    cout << t2 - t1 << endl;

    double r = sqrt(x * x + y * y + z * z);
    cout << x << ",  " << y << ", " << z << ", Radius = " << r << endl;
    cout << nx << ",  " << ny << ", " << nz << endl;
    cout << du_x << ",  " << du_y << ", " << du_z << endl;
    cout << dv_x << ",  " << dv_y << ", " << dv_z << endl;
    
    return 0;
}