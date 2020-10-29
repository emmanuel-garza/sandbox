#include "/home/emmanuel/Software/gmsh/api/gmsh.h"

#include <iostream>

using namespace std;

int main()
{

    gmsh::initialize();

    // gmsh::open("/home/emmanuel/Desktop/splitter.geo");

    string temp;

    gmsh::model::getCurrent(temp);

    cout << temp << endl;

    gmsh::vectorpair dimTags;
    gmsh::model::getPhysicalGroups(dimTags);

    vector<int> tags;

    vector<double> uv;

    vector<double> coord;

    for (int i = 0; i < dimTags.size(); i++)
    {
        // cout << dimTags[i].first  << endl;
        // gmsh::model::getEntityName(dimTags[i].first, dimTags[i].second, temp);
        gmsh::model::getEntitiesForPhysicalGroup(dimTags[i].first, dimTags[i].second, tags);

        // cout << tags[15] << endl;

        int k = 1;

        vector<double> min, max;

        gmsh::model::getParametrizationBounds(2, tags[k], min, max);

        uv = {max[0], max[1]};

        cout << max.size() << endl;

        cout << max[0] << " " << min[1] << endl;

        gmsh::model::getValue(2, tags[k], uv, coord);

        cout << coord[0] << " " << coord[1] << " " << coord[2] << endl;

        // gmsh::model::getType(2, tags[k], temp);

        // cout << temp << endl;
    }

    // gmsh::model::getValue(0, 26, uv, coord);

    // cout << coord[0] << " " << coord[1] << " " << coord[2] << endl;

    // cout << dimTags.size() << endl;

    gmsh::clear();

    return 0;
}