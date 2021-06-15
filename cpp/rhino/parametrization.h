#ifndef PARAMETRIZATION_H
#define PARAMETRIZATION_H

#include "opennurbs.h"
#include <string>
#include <vector>

class RhinoPatch
{
public:
    int patch_id; // global patch ID from 0 to M-1
    double u_min, u_max;
    double v_min, v_max;
    ON_Surface *pSrf; // pointer to the loaded surface

    RhinoPatch(int patch_id_in, ON_Surface *pSrf_in)
    {
        this->patch_id = patch_id_in;
        this->pSrf = pSrf_in;
        this->pSrf->GetDomain(0, &this->u_min, &this->u_max);
        this->pSrf->GetDomain(1, &this->v_min, &this->v_max);

        return;
    }

    void EvPoint(double up,
                 double vp,
                 ON_3dPoint &pos)
    {
        double u_scaled = (this->u_max - this->u_min) / 2.0 * up +
                          (this->u_max + this->u_min) / 2.0; // convert u in (-1,1) to U in (Umin,u_max)
        double v_scaled = (this->v_max - this->v_min) / 2.0 * vp +
                          (this->v_max + this->v_min) / 2.0; // convert v in (-1,1) to V in (Vmin,v_max)
        this->pSrf->EvPoint(u_scaled, v_scaled, pos);

        return;
    }

    void EvNormal()
    {
        return;
    }

    void EvDerivatives()
    {
        return;
    }

    void EvAll(double up,
               double vp,
               ON_3dPoint &pos,
               ON_3dVector &normal,
               ON_3dVector &du,
               ON_3dVector &dv)
    {
        double u_scaled = (this->u_max - this->u_min) / 2.0 * up +
                          (this->u_max + this->u_min) / 2.0; // convert u in (-1,1) to U in (Umin,u_max)
        double v_scaled = (this->v_max - this->v_min) / 2.0 * vp +
                          (this->v_max + this->v_min) / 2.0; // convert v in (-1,1) to V in (Vmin,v_max)

        this->pSrf->EvNormal(u_scaled, v_scaled, pos, du, dv, normal);

        // Change the derivatives since they are wrt up and vp
        du.x *= 2.0 / (this->u_max - this->u_min);
        du.y *= 2.0 / (this->u_max - this->u_min);
        du.z *= 2.0 / (this->u_max - this->u_min);

        dv.x *= 2.0 / (this->v_max - this->v_min);
        dv.y *= 2.0 / (this->v_max - this->v_min);
        dv.z *= 2.0 / (this->v_max - this->v_min);
    }
};

class RhinoParametrization
{
private:
    ON_TextLog dump;
    ONX_Model model;

    std::vector<RhinoPatch> rhino_patches;

public:
    //
    // Constructor
    //
    RhinoParametrization(std::string filename);

    //
    // Evaluate parametrization
    //
    void EvPoint(int ind_patch,
                 double u, double v,
                 double &x,
                 double &y,
                 double &z)
    {
        ON_3dPoint pos;
        this->rhino_patches[ind_patch].EvPoint(u, v, pos);

        x = pos.x;
        y = pos.y;
        z = pos.z;

        return;
    }

    void EvAll(int ind_patch, double u, double v,
               double &x, double &y, double &z,
               double &nx, double &ny, double &nz,
               double &du_x, double &du_y, double &du_z,
               double &dv_x, double &dv_y, double &dv_z)
    {
        ON_3dPoint pos;
        ON_3dVector normal;
        ON_3dVector du, dv;

        this->rhino_patches[ind_patch].EvAll(u, v, pos, normal, du, dv);

        x = pos.x;
        y = pos.y;
        z = pos.z;

        nx = normal.x;
        ny = normal.y;
        nz = normal.z;

        du_x = du.x;
        du_y = du.y;
        du_z = du.z;

        dv_x = dv.x;
        dv_y = dv.y;
        dv_z = dv.z;

        return;
    }
};

//
// Constructor
//
RhinoParametrization::RhinoParametrization(std::string filename)
{

    int ind_patch = 0;
    this->dump.SetIndentSize(2);

    ON_wString ws_arg = ON_FileSystemPath::ExpandUser(filename.c_str());

    const wchar_t *wchar_arg = ws_arg;

    dump.Print("\nOpenNURBS Archive File:  %ls\n", wchar_arg);

    //
    // Open file containing opennurbs archive
    //
    FILE *archive_fp = ON::OpenFile(wchar_arg, L"rb");

    if (!archive_fp)
    {
        dump.Print("  Unable to open file.\n");
        abort();
    }
    dump.PushIndent();

    // create achive object from file pointer
    ON_BinaryFile archive(ON::archive_mode::read3dm, archive_fp);
    // read the contents of the file into "model"
    bool rc = model.Read(archive, &dump);
    // close the file
    ON::CloseFile(archive_fp);

    // print diagnostic
    if (rc)
    {
        dump.Print("-> Successful read\n\n");
        ONX_ModelComponentIterator it(model, ON_ModelComponent::Type::ModelGeometry);
        const ON_ModelComponent *model_component = nullptr;
        for (model_component = it.FirstComponent(); model_component != nullptr; model_component = it.NextComponent())
        {
            const ON_ModelGeometryComponent *model_geometry = ON_ModelGeometryComponent::Cast(model_component);
            if (nullptr != model_geometry)
            {
                const ON_Brep *brep_cnst = (ON_Brep::Cast(model_geometry->Geometry(nullptr)));
                ON_Brep *brep;
                if (nullptr != brep_cnst)
                {
                    brep = new ON_Brep(brep_cnst[0]);
                    brep->FlipReversedSurfaces();
                    // dump.Print("Find Brep object\n");
                    if (brep->IsSurface())
                    {
                        // dump.Print("The Brep object is a surface\n");

                        this->rhino_patches.push_back(RhinoPatch(ind_patch, brep->m_S[0])); // create a new patch
                        // dump.Print("surface %d readed\n", ind_patch);
                        ind_patch++; // number of surface updated
                    }
                    else
                    {
                        for (int i = 0; i < brep->m_F.SizeOf(); i++)
                        {
                            if (brep->FaceIsSurface(i))
                            {
                                // dump.Print("Face %d is a surface\n", i);
                                this->rhino_patches.push_back(RhinoPatch(ind_patch, brep->m_S[brep->m_F[i].m_si])); // create a new patch
                                // dump.Print("surface %d readed\n", ind_patch);
                                ind_patch++; // number of surface updated
                            }
                        }
                    }
                }
            }
        }
    }
    else
        dump.Print("Errors during reading.\n");

    return;
}

#endif