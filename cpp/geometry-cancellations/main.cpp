#include "parametrization.cpp"

int main()
{

    auto f0 = [](precision u_aux, precision v_aux) {
        return std::sqrt(
            1.0f - u_aux * u_aux - v_aux * v_aux +
            std::pow(u_aux * v_aux, 2) / 3.0f);
    };

    std::cout << (f0(0.30f, 0.00f)) << std::endl;

    return 0;

    // int n = 20;
    // int n_repeat = 10;

    Array2D<precision> u_src(1, 1), v_src(1, 1);
    Array2D<precision> u_trg(1, 1), v_trg(1, 1);

    Array2D<precision_hd> u_src_hd(1, 1), v_src_hd(1, 1);
    Array2D<precision_hd> u_trg_hd(1, 1), v_trg_hd(1, 1);

    int ind_patch = 0;

    //
    // Let's test the function
    //
    u_src[0] = (precision)0.123;
    v_src[0] = (precision)0.656;

    u_trg[0] = u_src[0] + (precision)1.0e-5;
    v_trg[0] = v_src[0];

    //
    // Variables for the parametrization
    //
    std::vector<precision> geometry_param(4);
    bool normal_flag = true;
    bool tangential_flag = false;

    std::array<Array2D<precision>, 3> eu;
    std::array<Array2D<precision>, 3> ev;

    std::array<Array2D<precision_hd>, 3> eu_hd;
    std::array<Array2D<precision_hd>, 3> ev_hd;

    int normal_orientation;

    geometry_param = {0.0, 0.0, 0.0, 1.0};

    //
    //
    //
    std::array<Array2D<precision>, 3> x_src, x_trg;
    std::array<Array2D<precision>, 3> n_src, n_trg;

    std::array<Array2D<precision_hd>, 3> x_src_hd, x_trg_hd;
    std::array<Array2D<precision_hd>, 3> n_src_hd, n_trg_hd;

    for (int k = 0; k < 3; k++)
    {
        x_src[k].resize(1, 1);
        n_src[k].resize(1, 1);

        x_trg[k].resize(1, 1);
        n_trg[k].resize(1, 1);

        x_src_hd[k].resize(1, 1);
        n_src_hd[k].resize(1, 1);

        x_trg_hd[k].resize(1, 1);
        n_trg_hd[k].resize(1, 1);
    }

    {
        SphereTemplate(
            ind_patch, u_src, v_src, geometry_param, normal_flag, tangential_flag,
            x_src, n_src, eu, ev, normal_orientation);

        SphereTemplate(
            ind_patch, u_trg, v_trg, geometry_param, normal_flag, tangential_flag,
            x_trg, n_trg, eu, ev, normal_orientation);

        precision r2 =
            (std::pow(x_trg[0][0] - x_src[0][0], 2) +
             std::pow(x_trg[1][0] - x_src[1][0], 2) +
             std::pow(x_trg[2][0] - x_src[2][0], 2));

        precision beta =
            (n_trg[0][0] * (x_trg[0][0] - x_src[0][0]) +
             n_trg[1][0] * (x_trg[1][0] - x_src[1][0]) +
             n_trg[2][0] * (x_trg[2][0] - x_src[2][0])) /
            r2;

        std::cout << "Beta = " << beta << std::endl;
    }

    //
    // Now we do in higher precision
    //
    u_src_hd[0] = (precision_hd)u_src[0];
    v_src_hd[0] = (precision_hd)v_src[0];

    u_trg_hd[0] = (precision_hd)u_trg[0];
    v_trg_hd[0] = (precision_hd)v_trg[0];

    // u_src_hd[0] = (precision_hd)0.0L;
    // v_src_hd[0] = (precision_hd)0.0L;
    // u_trg_hd[0] = u_src_hd[0] + (precision_hd)1.5e-8L;
    // v_trg_hd[0] = v_src_hd[0]+ (precision_hd)1.5e-8L;

    {
        SphereTemplate(
            ind_patch, u_src_hd, v_src_hd, geometry_param, normal_flag, tangential_flag,
            x_src_hd, n_src_hd, eu_hd, ev_hd, normal_orientation);

        SphereTemplate(
            ind_patch, u_trg_hd, v_trg_hd, geometry_param, normal_flag, tangential_flag,
            x_trg_hd, n_trg_hd, eu_hd, ev_hd, normal_orientation);

        precision_hd r2 =
            (std::pow(x_trg_hd[0][0] - x_src_hd[0][0], 2) +
             std::pow(x_trg_hd[1][0] - x_src_hd[1][0], 2) +
             std::pow(x_trg_hd[2][0] - x_src_hd[2][0], 2));

        precision_hd beta =
            (n_trg_hd[0][0] * (x_trg_hd[0][0] - x_src_hd[0][0]) +
             n_trg_hd[1][0] * (x_trg_hd[1][0] - x_src_hd[1][0]) +
             n_trg_hd[2][0] * (x_trg_hd[2][0] - x_src_hd[2][0])) /
            r2;

        std::cout << "Beta = " << (precision)r2 << std::endl;
        std::cout << "Beta = " << endl
                  << (precision)n_trg_hd[0][0] << endl
                  << (precision)n_trg_hd[1][0] << endl
                  << (precision)n_trg_hd[2][0] << endl
                  << std::endl;

        std::cout << "Beta = " << (precision)beta << std::endl;
    }

    return 0;
}