
// #include <vector>
// #include <iostream>
// #include <complex>
#include <omp.h>

#include "/home/emmanuel/Desktop/SuperLU_MT_3.1/SRC/slu_mt_zdefs.h"

// using namespace std;

//
// Test from page 17 of the user manual
// Of a 5x5 sparse matrix
// //
// void SmallTest()
// {
//     SuperMatrix A, L, U, B;

//     int m = 5;
//     int n = 5;
//     int nnz = 12;

//     int nprocs = 8;

//     double s = 19.0;
//     double u = 21.0;
//     double p = 16.0;
//     double e = 5.0;
//     double r = 18.0;
//     double l = 12.0;

//     //
//     // Store the matrix column by columnd and also indicate which row each is
//     // a_mat = Values in sparse column format
//     // a_sub = which row each value is in
//     // a_x = Nonzero entries prior each column (cumulative)
//     //
//     std::vector<double> a_mat = {s, l, l, u, l, l, u, p, u, e, u, r};
//     std::vector<int> a_sub = {0, 1, 4, 1, 2, 4, 0, 2, 0, 3, 3, 4};
//     std::vector<int> x_a = {0, 3, 6, 8, 10, 12};

//     //
//     // Create matrix A in format for SuperLU
//     //
//     // dCreate_CompCol_Matrix(&A, m, n, nnz, &(a_mat[0]), &(a_sub[0]), &(x_a[0]), SLU_NC, SLU_D, SLU_GE);

//     //
//     // Factorize
//     //
//     // superlu_options_t options;
//     // pdgstrf(options, A, &

//     // Seems like if we destroy explicitly then there's a double free
//     // Probably the destructor is already deallocating
//     // Destroy_CompCol_Matrix(&A);
// }

//
// Based on pclinsolx1.c
//
main(int argc, char *argv[])
{
    SuperMatrix A, AC, L, U, B;
    NCformat *Astore;
    SCPformat *Lstore;
    NCPformat *Ustore;
    superlumt_options_t superlumt_options;
    pxgstrf_shared_t pxgstrf_shared;
    pzgstrf_threadarg_t *pzgstrf_threadarg;
    int_t nprocs;
    fact_t fact;
    trans_t trans;
    yes_no_t refact, usepr;
    double u, drop_tol;
    doublecomplex *a;
    int_t *asub, *xa;
    int_t *perm_c; /* column permutation vector */
    int_t *perm_r; /* row permutations from partial pivoting */
    void *work;
    int_t info, lwork, nrhs, ldx;
    int_t m, n, nnz, permc_spec, panel_size, relax;
    int_t i, firstfact;
    doublecomplex *rhsb, *xact;
    Gstat_t Gstat;
    flops_t flopcnt;
    void parse_command_line();

    /* Default parameters to control factorization. */
    nprocs = 1;
    fact = EQUILIBRATE;
    trans = NOTRANS;
    panel_size = sp_ienv(1);
    relax = sp_ienv(2);
    u = 1.0;
    usepr = NO;
    drop_tol = 0.0;
    work = NULL;
    lwork = 0;
    nrhs = 1;

    /* Get the number of processes from command line. */
    // parse_command_line(argc, argv, &nprocs);
    nprocs = 1;

    // printf("Rows cols" IFMT " " IFMT, m, n);

    /* Read the input matrix stored in Harwell-Boeing format. */
    // zreadhb(&m, &n, &nnz, &a, &asub, &xa);

    m = n = 5;
    nnz = 12;
    if (!(a = doublecomplexMalloc(nnz)))
        SUPERLU_ABORT("Malloc fails for a[].");
    if (!(asub = intMalloc(nnz)))
        SUPERLU_ABORT("Malloc fails for asub[].");
    if (!(xa = intMalloc(n + 1)))
        SUPERLU_ABORT("Malloc fails for xa[].");
    double ss = 19.0;
    double uu = 21.0;
    double pp = 16.0;
    double ee = 5.0;
    double rr = 18.0;
    double ll = 12.0;
    a[0].r = ss;
    a[1].r = ll;
    a[2].r = ll;
    a[3].r = uu;
    a[4].r = ll;
    a[5].r = ll;
    a[6].r = uu;
    a[7].r = pp;
    a[8].r = uu;
    a[9].r = ee;
    a[10].r = uu;
    a[11].r = rr;
    asub[0] = 0;
    asub[1] = 1;
    asub[2] = 4;
    asub[3] = 1;
    asub[4] = 2;
    asub[5] = 4;
    asub[6] = 0;
    asub[7] = 2;
    asub[8] = 0;
    asub[9] = 3;
    asub[10] = 3;
    asub[11] = 4;
    xa[0] = 0;
    xa[1] = 3;
    xa[2] = 6;
    xa[3] = 8;
    xa[4] = 10;
    xa[5] = 12;

    /* Set up the sparse matrix data structure for A. */
    zCreate_CompCol_Matrix(&A, m, n, nnz, a, asub, xa, SLU_NC, SLU_Z, SLU_GE);


    if (!(rhsb = doublecomplexMalloc(m * nrhs)))
        SUPERLU_ABORT("Malloc fails for rhsb[].");
    // zCreate_Dense_Matrix(&B, m, nrhs, rhsb, m, SLU_DN, SLU_Z, SLU_GE);
    // xact = doublecomplexMalloc(n * nrhs);
    // ldx = n;
    // zGenXtrue(n, nrhs, xact, ldx);
    // zFillRHS(trans, nrhs, xact, ldx, &A, &B);

    if (!(perm_r = intMalloc(m)))
        SUPERLU_ABORT("Malloc fails for perm_r[].");
    if (!(perm_c = intMalloc(n)))
        SUPERLU_ABORT("Malloc fails for perm_c[].");
    if (!(superlumt_options.etree = intMalloc(n)))
        SUPERLU_ABORT("Malloc fails for etree[].");
    if (!(superlumt_options.colcnt_h = intMalloc(n)))
        SUPERLU_ABORT("Malloc fails for colcnt_h[].");
    if (!(superlumt_options.part_super_h = intMalloc(n)))
        SUPERLU_ABORT("Malloc fails for colcnt_h[].");



    /********************************
     * THE FIRST TIME FACTORIZATION *
     ********************************/

    /* ------------------------------------------------------------
       Allocate storage and initialize statistics variables. 
       ------------------------------------------------------------*/
    StatAlloc(n, nprocs, panel_size, relax, &Gstat);
    StatInit(n, nprocs, &Gstat);

    /* ------------------------------------------------------------
       Get column permutation vector perm_c[], according to permc_spec:
       permc_spec = 0: natural ordering 
       permc_spec = 1: minimum degree ordering on structure of A'*A
       permc_spec = 2: minimum degree ordering on structure of A'+A
       permc_spec = 3: approximate minimum degree for unsymmetric matrices
       ------------------------------------------------------------*/
    permc_spec = 1;
    get_perm_c(permc_spec, &A, perm_c);



    /* ------------------------------------------------------------
       Initialize the option structure superlumt_options using the
       user-input parameters;
       Apply perm_c to the columns of original A to form AC.
       ------------------------------------------------------------*/
    refact = NO;
    pzgstrf_init(nprocs, fact, trans, refact, panel_size, relax,
                 u, usepr, drop_tol, perm_c, perm_r,
                 work, lwork, &A, &AC, &superlumt_options, &Gstat);

    /* ------------------------------------------------------------
       Compute the LU factorization of A.
       The following routine will create nprocs threads.
       ------------------------------------------------------------*/
    pzgstrf(&superlumt_options, &AC, perm_r, &L, &U, &Gstat, &info);

    flopcnt = 0;
    for (i = 0; i < nprocs; ++i)
        flopcnt += Gstat.procstat[i].fcops;
    Gstat.ops[FACT] = flopcnt;

    /* ------------------------------------------------------------
       Solve the system A*X=B, overwriting B with X.
       ------------------------------------------------------------*/
    zgstrs(trans, &L, &U, perm_r, perm_c, &B, &Gstat, &info);

    printf("\n** Result of sparse LU **\n");
    zinf_norm_error(nrhs, &B, xact); /* Check inf. norm of the error */

    Destroy_CompCol_Permuted(&AC); /* Free extra arrays in AC. */

    /*********************************
     * THE SUBSEQUENT FACTORIZATIONS *
     *********************************/

    /* ------------------------------------------------------------
       Re-initialize statistics variables and options used by the
       factorization routine pzgstrf().
       ------------------------------------------------------------*/
    StatInit(n, nprocs, &Gstat);
    refact = YES;
    pzgstrf_init(nprocs, fact, trans, refact, panel_size, relax,
                 u, usepr, drop_tol, perm_c, perm_r,
                 work, lwork, &A, &AC, &superlumt_options, &Gstat);

    /* ------------------------------------------------------------
       Compute the LU factorization of A.
       The following routine will create nprocs threads.
       ------------------------------------------------------------*/
    pzgstrf(&superlumt_options, &AC, perm_r, &L, &U, &Gstat, &info);

    flopcnt = 0;
    for (i = 0; i < nprocs; ++i)
        flopcnt += Gstat.procstat[i].fcops;
    Gstat.ops[FACT] = flopcnt;

    /* ------------------------------------------------------------
       Re-generate right-hand side B, then solve A*X= B.
       ------------------------------------------------------------*/
    zFillRHS(trans, nrhs, xact, ldx, &A, &B);
    zgstrs(trans, &L, &U, perm_r, perm_c, &B, &Gstat, &info);

    /* ------------------------------------------------------------
       Deallocate storage after factorization.
       ------------------------------------------------------------*/
    pxgstrf_finalize(&superlumt_options, &AC);

    printf("\n** Result of sparse LU **\n");
    zinf_norm_error(nrhs, &B, xact); /* Check inf. norm of the error */

    Lstore = (SCPformat *)L.Store;
    Ustore = (NCPformat *)U.Store;
    printf("No of nonzeros in factor L = " IFMT "\n", Lstore->nnz);
    printf("No of nonzeros in factor U = " IFMT "\n", Ustore->nnz);
    printf("No of nonzeros in L+U = " IFMT "\n", Lstore->nnz + Ustore->nnz - n);
    fflush(stdout);

    SUPERLU_FREE(rhsb);
    SUPERLU_FREE(xact);
    SUPERLU_FREE(perm_r);
    SUPERLU_FREE(perm_c);
    SUPERLU_FREE(superlumt_options.etree);
    SUPERLU_FREE(superlumt_options.colcnt_h);
    SUPERLU_FREE(superlumt_options.part_super_h);
    Destroy_CompCol_Matrix(&A);
    Destroy_SuperMatrix_Store(&B);
    if (lwork == 0)
    {
        Destroy_SuperNode_SCP(&L);
        Destroy_CompCol_NCP(&U);
    }
    else if (lwork > 0)
    {
        SUPERLU_FREE(work);
    }
    StatFree(&Gstat);

    return 0;
}

// int main()
// {
//     // cout << "Hello World!" << endl;

//     // SmallTest();

//     // TestPdrepeat();

//     SmallTest_2();

//     return 0;
// }

// /*
//  * Parse command line to get nprocs, the number of processes.
//  */
// void parse_command_line(int argc, char *argv[], int_t *nprocs)
// {
//     register int c;
//     extern char *optarg;

//     while ((c = getopt(argc, argv, "hp:")) != EOF)
//     {
//         switch (c)
//         {
//         case 'h':
//             printf("Options: (default values are in parenthesis)\n");
//             printf("\t-p <int> - number of processes     ( " IFMT " )\n", *nprocs);
//             exit(1);
//             break;
//         case 'p':
//             *nprocs = atoi(optarg);
//             break;
//         }
//     }
// }
