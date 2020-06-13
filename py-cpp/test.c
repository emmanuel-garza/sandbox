float loop_cube(int n)
{

    double res= 0.0;

    for (int i=0; i<n; i++)
    {
        for (int j=0; j<n; j++)
        {
            for (int k=0; k<n; k++)
            {
                res += 1;
            }
        }
        
    }

    return (float)res;

}