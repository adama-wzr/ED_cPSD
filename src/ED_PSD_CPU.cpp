/*

    CPU-based Dilation-Erosion algorithm for phase-size distribution.

    This version is just a main file, will call functions from helper files.

    Last Update:
    09/12/2025

    Andre Adam.
*/

#include "ED_PSD_CPU.hpp"

int main(void)
{

    // struct to hold user input

    options opts;

    // stdout - important call for efficiency on Linux

    fflush(stdout);

    // read user input

    char input[100];
    sprintf(input, "input.txt");

    readInput(input, &opts);

    if (opts.verbose)
        printOpts(&opts);
    
    if (opts.batch)
    {
        if(opts.nD == 2)
            batchSim2D(&opts);
        else
        {
            printf("Method not implemented yet. Batch only available in 2D\n");
            return 1;
        }
    } else
    {
        Sim2D(&opts);
    }
    
    if (opts.nD == 3)
        Sim3D(&opts);

    return 0;
}