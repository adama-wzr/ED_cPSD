/*

    CPU-based Dilation-Erosion algorithm for phase-size distribution.

    This version is just a main file, will call functions from helper files.

    Last Update:
    01/30/2025

    Andre Adam.
*/

#include "ED_PSD_CPU.hpp"

int main(void){

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

    if(opts.nD == 2)
        Sim2D(&opts);
    
    return 0;

    // flags to determine which code to run
    bool input2D;
    bool debugMode, particleFlag, poreFlag;

    input2D = false;
    debugMode = true;
    particleFlag = true;
    poreFlag = false;

    

    if (input2D)
    {
        // 2D code
        if (particleFlag)   ParticleSizeDist2D(debugMode);
        if (poreFlag)       PoreSizeDist2D(debugMode);
    } else
    {
        // 3D code
        if (particleFlag)   ParticleSizeDist3D(debugMode);
        if (poreFlag)       PoreSizeDist3D(debugMode);
    }

    return 0;
}