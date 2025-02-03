/*

    CPU-based Dilation-Erosion algorithm for phase-size distribution.

    This version is just a main file, will call functions from helper files.

    Last Update:
    01/31/2025

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
    
    if(opts.nD == 3)
        Sim3D(&opts);
    
    return 0;
}