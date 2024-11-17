/*

    CPU-based Dilation-Erosion algorithm for phase-size distribution.

    This version is just a main file, will call functions from helper files.

    Last Update:
    11/17/2024

    Andre Adam.
*/

#include "ED_PSD_CPU.hpp"

int main(void){
    // flags to determine which code to run
    bool input2D;
    bool debugMode, particleFlag, poreFlag;

    input2D = true;
    debugMode = true;
    particleFlag = true;
    poreFlag = false;

    fflush(stdout);

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