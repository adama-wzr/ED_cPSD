#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
#include <stdbool.h>
#include <omp.h>
#include <list>

// Load stb for reading jpg's

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

/*--------------------------------------------------------
    
                    Constants

 --------------------------------------------------------*/

#define MAX_R 65

/*--------------------------------------------------------
    
                    Data structures

 --------------------------------------------------------*/

typedef struct
{
    int width;
    int height;
    int depth;
    long int nElements;
} sizeInfo;


/*--------------------------------------------------------
    
        GPU Kernels

 --------------------------------------------------------*/

__global__ void CheckR_D_3D(char *d_targetArray, int* radius, int *d_Size, long int *d_InterfaceArray)
{
    long int myIdx = d_InterfaceArray[blockIdx.x];
    int height, width, depth;
    int myRow, myCol, mySlice;

    height = d_Size[0];
    width = d_Size[1];
    depth = d_Size[2];

    mySlice = myIdx/(height*width);
    myRow = (myIdx - mySlice*height*width)/width;
    myCol = (myIdx - mySlice*height*width - myRow*width);

    int r = radius[0];
    int rk = (threadIdx.x - r) + mySlice;
    if (rk < 0 || rk > depth - 1) return;
    

    for(int ri = myRow - r; ri <= myRow + r; ri++){
        if (ri < 0 || ri > height - 1) continue;
        for(int rj = myCol - r; rj <= myCol + r; rj++){
            if (rj < 0 || rj > width - 1) continue;

            if(pow(rk-mySlice,2) + pow(rj - myCol,2) + pow(ri - myRow,2) <= pow(r, 2))
            {
                d_targetArray[rk*width*height + ri*width + rj] = 0;
            }
        }
    }

    return;
}

__global__ void CheckR_E_3D(char *d_targetArray, int* radius, int *d_Size, long int *d_InterfaceArray)
{
    long int myIdx = d_InterfaceArray[blockIdx.x];

    int height, width, depth;
    int myRow, myCol, mySlice;

    height = d_Size[0];
    width = d_Size[1];
    depth = d_Size[2];

    mySlice = myIdx/(height*width);
    myRow = (myIdx - mySlice*height*width)/width;
    myCol = (myIdx - mySlice*height*width - myRow*width);

    int r = radius[0];
    int rk = (threadIdx.x - r) + mySlice;
    if (rk < 0 || rk > depth - 1) return;

    for(int ri = myRow - r; ri <= myRow + r; ri++){
        if (ri < 0 || ri > height - 1) continue;
        for(int rj = myCol - r; rj <= myCol + r; rj++){
            if (rj < 0 || rj > width - 1) continue;

            if(pow(rk-mySlice,2) + pow(rj - myCol,2) + pow(ri - myRow,2) <= pow(r, 2))
            {
                d_targetArray[rk*width*height + ri*width + rj] = 1;
            }
        }
    }

    return;
}


/*--------------------------------------------------------
    
                    Auxiliary Functions

 --------------------------------------------------------*/

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int readCSV(char*           target_name, 
            char*           P, 
            sizeInfo*       structureInfo,
            bool            debugFlag)
{
    // read data structure

    int height, width;
    height = structureInfo->height;
    width = structureInfo->width;
    long int nElements = structureInfo->nElements;

    // declare arrays to hold coordinate values for solid phase 

    int *x = (int *)malloc(sizeof(int)*nElements);
    int *y = (int *)malloc(sizeof(int)*nElements);
    int *z = (int *)malloc(sizeof(int)*nElements);

    // Read structure file

    FILE *target_data;

    target_data = fopen(target_name, "r");

    // check if file exists

    if (target_data == NULL){
        fprintf(stderr, "Error reading file. Exiting program.\n");
        return 1;
    }

    // Read header

    char header[3];

    fscanf(target_data, "%c,%c,%c", &header[0], &header[1], &header[2]);

    if (debugFlag) printf("Header = %s\n", header);

    // Read coordinates

    size_t count = 0;

    while (fscanf(target_data, "%d,%d,%d", &x[count], &y[count], &z[count]) == 3){
        count++;
    }

    // Build P based on the data we just read

    memset(P, 0, sizeof(char)*nElements);

    int index = 0;

    for (long int i = 0; i<count; i++){
        index = z[i]*height*width + y[i]*width + x[i];
        P[index] = 1;
    }

    // Free coordinate vectors and close file

    free(x);
    free(y);
    free(z);
    fclose(target_data);

    return 0;
}


int saveLabels3D(char*      P,
                 char*      R,
                 int*       L,
                 sizeInfo*  structureInfo,
                 char*      filename)
{
    // read data structure
    int height, width;

    height = structureInfo->height;
    width = structureInfo->width;
    
    long int nElements = structureInfo->nElements;
    
    // Open File
    
    FILE *Particle;

    Particle = fopen(filename, "a+");

    fprintf(Particle, "x,y,z,R,L\n");
    
    int slice,row,col;
    
    // save everything
    
    for(int i = 0; i<nElements;i++)
    {
        if(R[i]!=-1)
        {
            slice = i/(height*width);
            row = (i - slice*height*width)/width;
            col = (i - slice*height*width - row*width);
            fprintf(Particle,"%d,%d,%d,%d,%d\n", col, row, slice, (int) R[i], L[i]);
        } 
    }

    // close file
    
    fclose(Particle);
    
    return 0;
}


long int FindInterface_3D(  char*           mainArray,
                            long int*       InterfaceArray,
                            sizeInfo*       structureInfo,
                            int             primaryPhase,
                            int             numThreads)
{
    // read data structure

    int height, width, depth;
    
    height = structureInfo->height;
    width = structureInfo->width;
    depth = structureInfo->depth;
    long int nElements = structureInfo->nElements;

    long int interfaceCount = 0;

    // set omp number of CPU threads to use

    omp_set_num_threads(numThreads);

    // Loop variables

    int row, col, slice;
    bool interfaceFlag;
    long int temp_index;

    // main loop

    #pragma omp parallel for schedule(auto) private(row, col, slice, interfaceFlag, temp_index)
    for(long int i = 0; i<nElements; i++)
    {
        if(mainArray[i] != primaryPhase) continue;

        interfaceFlag = false;                          // false until proven otherwise

        // Decode index

        slice = i/(height*width);
        row = (i - slice*height*width)/width;
        col = (i - slice*height*width - row*width);

        // Interface search

        if(slice != 0)
        {
            temp_index = i - height*width;
            if (mainArray[i] != mainArray[temp_index]) interfaceFlag = true;
        }

        if(slice != depth - 1)
        {
            temp_index = i + height*width;
            if(mainArray[i] != mainArray[temp_index]) interfaceFlag = true;
        }

        if (row != 0)
        {
            temp_index = i - width;
            if( mainArray[i] != mainArray[temp_index]) interfaceFlag = true;
        }

        if (row != height - 1)
        {
            temp_index = i + width;
            if( mainArray[i] != mainArray[temp_index]) interfaceFlag = true;
        }

        if(col != 0)
        {
            temp_index = i - 1;
            if( mainArray[i] != mainArray[temp_index]) interfaceFlag = true;
        }

        if(col != width - 1)
        {
            temp_index = i + 1;
            if( mainArray[i] != mainArray[temp_index]) interfaceFlag = true;
        }

        // Continue if it is not an interface

        if (!interfaceFlag) continue;

        #pragma omp critical
        {
            InterfaceArray[interfaceCount] = i;
            interfaceCount++;
        }

    }


    return interfaceCount;
}


void ParticleLabel3D(   int             rMin,
                        int             rMax,
                        char*           R,
                        int*            L,
                        sizeInfo*       structureInfo)
{

    // open list

    std::list<long int> oList;

    // read size

    int height, width, depth;
    height = structureInfo->height;
    width = structureInfo->width;
    depth = structureInfo->depth;
    long int nElements = structureInfo->nElements;

    // Loop variables

    int myRow, myCol, mySlice;
    long int temp_index;
    int particleLabel = 0;

    // create iterable radius and begin labelling

    int r = rMin;

    while (r <= rMax)       // main loop
    {
        for(int i = 0; i<nElements; i++)
        {
            if (R[i] != r || L[i] != -1) continue;

            // Label L[i] accordingly and add it to scan list

            L[i] = particleLabel;

            oList.push_back(i);
            while(!oList.empty())       // Flood-Fill search starting from this label alone
            {
                // pop first index on the list and erase it
                long int index = *oList.begin();
                oList.erase(oList.begin());

                // decode index

                mySlice = index/(height*width);
                myRow = (index - mySlice*height*width)/width;
                myCol = (index - mySlice*height*width - myRow*width);

                // Search Neighbords with the same r

                if(mySlice != 0)
                {
                    temp_index = index - height*width;
                    if(L[temp_index] == -1 && R[temp_index] == r)
                    {
                        oList.push_back(temp_index);
                        L[temp_index] = particleLabel;
                    }
                }

                if(mySlice != depth - 1)
                {
                    temp_index = index + height*width;
                    if(L[temp_index] == -1 && R[temp_index] == r)
                    {
                        oList.push_back(temp_index);
                        L[temp_index] = particleLabel;
                    }
                }

                if(myRow != 0)
                {
                    temp_index = index - width;
                    if(L[temp_index] == -1 && R[temp_index] == r)
                    {
                        oList.push_back(temp_index);
                        L[temp_index] = particleLabel;
                    }
                }

                if(myRow != height - 1)
                {
                    temp_index = index + width;
                    if(L[temp_index] == -1 && R[temp_index] == r)
                    {
                        oList.push_back(temp_index);
                        L[temp_index] = particleLabel;
                    }
                }

                if(myCol != 0)
                {
                    temp_index = index - 1;
                    if(L[temp_index] == -1 && R[temp_index] == r)
                    {
                        oList.push_back(temp_index);
                        L[temp_index] = particleLabel;
                    }
                }

                if(myCol != width - 1)
                {
                    temp_index = index + 1;
                    if(L[temp_index] == -1 && R[temp_index] == r)
                    {
                        oList.push_back(temp_index);
                        L[temp_index] = particleLabel;
                    }
                }

            }   // end inner while
            particleLabel++;        // push label increment
        }   // end for
        r++;                        // increase radius label
    } // end while

    return;
}


/*--------------------------------------------------------
    
                    Worker Functions

 --------------------------------------------------------*/

int Hybrid_particleSD_3D(char*      P,
                         char*      E, 
                         char*      D,
                         char*      R, 
                         long int*  InterfaceArray, 
                         int        radius, 
                         int        numThreads, 
                         char*      output_name,
                         sizeInfo*  structureInfo,
                         bool       debugFlag)
{
    // read data structure
    int height, width, depth;
    height = structureInfo->height;
    width = structureInfo->width;
    depth = structureInfo->depth;
    long int nElements = structureInfo->nElements;

    long int p_sum, d_sum, e_sum;
    p_sum = 1;
    e_sum = 1;
    d_sum = 1;
    int primaryPhase = 0;

    int* size = (int *)malloc(sizeof(int)*3);
    size[0] = height;
    size[1] = width;
    size[2] = depth;

    /*-------------------------------------------------------
    
                Declare/Define GPU Variables
    
    -------------------------------------------------------*/

    // declare

    char *d_P, *d_E, *d_D;
    long int *d_InterfaceArray;
    int *d_R;
    int *d_Size;

    // allocate the device space

    CHECK_CUDA( cudaMalloc((void **) &d_P, nElements*sizeof(char)));
    CHECK_CUDA( cudaMalloc((void **) &d_E, nElements*sizeof(char)));
    CHECK_CUDA( cudaMalloc((void **) &d_D, nElements*sizeof(char)));
    CHECK_CUDA( cudaMalloc((void **) &d_InterfaceArray, nElements*sizeof(long int)));

    CHECK_CUDA( cudaMalloc((void **) &d_R, sizeof(int)));
    CHECK_CUDA( cudaMalloc((void **) &d_Size, 3*sizeof(int)));

    // Define

    CHECK_CUDA( cudaMemcpy(d_P, P, nElements*sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(d_E, P, nElements*sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(d_D, P, nElements*sizeof(char), cudaMemcpyHostToDevice));

    CHECK_CUDA( cudaMemset(d_InterfaceArray, 0, nElements*sizeof(long int)));
    CHECK_CUDA( cudaMemcpy(d_Size, size, 3*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(d_R, &radius, sizeof(int), cudaMemcpyHostToDevice));


    /*-------------------------------------------------------
    
                        Main Loop
    
    -------------------------------------------------------*/


    // Open output file

    FILE *OUT;

    OUT = fopen(output_name, "w+");
    fprintf(OUT, "R,P,E,D\n");

    long int interfaceCount = 0;

    while (e_sum != 0 && radius < MAX_R )
    {
        memcpy(D, P, sizeof(char)*nElements);       // copy P into D
        CHECK_CUDA( cudaMemcpy(d_D, d_P, sizeof(char)*nElements, cudaMemcpyDeviceToDevice));

        primaryPhase = 0;

        // Find interfaces:

        interfaceCount = FindInterface_3D(P, InterfaceArray, structureInfo, primaryPhase, numThreads);

        // Calculate number of blocks and threads per block based on radius

        int num_blocks, threads_per_block;

        num_blocks = interfaceCount;
        threads_per_block = 2*radius + 1;

        // Copy interface array into the GPU

        CHECK_CUDA( cudaMemcpy(d_InterfaceArray, InterfaceArray, sizeof(long int)*nElements, cudaMemcpyHostToDevice) );

        // Do Dilation on GPU

        CheckR_D_3D<<<num_blocks, threads_per_block>>>(d_D, d_R, d_Size, d_InterfaceArray);

        // Copy D into E

        CHECK_CUDA( cudaMemcpy(D, d_D, sizeof(char)*nElements, cudaMemcpyDeviceToHost));
        memcpy(E, D, sizeof(char)*nElements);
        CHECK_CUDA( cudaMemcpy(d_E, d_D, sizeof(char)*nElements, cudaMemcpyDeviceToDevice));

        // reset base variables for the new loop

        interfaceCount = 0;
        memset(InterfaceArray, 0, sizeof(long int)*nElements);
        CHECK_CUDA( cudaMemset(d_InterfaceArray, 0, sizeof(long int)*nElements));

        primaryPhase = 1;

        // Find interfaces

        interfaceCount = FindInterface_3D(D, InterfaceArray, structureInfo, primaryPhase, numThreads);

        // GPU Operations

        num_blocks = interfaceCount;

        CHECK_CUDA( cudaMemcpy(d_InterfaceArray, InterfaceArray, sizeof(long int)*nElements, cudaMemcpyHostToDevice) );

        // Perform Erosion

        CheckR_E_3D<<<num_blocks, threads_per_block>>>(d_E, d_R, d_Size, d_InterfaceArray);

        // Copy E back

        CHECK_CUDA( cudaMemcpy(E, d_E, sizeof(char)*nElements, cudaMemcpyDeviceToHost));

        // evaluate sums, print results to output file

        e_sum = 0;
        d_sum = 0;
        p_sum = 0;
        #pragma omp parallel for reduction(+:p_sum, d_sum, e_sum)
        for(int i = 0; i<nElements; i++){
            p_sum += P[i];
            d_sum += D[i];
            e_sum += E[i];
            if(P[i] - E[i] == 1 && R[i] == -1) R[i] = radius;
        }

        // reset base variables for the new loop

        interfaceCount = 0;
        memset(InterfaceArray, 0, sizeof(long int)*nElements);
        CHECK_CUDA( cudaMemset(d_InterfaceArray, 0, sizeof(long int)*nElements));

        // Print to output file

        if(debugFlag) printf("R = %d, P = %ld, E = %ld, D = %ld\n", radius, p_sum, e_sum, d_sum);
        fprintf(OUT, "%d,%ld,%ld,%ld\n", radius, p_sum, e_sum, d_sum);

        // Increment radius

        radius++;
        CHECK_CUDA( cudaMemcpy(d_R, &radius, sizeof(int), cudaMemcpyHostToDevice));
    } // end while

    /*---------------------------------------------------------------------
    
                            Memory Management

    ------------------------------------------------------------------------*/
    // Files

    fclose(OUT);

    // Device

    CHECK_CUDA( cudaFree(d_P));
    CHECK_CUDA( cudaFree(d_E));
    CHECK_CUDA( cudaFree(d_D));
    CHECK_CUDA( cudaFree(d_InterfaceArray));
    CHECK_CUDA( cudaFree(d_R));
    CHECK_CUDA( cudaFree(d_Size));

    // Host

    free(size);

    return radius;
}

/*--------------------------------------------------------
    
                    Parent Functions

 --------------------------------------------------------*/

int ParticleSizeDist2D(bool debugMode)
{
    if (debugMode)
    {
        printf("------------------------------------------------\n\n");
        printf("       Particle Size Distribution 2D\n");
        printf("               (Debug Mode)\n\n");
        printf("------------------------------------------------\n\n");
    }
    return 0;
}

int PoreSizeDist2D(bool debugMode)
{
    if (debugMode)
    {
        printf("------------------------------------------------\n\n");
        printf("         Pore Size Distribution 2D\n");
        printf("             (Debug Mode)\n\n");
        printf("------------------------------------------------\n\n");
    }
    return 0;
}


int ParticleSizeDist3D(bool debugMode)
{
    if (debugMode)
    {
        printf("------------------------------------------------\n\n");
        printf("       Particle Size Distribution 3D\n");
        printf("               (Debug Mode)\n\n");
        printf("------------------------------------------------\n\n");
    }
    /*---------------------------------------------------------------------
    
                            Read Input
                                &
                          Declare Arrays 

        Input mode flags:
        - Flag = 0 means .csv file with x,y,z coordinates of the particles.
        - Flag = 1 means stack of .jpg files    (NOT IMPLEMENTED)
        - Flag = 2 means tiff file.             (NOT IMPLEMENTED)

    ------------------------------------------------------------------------*/

    // User Entered Options:
    int radius_offset = 0;
    int radius = 1;
    const unsigned char inputMode = 0;          // hardcoded since its the only option
    char target_name[100];
    char output_name[100];
    bool saveLabels = true;

    char labelsOutput_name[100];

    sprintf(labelsOutput_name, "Test_labels.csv");

    // Output Options

    int numThreads = 8;                         // Number of CPU threads to be used

    // Structure dimension

    sizeInfo structureInfo;

    structureInfo.height = 300;
    structureInfo.width = 300;
    structureInfo.depth = 300;
    structureInfo.nElements = structureInfo.height  * structureInfo.width
                                                    * structureInfo.depth;
    // File names
    sprintf(target_name, "rec_729_300_int1.csv");
    sprintf(output_name, "Test_ED.csv");

    if (debugMode)
    {
        printf("Input File: %s\n", target_name);
        printf("Height = %d, Width = %d, Depth = %d\n\n", 
                    structureInfo.height, structureInfo.width,
                    structureInfo.depth);
    }

    // Allocate Space for Erosion-Dilation

    char *P = (char *)malloc(sizeof(char)*structureInfo.nElements);
    char *E = (char *)malloc(sizeof(char)*structureInfo.nElements);
    char *D = (char *)malloc(sizeof(char)*structureInfo.nElements);

    // Initialize ED arrays

    memset(P, 0, sizeof(char) * structureInfo.nElements);
    memset(E, 0, sizeof(char) * structureInfo.nElements);
    memset(D, 0, sizeof(char) * structureInfo.nElements);

    // Label Arrays

    char *R = (char *)malloc(sizeof(char)* structureInfo.nElements);        // array for particle radius
    int  *L =  (int *)malloc(sizeof(int) * structureInfo.nElements);        // array for particle labels

    // Interface array

    long int *InterfaceArray = (long int *)malloc(sizeof(long int) * structureInfo.nElements);

    // Initialize label arrays and Interface Array

    memset(R,                0, sizeof(char)    * structureInfo.nElements);
    memset(L,                0, sizeof(int)     * structureInfo.nElements);
    memset(InterfaceArray,   0, sizeof(long int)* structureInfo.nElements);

    // Read and store input

    if (inputMode == 0) readCSV(target_name, P, &structureInfo, debugMode);

    if (debugMode) printf("Structure Read\n");

    // Prepare variables for the main loop

    for(long int i = 0; i<structureInfo.nElements; i++)
    {
        R[i] = -1;
        L[i] = -1;
    }

    if(radius_offset != 0) radius = radius_offset;

    /*---------------------------------------------------------------------
    
                            Main Loop

    ------------------------------------------------------------------------*/

    if (debugMode) printf("Starting Main Loop\n");

    radius = Hybrid_particleSD_3D(P, E, D, R, InterfaceArray, radius,
                                  numThreads, output_name, &structureInfo, debugMode);

    /*---------------------------------------------------------------------
    
                            Save Labels

    ------------------------------------------------------------------------*/

    if(saveLabels)
    {
        // Label Particles

        ParticleLabel3D(3, radius, R, L, &structureInfo);
        saveLabels3D(P, R, L, &structureInfo, labelsOutput_name);
    }

    /*---------------------------------------------------------------------
    
                            Memory Management

    ------------------------------------------------------------------------*/

    free(P);
    free(E);
    free(D);
    free(InterfaceArray);
    free(R);
    free(L);
    return 0;
}

int PoreSizeDist3D(bool debugMode)
{
    if (debugMode)
    {
        printf("------------------------------------------------\n\n");
        printf("         Pore Size Distribution 3D\n");
        printf("             (Debug Mode)\n\n");
        printf("------------------------------------------------\n\n");
    }
    return 0;
}