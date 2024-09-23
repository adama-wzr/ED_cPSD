/*

    GPU-based Dilation-Erosion algorithm for phase-size distribution.

    The current approach will identify interfaces in CPU time, and launch
    GPU kernels for scanning the radii around these interface pixels.

    Last Change:
    09/22/2024

    Andre Adam.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
#include <stdbool.h>
#include <omp.h>

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

__global__ void CheckR_1D(char *d_targetArray, int* radius, int *d_Size, long int *d_InterfaceArray)
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
        if (ri < 0 || ri > height) continue;
        for(int rj = myCol - r; rj <= myCol + r; rj++){
            if (rj < 0 || rj > width) continue;

            if(pow(rk-mySlice,2) + pow(rj - myCol,2) + pow(ri - myRow,2) <= pow(r, 2))
            {
                d_targetArray[rk*width*height + ri*width + rj] = 0;
            }
        }
    }

    return;
}

/*--------------------------------------------------------
    
        Functions

 --------------------------------------------------------*/

// Function to return error upon cuda call

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


int main(void){

    /*--------------------------------------------------------
    
            Declare and define useful variables

    --------------------------------------------------------*/

    // OMP options
    int num_threads = 1;
    omp_set_num_threads(num_threads);
    
    // Flags

    bool debugFlag = true;      // Controls print statements

    // Image name is manually entered

    char target_name[50];
    char output_name[50];

    sprintf(target_name, "Last200.csv");
    sprintf(output_name, "Last200_ED.csv");

    // Image size is manually entered

    int width, height, depth;

    int *size = (int *)malloc(sizeof(int)*3);

    width = 200;
    height = 200;
    depth = 200;

    size[0] = height;
    size[1] = width;
    size[2] = depth;

    int max_size = width*height*depth;

    sizeInfo mesh;

    mesh.depth = depth;
    mesh.width = width;
    mesh.height = height;
    mesh.nElements = max_size;

    // P holds the structure. E and D are essential for the operations

    char *P = (char *)malloc(sizeof(char)*max_size);
    char *D = (char *)malloc(sizeof(char)*max_size);
    char *D_test= (char *)malloc(sizeof(char)*max_size);
    char *E = (char *)malloc(sizeof(char)*max_size);

    long int *h_InterfaceArray = (long int *)malloc(sizeof(long int)*max_size);

    // Other variables

    double porosity = 0;
    long int p_sum, d_sum, e_sum;
    p_sum = 1;
    e_sum = 1;
    d_sum = 1;

    /*--------------------------------------------------------
    
    Step 1: Read target structure (3D)

    --------------------------------------------------------*/

    // declare arrays to hold coordinate values for solid phase 

    int *x = (int *)malloc(sizeof(int)*max_size);
    int *y = (int *)malloc(sizeof(int)*max_size);
    int *z = (int *)malloc(sizeof(int)*max_size);

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

    p_sum = (long int)count;

    porosity = 1.0f - (double)count/(double)max_size;

    if (debugFlag) printf("Total Count %ld, Porosity = %1.3f\n", count, porosity);

    /*--------------------------------------------------------
    
    Step 2: Populate arrays

    --------------------------------------------------------*/

    // Build P based on the data we just read

    memset(P, 0, sizeof(char)*max_size);
    memset(h_InterfaceArray, 0, sizeof(long int)*max_size);

    int index = 0;

    for (long int i = 0; i<count; i++){
        index = z[i]*height*width + y[i]*width + x[i];
        P[index] = 1;
    }

    // Free coordinate vectors, close file

    free(x);
    free(y);
    free(z);
    fclose(target_data);

    // Ready to start the ED-PSD

    int radius = 1;

    FILE *OUT;

    OUT = fopen(output_name, "w+");
    fprintf(OUT, "R,P,E,D\n");

    long int interfaceCount = 0;

    /*--------------------------------------------------------
    
            Step 3: Allocate device space

    --------------------------------------------------------*/

    // declare

    char *d_P, *d_E, *d_D;
    long int *d_InterfaceArray;
    int *d_InterfaceNum, *debugNum;
    int *d_R;
    int *d_Size;

    // allocate the device space

    CHECK_CUDA( cudaMalloc((void **) &d_P, max_size*sizeof(char)));
    CHECK_CUDA( cudaMalloc((void **) &d_E, max_size*sizeof(char)));
    CHECK_CUDA( cudaMalloc((void **) &d_D, max_size*sizeof(char)));
    CHECK_CUDA( cudaMalloc((void **) &d_InterfaceArray, max_size*sizeof(long int)));

    CHECK_CUDA( cudaMalloc((void **) &d_InterfaceNum, sizeof(int)));
    CHECK_CUDA( cudaMalloc((void **) &debugNum, sizeof(int)));
    CHECK_CUDA( cudaMalloc((void **) &d_R, sizeof(int)));
    CHECK_CUDA( cudaMalloc((void **) &d_Size, 3*sizeof(int)));

    // Define

    CHECK_CUDA( cudaMemcpy(d_P, P, max_size*sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(d_E, P, max_size*sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(d_D, P, max_size*sizeof(char), cudaMemcpyHostToDevice));

    CHECK_CUDA( cudaMemset(d_InterfaceNum, 0, sizeof(int)));
    CHECK_CUDA( cudaMemset(debugNum, 0, sizeof(int)));
    CHECK_CUDA( cudaMemset(d_InterfaceArray, 0, max_size*sizeof(long int)));
    CHECK_CUDA( cudaMemcpy(d_Size, size, 3*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(d_R, &radius, sizeof(int), cudaMemcpyHostToDevice));

    // CheckR_1D<<<num_blocks, threads_per_block>>>(d_D, d_R, d_Size, debugNum, d_InterfaceArray);

    // DilateInterface<<<num_blocks, threads_per_block>>>(d_InterfaceArray, d_InterfaceNum, d_D, d_R, d_Size, debugNum);

    // debug
    // int h_debugNum = 0;
    // CHECK_CUDA( cudaMemcpy(&h_debugNum, debugNum, sizeof(int), cudaMemcpyDeviceToHost));
    // printf("Debug num = %d\n", h_debugNum);

    // continue

    // CHECK_CUDA( cudaMemcpy(D_test, d_D, max_size * sizeof(char), cudaMemcpyDeviceToHost));

    // e_sum = 0;

    interfaceCount = 0;

    while(e_sum != 0 && radius < MAX_R)
    {
        
        /*--------------------------------------------------------
    
                    Step 4: Dilate Interfaces by R

        --------------------------------------------------------*/

        // Copy P into dilation array

        memcpy(D, P, sizeof(char)*max_size);

        // search entire array

        int row, col, slice;
        bool interfaceFlag;
        long int temp_index;
        long int myCount;
        #pragma omp parallel for schedule(auto) private(myCount)
        for(long int i = 0; i<max_size; i++)
        {
            if (P[i] == 0)
            {
                interfaceFlag = false;                          // false until proven otherwise
                slice = i/(height*width);
                row = (i - slice*height*width)/width;
                col = (i - slice*height*width - row*width);

                // Need to find out if this is an interface

                if(slice != 0)
                {
                    temp_index = i - height*width;
                    if (P[i] != P[temp_index]) interfaceFlag = true;
                }

                if(slice != depth - 1)
                {
                    temp_index = i + height*width;
                    if(P[i] != P[temp_index]) interfaceFlag = true;
                }

                if (row != 0)
                {
                    temp_index = i - width;
                    if( P[i] != P[temp_index]) interfaceFlag = true;
                }

                if (row != height - 1)
                {
                    temp_index = i + width;
                    if( P[i] != P[temp_index]) interfaceFlag = true;
                }

                if(col != 0)
                {
                    temp_index = i - 1;
                    if( P[i] != P[temp_index]) interfaceFlag = true;
                }

                if(col != width - 1)
                {
                    temp_index = i + 1;
                    if( P[i] != P[temp_index]) interfaceFlag = true;
                }

                // Continue if it is not an interface

                if (!interfaceFlag) continue;

                #pragma omp capture
                myCount = interfaceCount;
                #pragma omp atomic write
                h_InterfaceArray[interfaceCount] = i;

                #pragma omp atomic update
                interfaceCount++;

                // it is an interface, so we must scan a radius around it

                for(int rk = slice - radius; rk <= slice + radius; rk++){
                    if(rk < 0 || rk > depth - 1) continue;
                    for(int ri = row - radius; ri <= row + radius; ri++){
                        if(ri < 0 || ri > height - 1) continue;
                        for(int rj = col - radius; rj <= col + radius; rj++){
                            if(rj < 0 || rj > width - 1) continue;

                            if(pow(rk-slice,2) + pow(rj-col,2) + pow(ri - row,2) <= pow(radius, 2))
                            {
                                D[rk*height*width + ri*width + rj] = 0;
                            }
                            
                        }
                    }
                }
            }
        }

        int num_blocks, threads_per_block;

        num_blocks = interfaceCount;
        threads_per_block = 2*radius + 1;

        CHECK_CUDA( cudaMemcpy(d_InterfaceArray, d_InterfaceArray, sizeof(long int)*max_size, cudaMemcpyHostToDevice) );


        CheckR_1D<<<num_blocks, threads_per_block>>>(d_D, d_R, d_Size, d_InterfaceArray);

        printf("Interface count = %ld\n", interfaceCount);

        CHECK_CUDA( cudaMemcpy(D_test, d_D, sizeof(char)*max_size, cudaMemcpyDeviceToHost) );

        // printf("Interface count = %ld\n", interfaceCount);

        for(int i = 0; i < max_size; i++)
        {
            if (D[i] != D_test[i]){
                printf("Error at i = %d\n", i);
                break;
            }
        }

        printf("Comparison complete, D = D_test\n");

        /*--------------------------------------------------------
    
                    Step 5: Erode Interfaces by R

        --------------------------------------------------------*/
        // Copy D into E

        memcpy(E, D, sizeof(char)*max_size);

        // Erosion
        for(long int i = 0; i<max_size; i++)
        {
            if (D[i] == 1)
            {
                interfaceFlag = false;                          // false until proven otherwise
                slice = i/(height*width);
                row = (i - slice*height*width)/width;
                col = (i - slice*height*width - row*width);

                // Need to find out if this is an interface

                if(slice != 0)
                {
                    temp_index = i - height*width;
                    if (D[i] != D[temp_index]) interfaceFlag = true;
                }

                if(slice != depth - 1)
                {
                    temp_index = i + height*width;
                    if(D[i] != D[temp_index]) interfaceFlag = true;
                }

                if (row != 0)
                {
                    temp_index = i - width;
                    if( D[i] != D[temp_index]) interfaceFlag = true;
                }

                if (row != height - 1)
                {
                    temp_index = i + width;
                    if( D[i] != D[temp_index]) interfaceFlag = true;
                }

                if(col != 0)
                {
                    temp_index = i - 1;
                    if( D[i] != D[temp_index]) interfaceFlag = true;
                }

                if(col != width - 1)
                {
                    temp_index = i + 1;
                    if( D[i] != D[temp_index]) interfaceFlag = true;
                }

                // Continue if it is not an interface

                if (!interfaceFlag) continue;

                // it is an interface, so we must scan a radius around it

                for(int rk = slice - radius; rk <= slice + radius; rk++){
                    if(rk < 0 || rk > depth - 1) continue;
                    for(int ri = row - radius; ri <= row + radius; ri++){
                        if(ri < 0 || ri > height - 1) continue;
                        for(int rj = col - radius; rj <= col + radius; rj++){
                            if(rj < 0 || rj > width - 1) continue;

                            if(pow(rk-slice,2) + pow(rj-col,2) + pow(ri - row,2) <= pow(radius, 2)) E[rk*height*width + ri*width + rj] = 1;
                        }
                    }
                }
            }
        }

        p_sum = 0;
        e_sum = 0;
        d_sum = 0;

        for(int i = 0; i<max_size; i++){
            p_sum += P[i];
            d_sum += D[i];
            e_sum += E[i];
        }

        if(debugFlag) printf("R = %d, P = %ld, E = %ld, D = %ld\n", radius, p_sum, e_sum, d_sum);
        fprintf(OUT, "%d,%ld,%ld,%ld\n", radius, p_sum, e_sum, d_sum);
        // radius++;
        radius = 1000;
    }

    fclose(OUT);


    return 0;
}