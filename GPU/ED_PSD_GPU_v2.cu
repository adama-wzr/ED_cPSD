/*

    GPU-based Dilation-Erosion algorithm for phase-size distribution.

    This is the second version of the code. The main objectives of this 
    version are:
    - multi-GPU approach.
    - Individual particle identification.
    - Optimized memory allocation and transfers.
    - Diligency in freeing unused memory.

    Last Update:
    09/25/2024

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
#include <list>

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

__global__ void CheckR_D(char *d_targetArray, int* radius, int *d_Size, long int *d_InterfaceArray)
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

__global__ void CheckR_E(char *d_targetArray, int* radius, int *d_Size, long int *d_InterfaceArray)
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
    
        Functions

 --------------------------------------------------------*/

// Function to check CUDA calls

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void ParticleLabel3D(int rMin, int rMax, char *R, int *L, int *size)
{
    // open list
    std::list<long int> oList;

    // read size
    int height, width, depth;
    size[0] = height;
    size[1] = width;
    size[2] = depth;
    int max_size = height*depth*width;
    int myRow, myCol, mySlice;
    long int temp_index;
    int particleLabel = 0;

    // create iterable radius and begin labelling

    int r = rMin;

    while (r <= rMax)       // main loop
    {
        for(int i = 0; i<max_size; i++)
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


int main(void){

    /*--------------------------------------------------------
    
            Declare and define useful variables

    --------------------------------------------------------*/
    // OMP options
    int num_threads = 16;
    omp_set_num_threads(num_threads);

    // Flags

    bool debugFlag = true;      // Controls print statements

    // Image name is manually entered for now

    char target_name[50];
    char output_name[50];

    sprintf(target_name, "Last200.csv");
    sprintf(output_name, "Last200_ED.csv");

    // Image size is manually entered for now

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

    char *P = (char *)malloc(sizeof(char)* max_size);        // will always store the original image
    char *D = (char *)malloc(sizeof(char)* max_size);        // array for the dilation operation
    char *E = (char *)malloc(sizeof(char)* max_size);        // array for the erosion operation
    char *R = (char *)malloc(sizeof(char)* max_size);        // array for particle identification
    int  *L = (int *) malloc(sizeof(int) * max_size);        // array for particle labels

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

    if (debugFlag) printf("Total Count %zd, Porosity = %1.3f\n", count, porosity);

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

    for(long int i = 0; i<max_size; i++)
    {
        R[i] = -1;
        L[i] = -1;
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
    int *d_R;
    int *d_Size;

    // allocate the device space

    CHECK_CUDA( cudaMalloc((void **) &d_P, max_size*sizeof(char)));
    CHECK_CUDA( cudaMalloc((void **) &d_E, max_size*sizeof(char)));
    CHECK_CUDA( cudaMalloc((void **) &d_D, max_size*sizeof(char)));
    CHECK_CUDA( cudaMalloc((void **) &d_InterfaceArray, max_size*sizeof(long int)));

    CHECK_CUDA( cudaMalloc((void **) &d_R, sizeof(int)));
    CHECK_CUDA( cudaMalloc((void **) &d_Size, 3*sizeof(int)));

    // Define

    CHECK_CUDA( cudaMemcpy(d_P, P, max_size*sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(d_E, P, max_size*sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(d_D, P, max_size*sizeof(char), cudaMemcpyHostToDevice));

    CHECK_CUDA( cudaMemset(d_InterfaceArray, 0, max_size*sizeof(long int)));
    CHECK_CUDA( cudaMemcpy(d_Size, size, 3*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(d_R, &radius, sizeof(int), cudaMemcpyHostToDevice));

    interfaceCount = 0;

    while(e_sum != 0 && radius < MAX_R)
    {
        
        /*--------------------------------------------------------
    
                    Step 4: Dilate Interfaces by R

        --------------------------------------------------------*/

        // Copy P into dilation array

        memcpy(D, P, sizeof(char)*max_size);
        CHECK_CUDA( cudaMemcpy(d_D, d_P, sizeof(char)*max_size, cudaMemcpyDeviceToDevice));

        // search entire array

        int row, col, slice;
        bool interfaceFlag;
        long int temp_index;
        #pragma omp parallel for schedule(auto) private(row, col, slice, interfaceFlag, temp_index)
        for(long int i = 0; i<max_size; i++)
        {
            if (P[i] != 0) continue;
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

            #pragma omp critical
            {
                h_InterfaceArray[interfaceCount] = i;
                interfaceCount++;
            }
        }

        int num_blocks, threads_per_block;

        num_blocks = interfaceCount;
        threads_per_block = 2*radius + 1;

        CHECK_CUDA( cudaMemcpy(d_InterfaceArray, h_InterfaceArray, sizeof(long int)*max_size, cudaMemcpyHostToDevice) );


        CheckR_D<<<num_blocks, threads_per_block>>>(d_D, d_R, d_Size, d_InterfaceArray);

        /*--------------------------------------------------------
    
                    Step 5: Erode Interfaces by R

        --------------------------------------------------------*/
        // Copy D into E

        CHECK_CUDA( cudaMemcpy(D, d_D, sizeof(char)*max_size, cudaMemcpyDeviceToHost));
        memcpy(E, D, sizeof(char)*max_size);
        CHECK_CUDA( cudaMemcpy(d_E, d_D, sizeof(char)*max_size, cudaMemcpyDeviceToDevice));

        // reset base variables for the new loop

        interfaceCount = 0;
        memset(h_InterfaceArray, 0, sizeof(long int)*max_size);
        CHECK_CUDA( cudaMemset(d_InterfaceArray, 0, sizeof(long int)*max_size));

        // Erosion
        #pragma omp parallel for schedule(auto) private(row, col, slice, interfaceFlag, temp_index)
        for(long int i = 0; i<max_size; i++)
        {
            if (D[i] != 1) continue;

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

            #pragma omp critical
            {
                h_InterfaceArray[interfaceCount] = i;
                interfaceCount++;
            }
        }

        // GPU Operations

        num_blocks = interfaceCount;

        CHECK_CUDA( cudaMemcpy(d_InterfaceArray, h_InterfaceArray, sizeof(long int)*max_size, cudaMemcpyHostToDevice) );

        CheckR_E<<<num_blocks, threads_per_block>>>(d_E, d_R, d_Size, d_InterfaceArray);

        CHECK_CUDA( cudaMemcpy(E, d_E, sizeof(char)*max_size, cudaMemcpyDeviceToHost));

        p_sum = 0;
        e_sum = 0;
        d_sum = 0;
        #pragma omp parallel for reduction(+:p_sum, d_sum, e_sum)
        for(int i = 0; i<max_size; i++){
            p_sum += P[i];
            d_sum += D[i];
            e_sum += E[i];
            if(P[i] - E[i] == 1 && R[i] == -1) R[i] = radius;
        }

        // reset base variables for the new loop

        interfaceCount = 0;
        memset(h_InterfaceArray, 0, sizeof(long int)*max_size);
        CHECK_CUDA( cudaMemset(d_InterfaceArray, 0, sizeof(long int)*max_size));

        if(debugFlag) printf("R = %d, P = %ld, E = %ld, D = %ld\n", radius, p_sum, e_sum, d_sum);
        fprintf(OUT, "%d,%ld,%ld,%ld\n", radius, p_sum, e_sum, d_sum);
        radius++;
        CHECK_CUDA( cudaMemcpy(d_R, &radius, sizeof(int), cudaMemcpyHostToDevice));
    }

    fclose(OUT);

    FILE *Particle;

    Particle = fopen("test_particle_Label.csv", "a+");
    fprintf(Particle, "x,y,z,R,L\n");
    int slice,row,col;
    for(int i = 0; i<max_size;i++){
        if(R[i]!=-1)
        {
            slice = i/(height*width);
            row = (i - slice*height*width)/width;
            col = (i - slice*height*width - row*width);
            fprintf(Particle,"%d,%d,%d,%d,%d\n", col, row, slice, (int) R[i], L[i]);
        } 
    }
    fclose(Particle);


    return 0;
}