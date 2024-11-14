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

typedef struct
{
    int width;
    int height;
    long int nElements;
} sizeInfo2D;

/*--------------------------------------------------------
    
                        GPU Kernels

 --------------------------------------------------------*/

 __global__ void CheckR_D_2D(char *d_targetArray, int* radius, int *d_Size, long int *d_InterfaceArray)
{
    long int myIdx = d_InterfaceArray[blockIdx.x];
    int height, width;
    int myRow, myCol;

    height = d_Size[0];
    width = d_Size[1];

    myRow = myIdx/width;
    myCol = myIdx - myRow*width;

    int r = radius[0];
    int ri = (threadIdx.x - r) + myRow;
    if (ri < 0 || ri > height - 1) return;

    for(int rj = myCol - r; rj <= myCol + r; rj++)
    {
        if (rj < 0 || rj > width - 1) continue;

        if(pow(rj - myCol,2) + pow(ri - myRow,2) <= pow(r, 2))
        {
            d_targetArray[ri*width + rj] = 0;
        }

    }

    return;
}


 __global__ void CheckR_E_2D(char *d_targetArray, int* radius, int *d_Size, long int *d_InterfaceArray)
{
    long int myIdx = d_InterfaceArray[blockIdx.x];
    int height, width;
    int myRow, myCol;

    height = d_Size[0];
    width = d_Size[1];

    myRow = myIdx/width;
    myCol = myIdx - myRow*width;

    int r = radius[0];
    int ri = (threadIdx.x - r) + myRow;
    if (ri < 0 || ri > height - 1) return;

    for(int rj = myCol - r; rj <= myCol + r; rj++)
    {
        if (rj < 0 || rj > width - 1) continue;

        if(pow(rj - myCol,2) + pow(ri - myRow,2) <= pow(r, 2))
        {
            d_targetArray[ri*width + rj] = 1;
        }

    }

    return;
}


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


int readImg_2D( char*           target_name,
                unsigned char** targetPtr,
                sizeInfo2D*     imgInfo)
{
    int channel;

    *targetPtr = stbi_load(target_name, &imgInfo->width, &imgInfo->height, &channel, 1);

    if (channel != 1) return 1;

    imgInfo->nElements = imgInfo->height*imgInfo->width;

    return 0;
}


int readStack   (char           *P,
                sizeInfo*       structureInfo,
                bool            debugMode)
{
    long int index;

    char imgName[100];

    sizeInfo2D tempSize;

    unsigned char* target_img;

    for(int sliceNum = 0; sliceNum < structureInfo->depth; sliceNum++)
    {
        // get image name
        sprintf(imgName, "%05d.jpg", sliceNum);
        
        // read image

        if(readImg_2D(imgName, &target_img, &tempSize) == 1)
        {
            // print error if wrong number of channels and debugMode is on
            if (debugMode) printf("Wrong number of channels on image %d\n", sliceNum);
            return 1;
        }

        // Check that the image is the appropriate size
        // return error statements if the debugFlag is on.
        
        if (structureInfo->width != tempSize.width)
        {
            if(debugMode)
            {
                printf("Dimension mismatch on slice %d\n", sliceNum);
                printf("Expected Width = %d\n", structureInfo->width);
                printf("Actual Width   = %d\n", tempSize.width);
            }
            return 1;
        } else if(structureInfo->height != tempSize.height)
        {
            if(debugMode)
            {
                printf("Dimension mismatch on slice %d\n", sliceNum);
                printf("Expected Height = %d\n", structureInfo->height);
                printf("Actual Height   = %d\n", tempSize.height);
            }
            return 1;
        }

        // use the information we just read to populate the 3d structure

        for(int rowNum = 0; rowNum < tempSize.height; rowNum++)
        {
            for(int colNum = 0; colNum < tempSize.width; colNum++)
            {
                index = sliceNum*tempSize.width*tempSize.height +        \
                        rowNum*tempSize.width + colNum;
                // thresholding at 150
                if (target_img[rowNum*tempSize.width + colNum] < 150) P[index] = 0;
                else P[index] = 1;
            }
        }
    }

    // Free memory

    free(target_img);
    return 0;
}


int saveLabels2D(char*          P,
                 char*          R,
                 int*           L,
                 sizeInfo2D*    structureInfo,
                 char*          filename)
{
    // read data structure
    int width;

    width = structureInfo->width;
    
    long int nElements = structureInfo->nElements;
    
    // Open File
    
    FILE *Particle;

    Particle = fopen(filename, "a+");

    fprintf(Particle, "x,y,R,L\n");
    
    int row,col;
    
    // save everything
    
    for(int i = 0; i<nElements;i++)
    {
        if(R[i]!=-1)
        {
            row = i/width;
            col = i - row*width;
            fprintf(Particle,"%d,%d,%d,%d\n", col, row, (int) R[i], L[i]);
        } 
    }

    // close file
    
    fclose(Particle);
    
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


long int FindInterface_2D(  char*           mainArray,
                            long int*       InterfaceArray,
                            sizeInfo2D*     structureInfo,
                            int             primaryPhase,
                            int             numThreads)
{
    // read data structure

    int height, width;
    
    height = structureInfo->height;
    width = structureInfo->width;
    long int nElements = structureInfo->nElements;

    long int interfaceCount = 0;

    // set omp number of CPU threads to use

    omp_set_num_threads(numThreads);

    // Loop variables

    int row, col;
    bool interfaceFlag;
    long int temp_index;

    // main loop

    #pragma omp parallel for schedule(auto) private(row, col, interfaceFlag, temp_index)
    for(long int i = 0; i<nElements; i++)
    {
        if(mainArray[i] != primaryPhase) continue;

        interfaceFlag = false;                          // false until proven otherwise

        // Decode index
        row = i/width;
        col = (i - row*width);

        // Interface search

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


int f_EDT_2D(   int*        g,
                int         g_index,
                int         x,
                int         i)
{
    int f = (x - i)*(x - i) + g[g_index]*g[g_index];
    return f;
}

int f_EDT_3D(   int*        EDT2D,
                int         EDT2D_index,
                int         x,
                int         i)
{
    int f = (x - i)*(x - i) + EDT2D[EDT2D_index];
    return f;
}

int Sep_EDT_2D( int*        g,
                int         i,
                int         u,
                int         g_index_i,
                int         g_index_u)
{
    int Sep = trunc(u*u - i*i + g[g_index_u]*g[g_index_u] - g[g_index_i]*g[g_index_i])/(2*(u-i));
    return Sep;
}

int Sep_EDT_3D( int*        EDT2D,
                int         i,
                int         u,
                int         EDT2D_index_i,
                int         EDT2D_index_u)
{
    int Sep = trunc(u*u - i*i + EDT2D[EDT2D_index_u] - EDT2D[EDT2D_index_i])/(2*(u-i));
    return Sep;
}

void Meijster2D(char*           targetArray,
                int*            targetEDT,
                sizeInfo2D*     structureInfo,
                int             primaryPhase)
{
    /*
        The primary phase determines which phase will be used as the anchor to calculate the EDT.
        In other words, a primary phase of 1 means the EDT will be calculated on phase 0, and the
        distances are relative to phase 1.
    */
    int height, width;
    
    height = structureInfo->height;
    width = structureInfo->width;
    long int nElements = structureInfo->nElements;

    int* g = (int *)malloc(sizeof(int)*nElements);

    // First phase of the array: computation of g

    for(int j = 0; j < width; j++)
    {
        // scan 1
        if (targetArray[j] == primaryPhase) g[j] = 0;
        else g[j] = height+width;

        for (int i = 1; i < height; i++)
        {
            if (targetArray[i*width + j] == primaryPhase) g[i*width + j] = 0;
            else g[i*width + j] = 1 + g[(i - 1)*width + j];
        }
        // scan 2
        for(int i = height - 2; i >= 0; i--)
        {
            if (g[(i+1)*width + j] < g[i*width + j]) g[i*width + j] = 1 + g[(i+1)*width + j];
        }
    }

    // print g

    // if (primaryPhase){
    //     FILE *OUT;

    //     OUT = fopen("g_test_pore.csv", "w+");
    //     fprintf(OUT, "row,col,g\n");
    //     for(int i = 0; i<height; i++){
    //         for(int j = 0; j<width; j++){
    //             fprintf(OUT, "%d,%d,%d\n",i,j,g[i*width + j]);
    //         }
    //     }

    //     fclose(OUT);
    // } else{
    //     FILE *OUT;

    //     OUT = fopen("g_test_particle.csv", "w+");
    //     fprintf(OUT, "row,col,g\n");
    //     for(int i = 0; i<height; i++){
    //         for(int j = 0; j<width; j++){
    //             fprintf(OUT, "%d,%d,%d\n",i,j,g[i*width + j]);
    //         }
    //     }

    //     fclose(OUT);
    // }

    // FILE *OUT;

    // OUT = fopen("g_test.csv", "w+");
    // fprintf(OUT, "row,col,g\n");
    // for(int i = 0; i<height; i++){
    //     for(int j = 0; j<width; j++){
    //         fprintf(OUT, "%d,%d,%d,%d\n",i,j,g[i*width + j]);
    //     }
    // }

    // fclose(OUT);

    // Second Phase

    int* s = (int *)malloc(sizeof(int)*width);
    int* t = (int *)malloc(sizeof(int)*width);

    memset(s, 0, sizeof(int)*width);
    memset(t, 0, sizeof(int)*width);

    for(int row = 0; row < height; row++)
    {
        int q = 0;
        int w = 0;
        s[0] = 0;
        t[0] = 0;
        // scan 3
        for(int u = 1; u<width; u++)
        {
            while (q >= 0 && f_EDT_2D(g, row*width + s[q], t[q], s[q]) > f_EDT_2D(g, row*width + u, t[q], u)) q = q - 1;

            if (q < 0){
                q = 0;
                s[0] = u;
            }else
            {
                w = 1 + Sep_EDT_2D(g, s[q], u, row*width + s[q], row*width + u);
                if (w < width)
                {
                    q = q + 1;
                    s[q] = u;
                    t[q] = w;
                }
            }
        }
        // scan 4
        for(int u = width - 1; u >= 0; u--)
        {
            targetEDT[row*width + u] = f_EDT_2D(g, row*width + s[q], u, s[q]);
            if(u == t[q]) q = q - 1;
        }
    }

    free(g);
    free(s);
    free(t);

    return;
}

void Meijster3D(char*           targetArray,
                int*            targetEDT,
                sizeInfo*       structureInfo,
                int             primaryPhase)
{
    /*
        The primary phase determines which phase will be used as the anchor to calculate the EDT.
        In other words, a primary phase of 1 means the EDT will be calculated on phase 0, and the
        distances are relative to phase 1.
    */
    int height, width, depth;
    
    height = structureInfo->height;
    width = structureInfo->width;
    depth = structureInfo->depth;
    long int nElements = structureInfo->nElements;
    long int index = 0;

    int* g = (int *)malloc(sizeof(int)*nElements);

    // First phase of the array: computation of g (column-wise operation)

    // index = slice*(width*height) + row*(width) + col

    for(int j = 0; j < width; j++)
    {
        for(int k = 0; k<depth; k++)
        {
            // scan 1 (top to bottom)
            if (targetArray[k*(width*height) + j] == primaryPhase) g[k*(width*height) + j] = 0;
            else g[k*(width*height) + j] = height+width;

            for (int i = 1; i<height; i++)
            {
                index = k*(width*height) + i*width + j;
                if (targetArray[index] == primaryPhase) g[index] = 0;
                else g[index] = 1 + g[index - width];
            }

            // scan 2 (bottom to top)
            for(int i = height-2; i >= 0; i--)
            {
                index = k*(width*height) + i*width + j;
                if (g[index + width] < g[index]) g[index] = 1 + g[index + width];
            }
        }
    }

    // second phase, computation of distance transforms explicitly

    int* s = (int *)malloc(sizeof(int)*width*height);
    int* t = (int *)malloc(sizeof(int)*width*height);

    memset(s, 0, sizeof(int)*width*height);
    memset(t, 0, sizeof(int)*width*height);

    // begin scans

    for(int k = 0; k<depth; k++)
    {   // all slices
        for(int i = 0; i<height; i++)
        {   // all rows
            // forward pass 3 (fixed row and slice)
            int q = 0;
            int w = 0;
            s[0] = 0;
            t[0] = 0;
            for (int u = 1; u<width; u++)
            {   // all columns
                while(q >= 0 && 
                            f_EDT_2D(g, k*width*depth + i*width + s[q], t[q], s[q]) > 
                                    f_EDT_2D(g, k*width*depth + i*width + u, t[q], u)) q--;
                if(q < 0)
                {
                    q = 0;
                    s[0] = u;
                    t[0] = 0;
                } else
                {
                    w = 1 + Sep_EDT_2D(g, s[q], u, k*width*height + i*width + s[q], k*width*height + i*width + u);
                    if (w < width)
                    {
                        q++;
                        s[q] = u;
                        t[q] = w;
                    }
                }
            }

            // backward scan
            for(int u = width - 1; u >= 0; u--)
            {
                targetEDT[k*width*height + i*width + u] = f_EDT_2D(g, k*width*height + i*width + s[q], u, s[q]);
                if(u == t[q]) q--;
            }
        }
    }

    // reset s and t

    memset(s, 0, sizeof(int)*width*height);
    memset(t, 0, sizeof(int)*width*height);

    // now for all slices

    for(int j = 0; j<width; j++)
    {
        for(int i = 0; i<height; i++)
        {
            // forward pass 3
            int q = 0;
            int w = 0;
            s[0] = 0;
            t[0] = 0;

            for(int u = 1; u<depth; u++){
                while(q >= 0 && 
                            f_EDT_3D(targetEDT, s[q]*width*depth + i*width + j, t[q], s[q]) > 
                                    f_EDT_3D(targetEDT, u*width*depth + i*width + j, t[q], u)) q--;

                if(q < 0)
                {
                    q = 0;
                    s[0] = u;
                    t[0] = 0;
                } else
                {
                    w = 1 + Sep_EDT_3D(targetEDT, s[q], u, s[q]*width*height + i*width + j, u*width*height + i*width + j);
                    if (w < depth)
                    {
                        q++;
                        s[q] = u;
                        t[q] = w;
                    }
                }
            }
            // backward scan
            for(int u = depth - 1; u >= 0; u--)
            {
                targetEDT[u*width*height + i*width + j] = f_EDT_3D(targetEDT, s[q]*width*height + i*width + j, u, s[q]);
                if(u == t[q]) q--;
            }
        }
    }

    // memory management

    free(s);
    free(t);
    free(g);

    return;
}

void ParticleLabel2D(   int             rMin,
                        int             rMax,
                        char*           R,
                        int*            L,
                        sizeInfo2D*     structureInfo)
{

    // open list

    std::list<long int> oList;

    // read size

    int height, width;
    height = structureInfo->height;
    width = structureInfo->width;
    long int nElements = structureInfo->nElements;

    // Loop variables

    int myRow, myCol;
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

                myRow = index/width;
                myCol = index - myRow*width;

                // Search Neighbords with the same r

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
            printf("Particle Label = %d, rad = %d\n", particleLabel, r);
        }   // end for
        r++;                        // increase radius label
    } // end while

    return;
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

int Hybrid_particleSD_2D(char*          P,
                         char*          E, 
                         char*          D,
                         char*          R, 
                         long int*      InterfaceArray, 
                         int            radius, 
                         int            numThreads, 
                         char*          output_name,
                         sizeInfo2D*    structureInfo,
                         bool           debugFlag)
{
    // read data structure
    int height, width;
    height = structureInfo->height;
    width = structureInfo->width;
    long int nElements = structureInfo->nElements;

    // loop variables

    long int p_sum, d_sum, e_sum;
    p_sum = 1;
    e_sum = 1;
    d_sum = 1;
    int primaryPhase = 0;

    int* size = (int *)malloc(sizeof(int)*2);
    size[0] = height;
    size[1] = width;

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
    CHECK_CUDA( cudaMalloc((void **) &d_Size, 2*sizeof(int)));

    // Define

    CHECK_CUDA( cudaMemcpy(d_P, P, nElements*sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(d_E, P, nElements*sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(d_D, P, nElements*sizeof(char), cudaMemcpyHostToDevice));

    CHECK_CUDA( cudaMemset(d_InterfaceArray, 0, nElements*sizeof(long int)));
    CHECK_CUDA( cudaMemcpy(d_Size, size, 2*sizeof(int), cudaMemcpyHostToDevice));
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

        interfaceCount = FindInterface_2D(P, InterfaceArray, structureInfo, primaryPhase, numThreads);

        // Calculate number of blocks and threads per block based on radius

        int num_blocks, threads_per_block;

        num_blocks = interfaceCount;
        threads_per_block = 2*radius + 1;

        // Copy interface array into the GPU

        CHECK_CUDA( cudaMemcpy(d_InterfaceArray, InterfaceArray, sizeof(long int)*nElements, cudaMemcpyHostToDevice) );

        // Do Dilation on GPU

        CheckR_D_2D<<<num_blocks, threads_per_block>>>(d_D, d_R, d_Size, d_InterfaceArray);

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

        interfaceCount = FindInterface_2D(D, InterfaceArray, structureInfo, primaryPhase, numThreads);

        // GPU Operations

        num_blocks = interfaceCount;

        CHECK_CUDA( cudaMemcpy(d_InterfaceArray, InterfaceArray, sizeof(long int)*nElements, cudaMemcpyHostToDevice) );

        // Perform Erosion

        CheckR_E_2D<<<num_blocks, threads_per_block>>>(d_E, d_R, d_Size, d_InterfaceArray);

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
    }


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

int particleSD_2D_Meijster( char*          P,
                            char*          E, 
                            char*          D,
                            char*          R, 
                            long int*      InterfaceArray, 
                            int            radius, 
                            int            numThreads, 
                            char*          output_name,
                            sizeInfo2D*    structureInfo,
                            bool           debugFlag)
{
    // read data structure
    int height, width;
    height = structureInfo->height;
    width = structureInfo->width;
    long int nElements = structureInfo->nElements;

    // loop variables

    long int p_sum, d_sum, e_sum;
    p_sum = 1;
    e_sum = 1;
    d_sum = 1;
    int primaryPhase = 0;

    int* EDT_Pore = (int *)malloc(sizeof(int)*nElements);
    int* EDT_Particle = (int *)malloc(sizeof(int)*nElements);

    memset(EDT_Pore, 0, sizeof(int)*nElements);
    memset(EDT_Particle, 0, sizeof(int)*nElements);

    // Open output file

    FILE *OUT;

    OUT = fopen(output_name, "w+");
    fprintf(OUT, "R,P,E,D\n");

    // EDT for particle only needs to be done once, since we copy P into E at the first step each time

    Meijster2D(P, EDT_Particle, structureInfo, 0);

    while (e_sum != 0 && radius < MAX_R )
    {
        memcpy(D,P,sizeof(char)*nElements);     // probably not necessary

        for(int row = 0; row<height; row++)
        {
            for(int col = 0; col<width; col++)
            {
                if(EDT_Particle[row*width + col] <= radius*radius) D[row*width + col] = 0;
            }
        }

        // copy D into E

        memcpy(E, D, sizeof(char)*nElements);

        // Meijster algorithm in D for pore-space EDT

        Meijster2D(D, EDT_Pore, structureInfo, 1);

        // Update D
        for(int row = 0; row<height; row++)
        {
            for(int col = 0; col<width; col++)
            {
                if(EDT_Pore[row*width + col] <= radius*radius) E[row*width + col] = 1;
            }
        }

        // evaluate sums 

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
        
        // print to output file

        if(debugFlag) printf("R = %d, P = %ld, E = %ld, D = %ld\n", radius, p_sum, e_sum, d_sum);
        fprintf(OUT, "%d,%ld,%ld,%ld\n", radius, p_sum, e_sum, d_sum);

        radius++;
    }

    // close files
    fclose(OUT);

    // free memory on host

    free(EDT_Particle);
    free(EDT_Pore);

    return radius;
}


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


int particleSD_3D_Meijster( char*      P,
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

    int* EDT_Pore = (int *)malloc(sizeof(int)*nElements);
    int* EDT_Particle = (int *)malloc(sizeof(int)*nElements);

    memset(EDT_Pore, 0, sizeof(int)*nElements);
    memset(EDT_Particle, 0, sizeof(int)*nElements);

    /*-------------------------------------------------------
    
                        Main Loop
    
    -------------------------------------------------------*/
    // EDT for particle only needs to be done once, since we copy P into E at the first step each time

    Meijster3D(P, EDT_Particle, structureInfo, 0);

    // Open output file

    FILE *OUT;

    OUT = fopen(output_name, "w+");
    fprintf(OUT, "R,P,E,D\n");

    long int interfaceCount = 0;

    while (e_sum != 0 && radius < MAX_R )
    {
        memcpy(D,P,sizeof(char)*nElements);     // probably not necessary

        for(int i = 0; i<nElements; i++){
            if(EDT_Particle[i] <= radius*radius) D[i] = 0;
        }

        // copy D into E

        memcpy(E, D, sizeof(char)*nElements);

        // Meijster algorithm in D for pore-space EDT
        
        // memset(EDT_Pore, 0, sizeof(int)*nElements);

        Meijster3D(D, EDT_Pore, structureInfo, 1);

        // Update E
        for(int i = 0; i<nElements; i++){
            if(EDT_Pore[i] <= radius*radius) E[i] = 1;
        }

        // evaluate sums 

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
        
        // print to output file

        if(debugFlag) printf("R = %d, P = %ld, E = %ld, D = %ld\n", radius, p_sum, e_sum, d_sum);
        fprintf(OUT, "%d,%ld,%ld,%ld\n", radius, p_sum, e_sum, d_sum);

        radius++;
    } // end while

    /*---------------------------------------------------------------------
    
                            Memory Management

    ------------------------------------------------------------------------*/
    // Files

    fclose(OUT);

    // Device

    // Host

    free(EDT_Pore);
    free(EDT_Particle);

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
    /*---------------------------------------------------------------------
    
                            Read Input
                                &
                          Declare Arrays 

        Input mode flags:
        - Flag = 0 means jpg image (using stb image).
        - Flag = 1 means tiff file.             (NOT IMPLEMENTED)

    ------------------------------------------------------------------------*/

    // User set variables

    int radius_offset = 0;
    int radius = 1;

    unsigned char* target_img;
    sizeInfo2D imgInfo;
    char targetName[50];
    char output_name[100];
    bool saveLabels = false;

    char labelsOutput_name[100];

    // File names

    sprintf(targetName, "00000.jpg");
    sprintf(output_name, "00_ED_test.csv");

    sprintf(labelsOutput_name, "Test_labels2D.csv");

    // Parallel Computing Options Options

    int numThreads = 8;                         // Number of CPU threads to be used

    // Read image

    if (readImg_2D( targetName, &target_img, &imgInfo) == 1) printf("Error, image has wrong number of channels\n");

    if(readImg_2D( targetName, &target_img, &imgInfo) == 1) return 1;

    char* P = (char *)malloc(sizeof(char)*imgInfo.nElements);

    memset(P, 0, sizeof(char)*imgInfo.nElements);

    // Cast image into P, free original array

    for(int i = 0; i < imgInfo.nElements; i++)
    {
        if(target_img[i] < 150)
        {
            P[i] = 0;
        }else
        {
            P[i] = 1;
        }
    }

    // int* EDT_Pore = (int *)malloc(sizeof(int)*imgInfo.nElements);
    // int* EDT_Particle = (int *)malloc(sizeof(int)*imgInfo.nElements);

    // memset(EDT_Pore, 0, sizeof(int)*imgInfo.nElements);
    // memset(EDT_Particle, 0, sizeof(int)*imgInfo.nElements);

    // Meijster2D(P, EDT_Pore, &imgInfo, 1);
    // Meijster2D(P, EDT_Particle, &imgInfo, 0);

    // FILE *OUT;

    // OUT = fopen("PSD_Meijster_test.csv", "w+");
    // fprintf(OUT, "row,col,Pore,Part\n");
    // for(int i = 0; i<imgInfo.height; i++){
    //     for(int j = 0; j<imgInfo.width; j++){
    //         fprintf(OUT, "%d,%d,%d,%d\n",i,j,EDT_Pore[i*imgInfo.width + j],EDT_Particle[i*imgInfo.width + j]);
    //     }
    // }

    // fclose(OUT);

    // return 0;

    free(target_img);

    // Allocate Space for Erosion-Dilation

    char *E = (char *)malloc(sizeof(char)*imgInfo.nElements);
    char *D = (char *)malloc(sizeof(char)*imgInfo.nElements);

    // Initialize ED arrays

    memset(E, 0, sizeof(char) * imgInfo.nElements);
    memset(D, 0, sizeof(char) * imgInfo.nElements);

    // Label Arrays

    char *R = (char *)malloc(sizeof(char)* imgInfo.nElements);        // array for particle radius
    int  *L =  (int *)malloc(sizeof(int) * imgInfo.nElements);        // array for particle labels

    // Interface array

    long int *InterfaceArray = (long int *)malloc(sizeof(long int) * imgInfo.nElements);

    // Initialize label arrays and Interface Array

    memset(R,                0, sizeof(char)    * imgInfo.nElements);
    memset(L,                0, sizeof(int)     * imgInfo.nElements);
    memset(InterfaceArray,   0, sizeof(long int)* imgInfo.nElements);

    if (debugMode) printf("Structure Read\n");

    // Prepare variables for the main loop

    for(long int i = 0; i<imgInfo.nElements; i++)
    {
        R[i] = -1;
        L[i] = -1;
    }

    // set radius offset, if any

    if(radius_offset > 0) radius = radius_offset;

    /*---------------------------------------------------------------------
    
                            Main Loop

    ------------------------------------------------------------------------*/

    if (debugMode) printf("Starting Main Loop\n");

    // radius = Hybrid_particleSD_2D(P, E, D, R, InterfaceArray, radius,
    //                               numThreads, output_name, &imgInfo, debugMode);

    radius = particleSD_2D_Meijster(P, E, D, R, InterfaceArray, radius,
                                    numThreads, output_name, &imgInfo, debugMode);

    /*---------------------------------------------------------------------
    
                            Save Labels

    ------------------------------------------------------------------------*/

    if(saveLabels)
    {
        // Label Particles

        ParticleLabel2D(3, radius, R, L, &imgInfo);
        saveLabels2D(P, R, L, &imgInfo, labelsOutput_name);
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

    if (inputMode == 0)
    {
        if (debugMode) printf("Reading .csv\n");
        sprintf(target_name, "rec_837_300_int1.csv");
        readCSV(target_name, P, &structureInfo, debugMode);
    } else if(inputMode == 1)
    {
        if (debugMode) printf("Reading Stack\n");
        if (readStack(P, &structureInfo, debugMode) == 1)
        {
            printf("Error encountered when reading stack. Consider using Debug Mode\n");
            printf("Exiting now.\n");
            return 1;
        }
    }
    
    if (debugMode) printf("Structure Read\n");

    // Prepare variables for the main loop

    for(long int i = 0; i<structureInfo.nElements; i++)
    {
        R[i] = -1;
        L[i] = -1;
    }

    if(radius_offset > 0) radius = radius_offset;

    /*---------------------------------------------------------------------
    
                            Main Loop

    ------------------------------------------------------------------------*/

    if (debugMode) printf("Starting Main Loop\n");

    // radius = Hybrid_particleSD_3D(P, E, D, R, InterfaceArray, radius,
    //                               numThreads, output_name, &structureInfo, debugMode);
    radius = particleSD_3D_Meijster(P, E, D, R, InterfaceArray, radius,
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