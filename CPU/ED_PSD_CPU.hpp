#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <string>
#include <fstream>
#include <stdbool.h>
#include <omp.h>
#include <list>

// Load stb for reading jpg's

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

/*--------------------------------------------------------

                     Constants

 --------------------------------------------------------*/

#define MAX_R 100

/*--------------------------------------------------------

                    Data structures

 --------------------------------------------------------*/

typedef struct
{
    int nD;
    char *inputFilename;
    char *poreSD_Out;
    char *partSD_Out;
    char *poreLabel_Out;
    char *partLabel_Out;
    bool poreSD;
    bool partSD;
    bool poreLabel;
    bool partLabel;
    bool verbose;
    int nThreads;
    int height;
    int width;
    int depth;
    char inputType;
    bool batchFlag;
    unsigned char TH;
    int maxR;
} options;

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

/*

    Functions for reading and printing user input:

*/

void printOpts(options *opts)
{
    /*
        Function printOpts:
        Inputs:
            - pointer to opts struct
        Output:
            - None

        Function will print user options to the command line.
    */

    printf("--------------------------------------\n\n");
    printf("c-PSD Simulation\n");
    printf("Current selected options:\n\n");
    printf("--------------------------------------\n");
    printf("Number of Dimensions: %d\n", opts->nD);
    printf("InputType = %d\n", opts->inputType);

    // separate dimensions
    if (opts->nD == 2)
    {
        if (opts->inputType == 0)
        {
            printf("Input Name: %s\n", opts->inputFilename);
            printf("Phase Threshold: %d\n", (int)opts->TH);
        }
        // options related to simulation type
        if (opts->poreSD)
        {
            printf("Pore Size Distribution Output: %s\n", opts->poreSD_Out);
        }
        if (opts->poreLabel)
        {
            printf("Pore Label Output: %s\n", opts->poreLabel_Out);
        }
        if (opts->partSD)
        {
            printf("Partciel Size Distribution Output: %s\n", opts->partSD_Out);
        }
        if (opts->partLabel)
        {
            printf("Particle Label Output: %s\n", opts->partLabel_Out);
        }
    }

    // nThreads

    printf("Num. threads: %d\n", opts->nThreads);
    printf("Max. Scan Radius: %d\n", opts->maxR);

    printf("--------------------------------------\n\n");

    return;
}

int readInput(char *inputFilename, options *opts)
{
    /*
        Function readInput:
        Inputs:
            - pointer to array holding the input file name
            - pointer to optins struct
        Outputs:
            - none

        Function will read the user entered information from the input file and
        populate the opts array with the relevant information.
    */
    // open file and start reading
    std::string myText;

    char tempC[1000];
    double tempD;
    char tempFilenames[1000];

    std::ifstream InputFile(inputFilename);

    // defaults

    opts->nThreads = 1;
    opts->verbose = true;
    opts->batchFlag = false;
    opts->inputType = 0;

    opts->poreSD = false;
    opts->partSD = false;
    opts->partLabel = false;
    opts->poreSD = false;

    opts->TH = 128;

    opts->maxR = 100;

    /*
    --------------------------------------------------------------------------------

    If anybody has a better idea of how to parse inputs please let me know.
    Eventually I'm hoping the GUI will replace a lot of this code.

    --------------------------------------------------------------------------------
    */

    while (std::getline(InputFile, myText))
    {
        sscanf(myText.c_str(), "%s %lf", tempC, &tempD);
        if (strcmp(tempC, "nD:") == 0)
        {
            opts->nD = (int)tempD;
        }
        else if (strcmp(tempC, "inputFilename:") == 0)
        {
            sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
            opts->inputFilename = (char *)malloc(1000 * sizeof(char));
            strcpy(opts->inputFilename, tempFilenames);
        }
        else if (strcmp(tempC, "poreOut:") == 0)
        {
            opts->poreSD = true;
            sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
            opts->poreSD_Out = (char *)malloc(1000 * sizeof(char));
            strcpy(opts->poreSD_Out, tempFilenames);
        }
        else if (strcmp(tempC, "partOut:") == 0)
        {
            opts->partSD = true;
            sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
            opts->partSD_Out = (char *)malloc(1000 * sizeof(char));
            strcpy(opts->partSD_Out, tempFilenames);
        }
        else if (strcmp(tempC, "poreLabelOut:") == 0)
        {
            opts->poreLabel = true;
            sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
            opts->poreLabel_Out = (char *)malloc(1000 * sizeof(char));
            strcpy(opts->poreLabel_Out, tempFilenames);
        }
        else if (strcmp(tempC, "partLabelOut:") == 0)
        {
            opts->partLabel = true;
            sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
            opts->partLabel_Out = (char *)malloc(1000 * sizeof(char));
            strcpy(opts->partLabel_Out, tempFilenames);
        }
        else if (strcmp(tempC, "nThreads:") == 0)
        {
            opts->nThreads = (int)tempD;
        }
        else if (strcmp(tempC, "verbose:") == 0)
        {
            if ((int)tempD == 0)
                opts->verbose = false;
            else if ((int)tempD == 1)
                opts->verbose = true;
            else
                printf("Invalid verbose, default to 'true'.\n");
        }
        else if (strcmp(tempC, "width:") == 0)
        {
            opts->width = (int)tempD;
        }
        else if (strcmp(tempC, "height:") == 0)
        {
            opts->height = (int)tempD;
        }
        else if (strcmp(tempC, "depth:") == 0)
        {
            opts->depth = (int)tempD;
        }
        else if (strcmp(tempC, "TH:") == 0)
        {
            opts->TH = (unsigned char)tempD;
        }
        else if (strcmp(tempC, "inputType:") == 0)
        {
            opts->inputType = (int)tempD;
        }
    }
    return 0;
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
    int Sep = (u*u - i*i + g[g_index_u]*g[g_index_u] - g[g_index_i]*g[g_index_i])/(2*(u-i));
    return Sep;
}

int Sep_EDT_3D( int*        EDT2D,
                int         i,
                int         u,
                int         EDT2D_index_i,
                int         EDT2D_index_u)
{
    int Sep = (u*u - i*i + EDT2D[EDT2D_index_u] - EDT2D[EDT2D_index_i])/(2*(u-i));
    return Sep;
}

int pass12_Global(char* target_arr, double* EDT, int j, int k, int width, int height, int depth, int primaryPhase){
    int stride = width;
    int offset = k*(width*height) + j;

    if(target_arr[offset] == primaryPhase) EDT[offset] = 0;
    else EDT[offset] = height + width + depth;

    // scan 1: forward
    for(int i = 1; i<height; i++)
    {
        if(target_arr[offset+i*stride] == primaryPhase) EDT[offset+i*stride] = 0;
        else EDT[offset+i*stride] = 1 + EDT[offset + (i-1)*stride];
    }
    // scan 2: backward
    for(int i = height - 2; i >= 0; i--)
    {
        if(EDT[offset + (i + 1)*stride] < EDT[offset + i*stride]) EDT[offset + i*stride] = 1 + EDT[offset + (i + 1)*stride];
    }
    return 0;
}

int pass34_Global(double* EDT, double* EDT_temp, int scanLength, int stride, int* s, int* t)
{
    for(int i = 0; i<scanLength; i++) EDT_temp[i] = EDT[i*stride];

    int q = 0;
    double w = 0;
    s[0] = 0;
    t[0] = 0;

    // forward scan

    for(int u = 1; u<scanLength; u++)
    {
        while(q >= 0 && (   (pow((t[q] - s[q]),2) + pow(EDT_temp[s[q]], 2)) >= 
                            (pow((t[q] - u),2) + pow(EDT_temp[u], 2)) )) q--;
        
        if(q < 0){
            q = 0;
            s[0] = u;
            t[0] = 0;
        } else
        {
            w = 1 + trunc(
                (pow(u,2) - pow(s[q],2) + pow(EDT_temp[u],2) - pow(EDT_temp[s[q]],2))
                /(2*(double)(u - s[q]))  );
            if(w < scanLength)
            {
                q++;
                s[q] = u;
                t[q] = (int) w;
            }
        }
    }

    // backward update
    for(int u = scanLength - 1; u >=0; u--)
    {
        EDT[u*stride] = sqrt(pow(u-s[q],2) + pow(EDT_temp[s[q]],2));
        if(u == t[q]) q--;
    }

    return 0;
}

int pass12_2D(bool* target_arr, int* g, int height, int width, int offset, int primaryPhase)
{
    // scan 1
    if (target_arr[offset] == primaryPhase) g[offset] = 0;
    else g[offset] = height+width;

    for (int i = 1; i < height; i++)
    {
        if (target_arr[i*width + offset] == primaryPhase) g[i*width + offset] = 0;
        else g[i*width + offset] = 1 + g[(i - 1)*width + offset];
    }
    // scan 2
    for(int i = height - 2; i >= 0; i--)
    {
        if (g[(i+1)*width + offset] < g[i*width + offset]) g[i*width + offset] = 1 + g[(i+1)*width + offset];
    }
    return 0;
}

int pass34_2D(int* g, int* targetEDT, int width, int offset, int* s, int* t)
{
    // initialize local s and t
    int q = 0;
    int w = 0;
    s[0] = 0;
    t[0] = 0;

    // scan 3
    for(int u = 1; u<width; u++)
    {
        while (q >= 0 && f_EDT_2D(g, offset*width + s[q], t[q], s[q]) > f_EDT_2D(g, offset*width + u, t[q], u)) q = q - 1;

        if (q < 0){
            q = 0;
            s[0] = u;
        }else
        {
            w = 1 + Sep_EDT_2D(g, s[q], u, offset*width + s[q], offset*width + u);
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
        targetEDT[offset*width + u] = f_EDT_2D(g, offset*width + s[q], u, s[q]);
        if(u == t[q]) q = q - 1;
    }

    return 0;
}


int pass12_3D(char* targetArray, int* g, int j, int k, int primaryPhase, int width, int height)
{
    // scan 1 (top to bottom)
    if (targetArray[k*(width*height) + j] == primaryPhase) g[k*(width*height) + j] = 0;
    else g[k*(width*height) + j] = height+width;

    for (int i = 1; i<height; i++)
    {
        int index = k*(width*height) + i*width + j;
        if (targetArray[index] == primaryPhase) g[index] = 0;
        else g[index] = 1 + g[index - width];
    }

    // scan 2 (bottom to top)
    for(int i = height-2; i >= 0; i--)
    {
        int index = k*(width*height) + i*width + j;
        if (g[index + width] < g[index]) g[index] = 1 + g[index + width];
    }
    return 0;
}


int pass34_3D(int* g, int* targetEDT, int* s, int* t, int k, int i, int width, int height)
{
    // forward pass 3 (fixed row and slice)
    int q = 0;
    int w = 0;
    s[0] = 0;
    t[0] = 0;
    for (int u = 1; u<width; u++)
    {   // all columns
        while(q >= 0 && 
                    f_EDT_2D(g, k*width*height + i*width + s[q], t[q], s[q]) > 
                            f_EDT_2D(g, k*width*height + i*width + u, t[q], u)) q--;
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
    return 0;
}


int pass56_3D(int* targetEDT, int* s, int* t, int j, int i, int width, int height, int depth)
{
    // forward pass 3
    int q = 0;
    int w = 0;
    s[0] = 0;
    t[0] = 0;

    for(int u = 1; u<depth; u++){
        while(q >= 0 && 
                    f_EDT_3D(targetEDT, s[q]*width*height + i*width + j, t[q], s[q]) > 
                            f_EDT_3D(targetEDT, u*width*height + i*width + j, t[q], u)) q--;

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
    return 0;
}

int pass56_debug(int* EDT, int* s, int* t, int depth, int stride){

    int* EDT_temp = (int*)malloc(sizeof(int)*depth);

    int q = 0;
    int w = 0;

    s[0] = 0;
    t[0] = 0;

    for(int i = 0; i<depth; i++){
        EDT_temp[i] = sqrt(EDT[i*stride]);
    }

    for(int u = 1; u<depth; u++){
        while(q>=0 && ( (pow((t[q]-s[q]),2) + pow(EDT_temp[s[q]],2)) >=
                            (pow((t[q] - u),2) + pow(EDT_temp[u],2)) ) )
                            q--;
        
        if(q<0){
            q = 0;
            s[0] = u;
            t[0] = 0;
        }
        else
        {
            w = 1 + trunc(
                (pow(u,2) - pow(s[q],2) + pow(EDT_temp[u],2) - pow(EDT_temp[s[q]],2))/(2*(u - s[q]))
            );
            if(w < depth){
                q++;
                s[q] = u;
                t[q] = w;
            }
        }
    }

    // backward pass

    for(int u = depth - 1; u >= 0; u--)
    {
        EDT[u*stride] = pow((u - s[q]), 2) + pow(EDT_temp[s[q]],2);
        if(u <= t[q]) q--;
    }
    free(EDT_temp);
    return 0;
}

void pMeijster2D(bool*           targetArray,
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

    #pragma omp parallel
    {
        // local DT, s, and t for each column scan (fixed row)
        int* s = (int *)malloc(sizeof(int)*width);
        int* t = (int *)malloc(sizeof(int)*width);
        memset(s, 0, sizeof(int)*width);
        memset(t, 0, sizeof(int)*width);
        // phase 1
        #pragma omp for schedule(auto)
        for(int j = 0; j < width; j++)
        {
            int offset = j;
            pass12_2D(targetArray, g, height, width, offset, primaryPhase);
        }
        // phase 2
        #pragma omp for schedule(auto)
        for(int row = 0; row<height; row++)
        {
            int offset = row;
            pass34_2D(g, targetEDT, width, offset, s, t);
        }
        free(s);
        free(t);
    }

    free(g);

    return;
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

void pMeijster3D_debug( char*           targetArray,
                        double*         targetEDT,
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

    #pragma omp parallel
    {
        // initialize s and t locally
        int* s = (int *)malloc(sizeof(int)*width*height);
        int* t = (int *)malloc(sizeof(int)*width*height);
        memset(s, 0, sizeof(int)*width*height);
        memset(t, 0, sizeof(int)*width*height);

        int LL = (height>width) ? height : width;
        LL = (depth > LL) ? depth : LL;

        double* tempEDT = (double *)malloc(sizeof(double)*LL);

        // phase 1

        for(int j = 0; j<width; j++)
        {
            #pragma omp for schedule(auto)
            for(int k = 0; k<depth; k++)
            {
                pass12_Global(targetArray, targetEDT, j, k, width, height, depth, primaryPhase);
            }
        }

        // phase 2

        for(int k = 0; k<depth; k++)
        {   // all slices
            #pragma omp for schedule(auto)
            for(int i = 0; i<height; i++)
            {   // all rows
                size_t offset = k*width*height + i*width;
                pass34_Global(targetEDT + offset, tempEDT, width, 1, s, t);
            }
        }

        // maybe not necessary, but clean arrays anyways
        memset(s, 0, sizeof(int)*width*height);
        memset(t, 0, sizeof(int)*width*height);

        for(int j = 0; j<width; j++)
        {
            #pragma omp for schedule(auto)
            for(int i = 0; i<height; i++)
            {
                // pass56_3D(targetEDT, s, t, j, i, width, height, depth);
                size_t offset = i*width + j;
                pass34_Global(targetEDT + offset, tempEDT, depth, height*width, s, t);
            }
        }
        free(s);
        free(t);
    }

    return;
}

void pMeijster3D(char*           targetArray,
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

    #pragma omp parallel
    {
        // initialize s and t locally
        int* s = (int *)malloc(sizeof(int)*width*height);
        int* t = (int *)malloc(sizeof(int)*width*height);
        memset(s, 0, sizeof(int)*width*height);
        memset(t, 0, sizeof(int)*width*height);

        int LL = (height>width) ? height : width;
        LL = (depth > LL) ? depth : LL;

        double* tempEDT = (double *)malloc(sizeof(double)*LL);

        // phase 1

        for(int j = 0; j<width; j++)
        {
            #pragma omp for schedule(auto)
            for(int k = 0; k<depth; k++)
            {
                pass12_3D(targetArray, g, j, k, primaryPhase, width, height);
            }
        }

        // phase 2

        for(int k = 0; k<depth; k++)
        {   // all slices
            #pragma omp for schedule(auto)
            for(int i = 0; i<height; i++)
            {   // all rows
                pass34_3D(g, targetEDT, s, t, k, i, width, height);
            }
        }

        // maybe not necessary, but clean arrays anyways
        memset(s, 0, sizeof(int)*width*height);
        memset(t, 0, sizeof(int)*width*height);

        for(int j = 0; j<width; j++)
        {
            #pragma omp for schedule(auto)
            for(int i = 0; i<height; i++)
            {
                // pass56_3D(targetEDT, s, t, j, i, width, height, depth);
                size_t offset = i*width + j;
                pass56_debug(targetEDT + offset, s, t, depth, height*width);
            }
        }
        free(s);
        free(t);
    }

    free(g);
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

int partSD_2D(options *opts,
              sizeInfo2D *info,
              char *P,
              char POI)
{
    /*
        Function partSD_2D:
        Inputs:
            - pointer to options struct
            - pointer to structure info struct
            - pointer to phase-array
    */

    // Loop variables

    long int p_sum, d_sum, e_sum;
    
    e_sum = 1; // have to initialize, otherwise loop won't start

    // arrays for holding the EDM:

    int* EDT_D = (int *)malloc(sizeof(int) * info->nElements);
    int* EDT_E = (int *)malloc(sizeof(int) * info->nElements);

    memset(EDT_D, 0, sizeof(int) * info->nElements);
    memset(EDT_E, 0, sizeof(int) * info->nElements);

    // arrays for erosion and dilation

    bool* E = (bool *)malloc(sizeof(bool) * info->nElements);
    bool* D = (bool *)malloc(sizeof(bool) * info->nElements);
    bool* B = (bool *)malloc(sizeof(bool) * info->nElements);

    memset(E, 0, sizeof(bool) * info->nElements);
    memset(D, 0, sizeof(bool) * info->nElements);
    memset(B, 0, sizeof(bool) * info->nElements);

    // If POI, bool = 1

    for(int i = 0; i < info->nElements; i++)
    {
        if (P[i] == POI)
                B[i] = 1;
    }

    // EDT for dilation is a one time operation

    pMeijster2D(B, EDT_D, info, 0);     // 0 is the phase that will be dilated

    int radius = 1;

    while(e_sum != 0 && radius < opts->maxR)
    {
        // copy P into D (probably not necessary)

        memcpy(D, P, sizeof(bool) * info->nElements);

        for (int i = 0; i < info->nElements; i++)
        {
            if (EDT_D[i] <= radius * radius)
                D[i] = 0;
        }

        // copy D into E

        memcpy(E, D, sizeof(bool) * info->nElements);

        // Meijster in D

        pMeijster2D(D, EDT_E, info, 1);

        // Update E

        for (int i = 0; i < info->nElements; i++)
        {
            if (EDT_E[i] <= radius * radius)
                E[i] = 1;
        }

        e_sum = 0;
        d_sum = 0;
        p_sum = 0;
        #pragma omp parallel for reduction(+ : p_sum, d_sum, e_sum)
        for (int i = 0; i < info->nElements; i++)
        {
            p_sum += P[i];
            d_sum += D[i];
            e_sum += E[i];
            // if (P[i] - E[i] == 1 && R[i] == -1)
            //     R[i] = radius;
        }

        // print to output file

        if (opts->verbose)
            printf("R = %d, P = %ld, E = %ld, D = %ld\n", radius, p_sum, e_sum, d_sum);

        // increment radius
        radius++;
    }
    
    // memory management

    free(EDT_D);
    free(EDT_E);
    
    free(B);
    free(E);
    free(D);

    return 0;
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

    omp_set_num_threads(numThreads);

    // Open output file

    FILE *OUT;

    OUT = fopen(output_name, "w+");
    fprintf(OUT, "R,P,E,D\n");

    // EDT for particle only needs to be done once, since we copy P into E at the first step each time

    // pMeijster2D(P, EDT_Particle, structureInfo, 0);

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

        // pMeijster2D(D, EDT_Pore, structureInfo, 1);

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

    omp_set_num_threads(numThreads);

    // int* EDT_Pore = (int *)malloc(sizeof(int)*nElements);
    // int* EDT_Particle = (int *)malloc(sizeof(int)*nElements);

    // memset(EDT_Pore, 0, sizeof(int)*nElements);
    // memset(EDT_Particle, 0, sizeof(int)*nElements);
    double* EDT_Pore = (double *)malloc(sizeof(double)*nElements);
    double* EDT_Particle = (double *)malloc(sizeof(double)*nElements);

    memset(EDT_Pore, 0, sizeof(double)*nElements);
    memset(EDT_Particle, 0, sizeof(double)*nElements);

    /*-------------------------------------------------------
    
                        Main Loop
    
    -------------------------------------------------------*/
    // EDT for particle only needs to be done once, since we copy P into E at the first step each time

    // pMeijster3D(P, EDT_Particle, structureInfo, 0);
    pMeijster3D_debug(P, EDT_Particle, structureInfo, 0);

    // FILE *EDT;

    // EDT = fopen("Meijster_xz.csv", "w+");
    // fprintf(EDT, "x,z,EDT\n");

    // for(int j = 0; j<width; j++){
    //     for(int k = 0; k<depth; k++){
    //         fprintf(EDT,"%d,%d,%d\n", j,k, EDT_Particle[k*width*height + j]);
    //     }
    // }

    // fclose(EDT);

    // Open output file

    FILE *OUT;

    OUT = fopen(output_name, "w+");
    fprintf(OUT, "R,P,E,D\n");

    long int interfaceCount = 0;

    while (e_sum != 0 && radius < MAX_R )
    {
        memcpy(D,P,sizeof(char)*nElements);     // probably not necessary

        for(int i = 0; i<nElements; i++){
            if((int) pow(EDT_Particle[i],2) <= radius*radius) D[i] = 0;
        }

        // copy D into E

        memcpy(E, D, sizeof(char)*nElements);

        // Meijster algorithm in D for pore-space EDT
        
        memset(EDT_Pore, 0, sizeof(int)*nElements);

        pMeijster3D_debug(D, EDT_Pore, structureInfo, 1);

        // Update E
        for(int i = 0; i<nElements; i++){
            if((int) pow(EDT_Pore[i],2) <= radius*radius) E[i] = 1;
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

int Sim2D(options *opts)
{
    /*---------------------------------------------------------------------
    
                            Read Input
                                &
                          Declare Arrays 

        Input mode flags:
        - Flag = 0 means jpg image (using stb image).
        - Flag = 1 means tiff file.             (NOT IMPLEMENTED)

    ------------------------------------------------------------------------*/

    // declare structure related variables

    unsigned char* target_img;
    sizeInfo2D imgInfo;

    // set omp options

    omp_set_num_threads(opts->nThreads);

    // read structure

    if(opts->inputType == 0)
    {
        if (readImg_2D( opts->inputFilename, &target_img, &imgInfo) == 1)
                printf("Error, image has wrong number of channels\n");
    } else
    {
        printf("Method not implemented yet!\n");
    }
    
    // Create array to hold structure

    char* P = (char *)malloc(sizeof(char)*imgInfo.nElements);

    memset(P, 0, sizeof(char)*imgInfo.nElements);

    // Cast image into P, free original array

    for (int i = 0; i < imgInfo.nElements; i++)
    {
        if (target_img[i] < opts->TH)
            P[i] = 0;
        else
            P[i] = 1;
    }

    free(target_img);

    // Perform the selected simulations

    if(opts->poreSD)
        partSD_2D(opts, &imgInfo, P, 1);

    
    // Memory Management
    free(P);

    return 0;
}



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