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
    char *readFolder;
    char *saveFolder;
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
    int radOff;
    int stackSize;
    char LeadZero;
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
    }
    else if (opts->nD == 3)
    {
        if (opts->inputType == 0)
        {
            printf("Input Name: %s\n", opts->inputFilename);
            printf("Width  = %d\n", opts->width);
            printf("Height = %d\n", opts->height);
            printf("Depth  = %d\n", opts->depth);
        }
        else if (opts->inputType == 1)
        {
            printf("Stack Mode selected\n");
            printf("Stack size = %d\n", opts->stackSize);
            printf("Stack Threshold = %d\n", opts->TH);
            printf("Leading Zeroes: %d\n", (int)opts->LeadZero);
        }
        else
        {
            printf("Option not yet implemented. Exiting with error status.\n");
            return;
        }
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
    opts->poreLabel = false;

    opts->TH = 128;
    opts->maxR = 100;
    opts->radOff = 0;

    opts->stackSize = 1;
    opts->LeadZero = 5;

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
        else if (strcmp(tempC, "maxR:") == 0)
        {
            opts->maxR = (int)tempD;
        }
        else if (strcmp(tempC, "leadZero:") == 0)
        {
            opts->LeadZero = (char)tempD;
        }
        else if (strcmp(tempC, "nSlices:") == 0)
        {
            opts->stackSize = (int)tempD;
        }
    }
    return 0;
}

int readCSV(char *target_name,
            char *P,
            sizeInfo *structureInfo)
{
    // read data structure

    int height, width;
    height = structureInfo->height;
    width = structureInfo->width;
    long int nElements = structureInfo->nElements;

    // declare arrays to hold coordinate values for solid phase

    int *x = (int *)malloc(sizeof(int) * nElements);
    int *y = (int *)malloc(sizeof(int) * nElements);
    int *z = (int *)malloc(sizeof(int) * nElements);

    // Read structure file

    FILE *target_data;

    target_data = fopen(target_name, "r");

    // check if file exists

    if (target_data == NULL)
    {
        fprintf(stderr, "Error reading file. Exiting program.\n");
        return 1;
    }

    // Read header

    char header[3];

    fscanf(target_data, "%c,%c,%c", &header[0], &header[1], &header[2]);

    // Read coordinates

    size_t count = 0;

    while (fscanf(target_data, "%d,%d,%d", &x[count], &y[count], &z[count]) == 3)
    {
        count++;
    }

    // Build P based on the data we just read

    int index = 0;

    for (long int i = 0; i < count; i++)
    {
        index = z[i] * height * width + y[i] * width + x[i];
        P[index] = 1;
    }

    // Free coordinate vectors and close file

    free(x);
    free(y);
    free(z);

    fclose(target_data);
    return 0;
}

int readImg_2D(char *target_name,
               unsigned char **targetPtr,
               sizeInfo2D *imgInfo)
{
    int channel;

    *targetPtr = stbi_load(target_name, &imgInfo->width, &imgInfo->height, &channel, 1);

    if (channel != 1)
        return 1;

    imgInfo->nElements = imgInfo->height * imgInfo->width;

    return 0;
}

int readStack(char *P,
              options *opts)
{
    long int index;

    char imgName[1000];

    sizeInfo2D tempSize;

    for (int sliceNum = 0; sliceNum < opts->stackSize; sliceNum++)
    {
        // pointer to store image

        unsigned char *target_img;

        // get image name
        sprintf(imgName, "%0*d.jpg", opts->LeadZero, sliceNum);

        // read image

        bool flag = 0;

        flag = readImg_2D(imgName, &target_img, &tempSize);

        if (flag == 1)
        {
            // print error if wrong number of channels and debugMode is on
            if (opts->verbose)
                printf("Wrong number of channels on image %0*d\n", (int)opts->LeadZero, sliceNum);
            return 1;
        }

        // use the information we just read to populate the 3d structure
        for (int rowNum = 0; rowNum < tempSize.height; rowNum++)
        {
            for (int colNum = 0; colNum < tempSize.width; colNum++)
            {
                index = sliceNum * tempSize.width * tempSize.height +
                        rowNum * tempSize.width + colNum;
                // thresholding at opts->TH
                if (target_img[rowNum * tempSize.width + colNum] < opts->TH)
                    P[index] = 0;
                else
                    P[index] = 1;
            }
        }
        free(target_img);
    }

    return 0;
}

int saveLabels2D(int *R,
                 int *L,
                 sizeInfo2D *structureInfo,
                 char *filename)
{
    // read data structure
    int width;

    width = structureInfo->width;

    long int nElements = structureInfo->nElements;

    // Open File

    FILE *OUT;

    OUT = fopen(filename, "w+");

    fprintf(OUT, "x,y,R,L\n");

    int row, col;

    // save everything

    for (int i = 0; i < nElements; i++)
    {
        if (R[i] != -1)
        {
            row = i / width;
            col = i - row * width;
            fprintf(OUT, "%d,%d,%d,%d\n", col, row, R[i], L[i]);
        }
    }

    // close file

    fclose(OUT);

    return 0;
}

int saveLabels3D(int *R,
                 int *L,
                 sizeInfo *structureInfo,
                 char *filename)
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

    int slice, row, col;

    // save everything

    for (int i = 0; i < nElements; i++)
    {
        if (R[i] != -1)
        {
            slice = i / (height * width);
            row = (i - slice * height * width) / width;
            col = (i - slice * height * width - row * width);
            fprintf(Particle, "%d,%d,%d,%d,%d\n", col, row, slice, (int)R[i], L[i]);
        }
    }

    // close file

    fclose(Particle);

    return 0;
}

int f_EDT_2D(int *g,
             int g_index,
             int x,
             int i)
{
    int f = (x - i) * (x - i) + g[g_index] * g[g_index];
    return f;
}

int Sep_EDT_2D(int *g,
               int i,
               int u,
               int g_index_i,
               int g_index_u)
{
    int Sep = (u * u - i * i + g[g_index_u] * g[g_index_u] - g[g_index_i] * g[g_index_i]) / (2 * (u - i));
    return Sep;
}

int pass12_Global(bool *target_arr, float *EDT, int j, int k, int width, int height, int depth, int primaryPhase)
{
    int stride = width;
    int offset = k * (width * height) + j;

    if (target_arr[offset] == primaryPhase)
        EDT[offset] = 0;
    else
        EDT[offset] = height + width + depth;

    // scan 1: forward
    for (int i = 1; i < height; i++)
    {
        if (target_arr[offset + i * stride] == primaryPhase)
            EDT[offset + i * stride] = 0;
        else
            EDT[offset + i * stride] = 1 + EDT[offset + (i - 1) * stride];
    }
    // scan 2: backward
    for (int i = height - 2; i >= 0; i--)
    {
        if (EDT[offset + (i + 1) * stride] < EDT[offset + i * stride])
            EDT[offset + i * stride] = 1 + EDT[offset + (i + 1) * stride];
    }
    return 0;
}

int pass34_Global(float *EDT, float *EDT_temp, int scanLength, int stride, int *s, int *t)
{
    for (int i = 0; i < scanLength; i++)
        EDT_temp[i] = EDT[i * stride];

    int q = 0;
    double w = 0;
    s[0] = 0;
    t[0] = 0;

    // forward scan

    for (int u = 1; u < scanLength; u++)
    {
        while (q >= 0 && ((pow((t[q] - s[q]), 2) + pow(EDT_temp[s[q]], 2)) >=
                          (pow((t[q] - u), 2) + pow(EDT_temp[u], 2))))
            q--;

        if (q < 0)
        {
            q = 0;
            s[0] = u;
            t[0] = 0;
        }
        else
        {
            w = 1 + trunc(
                        (pow(u, 2) - pow(s[q], 2) + pow(EDT_temp[u], 2) - pow(EDT_temp[s[q]], 2)) / (2 * (double)(u - s[q])));
            if (w < scanLength)
            {
                q++;
                s[q] = u;
                t[q] = (int)w;
            }
        }
    }

    // backward update
    for (int u = scanLength - 1; u >= 0; u--)
    {
        EDT[u * stride] = sqrt(pow(u - s[q], 2) + pow(EDT_temp[s[q]], 2));
        if (u == t[q])
            q--;
    }

    return 0;
}

int pass12_2D(bool *target_arr, int *g, int height, int width, int offset, int primaryPhase)
{
    // scan 1
    if (target_arr[offset] == primaryPhase)
        g[offset] = 0;
    else
        g[offset] = height + width;

    for (int i = 1; i < height; i++)
    {
        if (target_arr[i * width + offset] == primaryPhase)
            g[i * width + offset] = 0;
        else
            g[i * width + offset] = 1 + g[(i - 1) * width + offset];
    }
    // scan 2
    for (int i = height - 2; i >= 0; i--)
    {
        if (g[(i + 1) * width + offset] < g[i * width + offset])
            g[i * width + offset] = 1 + g[(i + 1) * width + offset];
    }
    return 0;
}

int pass34_2D(int *g, int *targetEDT, int width, int offset, int *s, int *t)
{
    // initialize local s and t
    int q = 0;
    int w = 0;
    s[0] = 0;
    t[0] = 0;

    // scan 3
    for (int u = 1; u < width; u++)
    {
        while (q >= 0 && f_EDT_2D(g, offset * width + s[q], t[q], s[q]) > f_EDT_2D(g, offset * width + u, t[q], u))
            q = q - 1;

        if (q < 0)
        {
            q = 0;
            s[0] = u;
        }
        else
        {
            w = 1 + Sep_EDT_2D(g, s[q], u, offset * width + s[q], offset * width + u);
            if (w < width)
            {
                q = q + 1;
                s[q] = u;
                t[q] = w;
            }
        }
    }
    // scan 4
    for (int u = width - 1; u >= 0; u--)
    {
        targetEDT[offset * width + u] = f_EDT_2D(g, offset * width + s[q], u, s[q]);
        if (u == t[q])
            q = q - 1;
    }

    return 0;
}

void pMeijster2D(bool *targetArray,
                 int *targetEDT,
                 sizeInfo2D *structureInfo,
                 int primaryPhase)
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

    int *g = (int *)malloc(sizeof(int) * nElements);

#pragma omp parallel
    {
        // local DT, s, and t for each column scan (fixed row)
        int *s = (int *)malloc(sizeof(int) * width);
        int *t = (int *)malloc(sizeof(int) * width);
        memset(s, 0, sizeof(int) * width);
        memset(t, 0, sizeof(int) * width);
// phase 1
#pragma omp for schedule(auto)
        for (int j = 0; j < width; j++)
        {
            int offset = j;
            pass12_2D(targetArray, g, height, width, offset, primaryPhase);
        }
// phase 2
#pragma omp for schedule(auto)
        for (int row = 0; row < height; row++)
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

void pMeijster3D_debug(bool *targetArray,
                       float *targetEDT,
                       sizeInfo *structureInfo,
                       int primaryPhase)
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
        int *s = (int *)malloc(sizeof(int) * width * height);
        int *t = (int *)malloc(sizeof(int) * width * height);
        memset(s, 0, sizeof(int) * width * height);
        memset(t, 0, sizeof(int) * width * height);

        int LL = (height > width) ? height : width;
        LL = (depth > LL) ? depth : LL;

        float *tempEDT = (float *)malloc(sizeof(float) * LL);

        // phase 1

        for (int j = 0; j < width; j++)
        {
#pragma omp for schedule(auto)
            for (int k = 0; k < depth; k++)
            {
                pass12_Global(targetArray, targetEDT, j, k, width, height, depth, primaryPhase);
            }
        }

        // phase 2

        for (int k = 0; k < depth; k++)
        { // all slices
#pragma omp for schedule(auto)
            for (int i = 0; i < height; i++)
            { // all rows
                size_t offset = k * width * height + i * width;
                pass34_Global(targetEDT + offset, tempEDT, width, 1, s, t);
            }
        }

        // maybe not necessary, but clean arrays anyways
        memset(s, 0, sizeof(int) * width * height);
        memset(t, 0, sizeof(int) * width * height);

        for (int j = 0; j < width; j++)
        {
#pragma omp for schedule(auto)
            for (int i = 0; i < height; i++)
            {
                size_t offset = i * width + j;
                pass34_Global(targetEDT + offset, tempEDT, depth, height * width, s, t);
            }
        }
        free(s);
        free(t);
    }

    return;
}

void ParticleLabel2D(int rMin,
                     int rMax,
                     int *R,
                     int *L,
                     sizeInfo2D *structureInfo)
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

    while (r <= rMax) // main loop
    {
        for (int i = 0; i < nElements; i++)
        {
            if (R[i] != r || L[i] != -1)
                continue;

            // Label L[i] accordingly and add it to scan list

            L[i] = particleLabel;

            oList.push_back(i);
            while (!oList.empty()) // Flood-Fill search starting from this label alone
            {
                // pop first index on the list and erase it
                long int index = *oList.begin();
                oList.erase(oList.begin());

                // decode index

                myRow = index / width;
                myCol = index - myRow * width;

                // Search Neighbords with the same r

                if (myRow != 0)
                {
                    temp_index = index - width;
                    if (L[temp_index] == -1 && R[temp_index] == r)
                    {
                        oList.push_back(temp_index);
                        L[temp_index] = particleLabel;
                    }
                }

                if (myRow != height - 1)
                {
                    temp_index = index + width;
                    if (L[temp_index] == -1 && R[temp_index] == r)
                    {
                        oList.push_back(temp_index);
                        L[temp_index] = particleLabel;
                    }
                }

                if (myCol != 0)
                {
                    temp_index = index - 1;
                    if (L[temp_index] == -1 && R[temp_index] == r)
                    {
                        oList.push_back(temp_index);
                        L[temp_index] = particleLabel;
                    }
                }

                if (myCol != width - 1)
                {
                    temp_index = index + 1;
                    if (L[temp_index] == -1 && R[temp_index] == r)
                    {
                        oList.push_back(temp_index);
                        L[temp_index] = particleLabel;
                    }
                }

            } // end inner while
            particleLabel++; // push label increment
        } // end for
        r++; // increase radius label
    } // end while

    return;
}

void ParticleLabel3D(int rMin,
                     int rMax,
                     int *R,
                     int *L,
                     sizeInfo *structureInfo)
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

    while (r <= rMax) // main loop
    {
        for (int i = 0; i < nElements; i++)
        {
            if (R[i] != r || L[i] != -1)
                continue;

            // Label L[i] accordingly and add it to scan list

            L[i] = particleLabel;

            oList.push_back(i);
            while (!oList.empty()) // Flood-Fill search starting from this label alone
            {
                // pop first index on the list and erase it
                long int index = *oList.begin();
                oList.erase(oList.begin());

                // decode index

                mySlice = index / (height * width);
                myRow = (index - mySlice * height * width) / width;
                myCol = (index - mySlice * height * width - myRow * width);

                // Search Neighbords with the same r

                if (mySlice != 0)
                {
                    temp_index = index - height * width;
                    if (L[temp_index] == -1 && R[temp_index] == r)
                    {
                        oList.push_back(temp_index);
                        L[temp_index] = particleLabel;
                    }
                }

                if (mySlice != depth - 1)
                {
                    temp_index = index + height * width;
                    if (L[temp_index] == -1 && R[temp_index] == r)
                    {
                        oList.push_back(temp_index);
                        L[temp_index] = particleLabel;
                    }
                }

                if (myRow != 0)
                {
                    temp_index = index - width;
                    if (L[temp_index] == -1 && R[temp_index] == r)
                    {
                        oList.push_back(temp_index);
                        L[temp_index] = particleLabel;
                    }
                }

                if (myRow != height - 1)
                {
                    temp_index = index + width;
                    if (L[temp_index] == -1 && R[temp_index] == r)
                    {
                        oList.push_back(temp_index);
                        L[temp_index] = particleLabel;
                    }
                }

                if (myCol != 0)
                {
                    temp_index = index - 1;
                    if (L[temp_index] == -1 && R[temp_index] == r)
                    {
                        oList.push_back(temp_index);
                        L[temp_index] = particleLabel;
                    }
                }

                if (myCol != width - 1)
                {
                    temp_index = index + 1;
                    if (L[temp_index] == -1 && R[temp_index] == r)
                    {
                        oList.push_back(temp_index);
                        L[temp_index] = particleLabel;
                    }
                }

            } // end inner while
            particleLabel++; // push label increment
        } // end for
        r++; // increase radius label
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
            - char phase of interest
    */

    if (opts->verbose)
        printf("Particle-Size Distribution 2D:\n");

    // Loop variables

    long int p_sum, d_sum, e_sum;

    e_sum = 1; // have to initialize, otherwise loop won't start

    // array for saving p, e, and d at different R's

    // Index 0 = p, index 1 = d, index 2 = e

    long int *PDE_sum = (long int *)malloc(sizeof(long int) * opts->maxR * 3);

    memset(PDE_sum, 0, sizeof(long int) * opts->maxR * 3);

    // Array for storing radii

    int *R;
    int *L;

    if (opts->partLabel)
    {
        R = (int *)malloc(sizeof(int) * info->nElements);
        L = (int *)malloc(sizeof(int) * info->nElements);

        for (int i = 0; i < info->nElements; i++)
        {
            R[i] = -1;
            L[i] = -1;
        }
    }

    // arrays for holding the EDM:

    /*
        Condsider making those long int's for large domains
    */

    int *EDT_D = (int *)malloc(sizeof(int) * info->nElements);
    int *EDT_E = (int *)malloc(sizeof(int) * info->nElements);

    memset(EDT_D, 0, sizeof(int) * info->nElements);
    memset(EDT_E, 0, sizeof(int) * info->nElements);

    // arrays for erosion and dilation

    bool *E = (bool *)malloc(sizeof(bool) * info->nElements);
    bool *D = (bool *)malloc(sizeof(bool) * info->nElements);
    bool *B = (bool *)malloc(sizeof(bool) * info->nElements);

    memset(E, 0, sizeof(bool) * info->nElements);
    memset(D, 0, sizeof(bool) * info->nElements);
    memset(B, 0, sizeof(bool) * info->nElements);

// If POI, bool = 1
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < info->nElements; i++)
    {
        if (P[i] == POI)
            B[i] = 1;
    }

    // EDT for dilation is a one time operation

    pMeijster2D(B, EDT_D, info, 0); // 0 is the phase that will be dilated

    int radius = 1;

    while (e_sum != 0 && radius <= opts->maxR)
    {
        // copy P into D (probably not necessary)

        memcpy(D, B, sizeof(bool) * info->nElements);
#pragma omp parallel for schedule(auto)
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
#pragma omp parallel for schedule(auto)
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
            p_sum += B[i];
            d_sum += D[i];
            e_sum += E[i];

            if (!opts->partLabel)
                continue;

            if (P[i] - E[i] == 1 && R[i] == -1)
                R[i] = radius;
        }

        PDE_sum[(radius - 1) * 3 + 0] = p_sum;
        PDE_sum[(radius - 1) * 3 + 1] = d_sum;
        PDE_sum[(radius - 1) * 3 + 2] = e_sum;

        // print verbose

        if (opts->verbose)
            printf("R = %d, P = %ld, E = %ld, D = %ld\n", radius, p_sum, e_sum, d_sum);

        // increment radius
        radius++;
    }

    int lastR = radius;

    // calculate partSD and print to output file

    long int sum_removed = 0;
    double *partRemoved = (double *)malloc(sizeof(double) * lastR);

    // get particles removed at R = 1

    partRemoved[0] = PDE_sum[0 * 3 + 0] - PDE_sum[0 * 3 + 2];
    sum_removed += (int)partRemoved[0];

    for (int i = 1; i < lastR; i++)
    {
        partRemoved[i] = PDE_sum[(i - 1) * 3 + 2] - PDE_sum[i * 3 + 2];
        sum_removed += (int)partRemoved[i];
    }

    FILE *partSD_OUT = fopen(opts->partSD_Out, "w+");

    fprintf(partSD_OUT, "r,p(r)\n");
    for (int i = 0; i < lastR; i++)
    {
        fprintf(partSD_OUT, "%d,%lf\n", i + 1, (double)partRemoved[i] / sum_removed);
    }

    fclose(partSD_OUT);

    // partial memory management

    free(partRemoved);
    free(PDE_sum);

    // Derive particle labels from R, if applicable

    if (opts->partLabel)
    {
        ParticleLabel2D(opts->radOff, lastR, R, L, info);
        saveLabels2D(R, L, info, opts->partLabel_Out);
        free(R);
        free(L);
    }

    // memory management

    free(EDT_D);
    free(EDT_E);

    free(B);
    free(E);
    free(D);

    return 0;
}

int poreSD_2D(options *opts,
              sizeInfo2D *info,
              char *P,
              char POI)
{
    /*
        Function poreSD_2D:
        Inputs:
            - pointer to options struct
            - pointer to structure info struct
            - pointer to phase-array
            - char phase of interest
    */

    if (opts->verbose)
        printf("Pore-Size Distribution 2D:\n");

    // Loop variables

    long int p_sum, d_sum, e_sum;

    e_sum = 1; // have to initialize, otherwise loop won't start

    // array for saving p, e, and d at different R's

    // Index 0 = p, index 1 = d, index 2 = e

    long int *PDE_sum = (long int *)malloc(sizeof(long int) * opts->maxR * 3);

    memset(PDE_sum, 0, sizeof(long int) * opts->maxR * 3);

    // Array for storing radii

    int *R;
    int *L;

    if (opts->partLabel)
    {
        R = (int *)malloc(sizeof(int) * info->nElements);
        L = (int *)malloc(sizeof(int) * info->nElements);

        for (int i = 0; i < info->nElements; i++)
        {
            R[i] = -1;
            L[i] = -1;
        }
    }

    // arrays for holding the EDM:

    int *EDT_D = (int *)malloc(sizeof(int) * info->nElements);
    int *EDT_E = (int *)malloc(sizeof(int) * info->nElements);

    memset(EDT_D, 0, sizeof(int) * info->nElements);
    memset(EDT_E, 0, sizeof(int) * info->nElements);

    // arrays for erosion and dilation

    bool *E = (bool *)malloc(sizeof(bool) * info->nElements);
    bool *D = (bool *)malloc(sizeof(bool) * info->nElements);
    bool *B = (bool *)malloc(sizeof(bool) * info->nElements);

    memset(E, 0, sizeof(bool) * info->nElements);
    memset(D, 0, sizeof(bool) * info->nElements);
    memset(B, 0, sizeof(bool) * info->nElements);

// If POI, bool = 1
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < info->nElements; i++)
    {
        if (P[i] == POI)
            B[i] = 1;
    }

    // EDT for dilation is a one time operation

    pMeijster2D(B, EDT_D, info, 0); // 0 is the phase that will be dilated

    int radius = 1;

    while (e_sum != 0 && radius < opts->maxR)
    {
        // copy P into D (probably not necessary)

        memcpy(D, B, sizeof(bool) * info->nElements);
#pragma omp parallel for schedule(auto)
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
#pragma omp parallel for schedule(auto)
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
            p_sum += B[i];
            d_sum += D[i];
            e_sum += E[i];

            if (!opts->poreLabel)
                continue;

            if (B[i] - E[i] == 1 && R[i] == -1)
                R[i] = radius;
        }

        PDE_sum[(radius - 1) * 3 + 0] = p_sum;
        PDE_sum[(radius - 1) * 3 + 1] = d_sum;
        PDE_sum[(radius - 1) * 3 + 2] = e_sum;

        // print to output file

        if (opts->verbose)
            printf("R = %d, P = %ld, E = %ld, D = %ld\n", radius, p_sum, e_sum, d_sum);
        // increment radius
        radius++;
    }

    int lastR = radius;

    // calculate partSD and print to output file

    long int sum_removed = 0;
    double *poreRemoved = (double *)malloc(sizeof(double) * lastR);

    // get particles removed at R = 1

    poreRemoved[0] = PDE_sum[0 * 3 + 0] - PDE_sum[0 * 3 + 2];
    sum_removed += (int)poreRemoved[0];

    for (int i = 1; i < lastR; i++)
    {
        poreRemoved[i] = PDE_sum[(i - 1) * 3 + 2] - PDE_sum[i * 3 + 2];
        sum_removed += (int)poreRemoved[i];
    }

    FILE *poreSD_OUT = fopen(opts->poreSD_Out, "w+");

    fprintf(poreSD_OUT, "r,p(r)\n");
    for (int i = 0; i < lastR; i++)
    {
        fprintf(poreSD_OUT, "%d,%lf\n", i + 1, (double)poreRemoved[i] / sum_removed);
    }

    fclose(poreSD_OUT);

    // partial memory management

    free(poreRemoved);
    free(PDE_sum);

    // Derive particle labels from R, if applicable

    if (opts->poreLabel)
    {
        ParticleLabel2D(opts->radOff, lastR, R, L, info);
        saveLabels2D(R, L, info, opts->poreLabel_Out);
        free(R);
        free(L);
    }

    // memory management

    free(EDT_D);
    free(EDT_E);

    free(B);
    free(E);
    free(D);

    return 0;
}

int partSD_3D(options *opts,
              sizeInfo *info,
              char *P,
              int POI)
{
    /*
        Function poreSD_3D:
        Inputs:
            - pointer to options struct
            - pointer to structure info struct
            - pointer to phase-array
            - char phase of interest
    */

    if (opts->verbose)
        printf("Particle-Size Distribution 3D:\n");

    // Loop variables

    long int p_sum, d_sum, e_sum;

    e_sum = 1; // have to initialize, otherwise loop won't start

    // array for saving p, e, and d at different R's

    // Index 0 = p, index 1 = d, index 2 = e

    long int *PDE_sum = (long int *)malloc(sizeof(long int) * opts->maxR * 3);

    memset(PDE_sum, 0, sizeof(long int) * opts->maxR * 3);

    // Array for storing radii

    int *R;
    int *L;

    if (opts->partLabel)
    {
        R = (int *)malloc(sizeof(int) * info->nElements);
        L = (int *)malloc(sizeof(int) * info->nElements);

        for (int i = 0; i < info->nElements; i++)
        {
            R[i] = -1;
            L[i] = -1;
        }
    }

    // arrays for holding the EDM:

    float *EDT_D = (float *)malloc(sizeof(float) * info->nElements);
    float *EDT_E = (float *)malloc(sizeof(float) * info->nElements);

    memset(EDT_D, 0, sizeof(float) * info->nElements);
    memset(EDT_E, 0, sizeof(float) * info->nElements);

    // arrays for erosion and dilation

    bool *E = (bool *)malloc(sizeof(bool) * info->nElements);
    bool *D = (bool *)malloc(sizeof(bool) * info->nElements);
    bool *B = (bool *)malloc(sizeof(bool) * info->nElements);

    memset(E, 0, sizeof(bool) * info->nElements);
    memset(D, 0, sizeof(bool) * info->nElements);
    memset(B, 0, sizeof(bool) * info->nElements);

// If POI, bool = 1
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < info->nElements; i++)
    {
        if (P[i] == POI)
            B[i] = 1;
    }

    // EDT for dilation is a one time operation

    pMeijster3D_debug(B, EDT_D, info, 0); // 0 is the phase that will be dilated

    int radius = 1;

    // Main Loop

    while (e_sum != 0 && radius <= opts->maxR)
    {
        // copy P into D (probably not necessary)

        memcpy(D, B, sizeof(bool) * info->nElements);
#pragma omp parallel for schedule(auto)
        for (int i = 0; i < info->nElements; i++)
        {
            if (pow(EDT_D[i], 2) <= (float)radius * radius)
                D[i] = 0;
        }

        // Copy D into E

        memcpy(E, D, sizeof(bool) * info->nElements);

        // Meijster in D

        pMeijster3D_debug(D, EDT_E, info, 1);

// Update E
#pragma omp parallel for schedule(auto)
        for (int i = 0; i < info->nElements; i++)
        {
            if (pow(EDT_E[i], 2) <= (float)radius * radius)
                E[i] = 1;
        }

        // Quantify changes

        e_sum = 0;
        d_sum = 0;
        p_sum = 0;

#pragma omp parallel for reduction(+ : p_sum, d_sum, e_sum)
        for (int i = 0; i < info->nElements; i++)
        {
            p_sum += B[i];
            d_sum += D[i];
            e_sum += E[i];

            if (!opts->partLabel)
                continue;

            if (B[i] - E[i] == 1 && R[i] == -1)
                R[i] = radius;
        }

        PDE_sum[(radius - 1) * 3 + 0] = p_sum;
        PDE_sum[(radius - 1) * 3 + 1] = d_sum;
        PDE_sum[(radius - 1) * 3 + 2] = e_sum;

        // print verbose

        if (opts->verbose)
            printf("R = %d, P = %ld, E = %ld, D = %ld\n", radius, p_sum, e_sum, d_sum);

        // increment radius
        radius++;
    }

    int lastR = radius;

    // calculate partSD and print to output file

    // calculate partSD and print to output file

    long int sum_removed = 0;
    double *partRemoved = (double *)malloc(sizeof(double) * lastR);

    // get particles removed at R = 1

    partRemoved[0] = PDE_sum[0 * 3 + 0] - PDE_sum[0 * 3 + 2];
    sum_removed += (int)partRemoved[0];

    for (int i = 1; i < lastR; i++)
    {
        partRemoved[i] = PDE_sum[(i - 1) * 3 + 2] - PDE_sum[i * 3 + 2];
        sum_removed += (int)partRemoved[i];
    }

    FILE *partSD_OUT = fopen(opts->partSD_Out, "w+");

    fprintf(partSD_OUT, "r,p(r)\n");
    for (int i = 0; i < lastR; i++)
    {
        fprintf(partSD_OUT, "%d,%lf\n", i + 1, (double)partRemoved[i] / sum_removed);
    }

    fclose(partSD_OUT);

    // partial memory management

    free(partRemoved);
    free(PDE_sum);

    // Derive particle labels from R, if applicable

    if (opts->partLabel)
    {
        ParticleLabel3D(opts->radOff, lastR, R, L, info);
        saveLabels3D(R, L, info, opts->partLabel_Out);
        free(R);
        free(L);
    }

    // memory management

    free(EDT_D);
    free(EDT_E);

    free(B);
    free(E);
    free(D);

    return 0;
}

int poreSD_3D(options *opts,
              sizeInfo *info,
              char *P,
              int POI)
{
    /*
        Function poreSD_3D:
        Inputs:
            - pointer to options struct
            - pointer to structure info struct
            - pointer to phase-array
            - char phase of interest
    */

    if (opts->verbose)
        printf("Pore-Size Distribution 3D:\n");

    // Loop variables

    long int p_sum, d_sum, e_sum;

    e_sum = 1; // have to initialize, otherwise loop won't start

    // array for saving p, e, and d at different R's

    // Index 0 = p, index 1 = d, index 2 = e

    long int *PDE_sum = (long int *)malloc(sizeof(long int) * opts->maxR * 3);

    memset(PDE_sum, 0, sizeof(long int) * opts->maxR * 3);

    // Array for storing radii

    int *R;
    int *L;

    if (opts->partLabel)
    {
        R = (int *)malloc(sizeof(int) * info->nElements);
        L = (int *)malloc(sizeof(int) * info->nElements);

        for (int i = 0; i < info->nElements; i++)
        {
            R[i] = -1;
            L[i] = -1;
        }
    }

    // arrays for holding the EDM:

    float *EDT_D = (float *)malloc(sizeof(float) * info->nElements);
    float *EDT_E = (float *)malloc(sizeof(float) * info->nElements);

    memset(EDT_D, 0, sizeof(float) * info->nElements);
    memset(EDT_E, 0, sizeof(float) * info->nElements);

    // arrays for erosion and dilation

    bool *E = (bool *)malloc(sizeof(bool) * info->nElements);
    bool *D = (bool *)malloc(sizeof(bool) * info->nElements);
    bool *B = (bool *)malloc(sizeof(bool) * info->nElements);

    memset(E, 0, sizeof(bool) * info->nElements);
    memset(D, 0, sizeof(bool) * info->nElements);
    memset(B, 0, sizeof(bool) * info->nElements);

// If POI, bool = 1
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < info->nElements; i++)
    {
        if (P[i] == POI)
            B[i] = 1;
    }

    // EDT for dilation is a one time operation

    pMeijster3D_debug(B, EDT_D, info, 0); // 0 is the phase that will be dilated

    int radius = 1;

    // Main Loop

    while (e_sum != 0 && radius <= opts->maxR)
    {
        // copy B into D (probably not necessary)

        memcpy(D, B, sizeof(bool) * info->nElements);
#pragma omp parallel for schedule(auto)
        for (int i = 0; i < info->nElements; i++)
        {
            if (pow(EDT_D[i], 2) <= (float)radius * radius)
                D[i] = 0;
        }

        // Copy D into E

        memcpy(E, D, sizeof(bool) * info->nElements);

        // Meijster in D

        pMeijster3D_debug(D, EDT_E, info, 1);

// Update E
#pragma omp parallel for schedule(auto)
        for (int i = 0; i < info->nElements; i++)
        {
            if (pow(EDT_E[i], 2) <= (float)radius * radius)
                E[i] = 1;
        }

        // Quantify changes

        e_sum = 0;
        d_sum = 0;
        p_sum = 0;

#pragma omp parallel for reduction(+ : p_sum, d_sum, e_sum)
        for (int i = 0; i < info->nElements; i++)
        {
            p_sum += B[i];
            d_sum += D[i];
            e_sum += E[i];

            if (!opts->poreLabel)
                continue;

            if (B[i] - E[i] == 1 && R[i] == -1)
                R[i] = radius;
        }

        PDE_sum[(radius - 1) * 3 + 0] = p_sum;
        PDE_sum[(radius - 1) * 3 + 1] = d_sum;
        PDE_sum[(radius - 1) * 3 + 2] = e_sum;

        // print verbose

        if (opts->verbose)
            printf("R = %d, P = %ld, E = %ld, D = %ld\n", radius, p_sum, e_sum, d_sum);

        // increment radius
        radius++;
    }

    int lastR = radius;

    // calculate partSD and print to output file

    long int sum_removed = 0;
    double *partRemoved = (double *)malloc(sizeof(double) * lastR);

    // get particles removed at R = 1

    partRemoved[0] = PDE_sum[0 * 3 + 0] - PDE_sum[0 * 3 + 2];
    sum_removed += (int)partRemoved[0];

    for (int i = 1; i < lastR; i++)
    {
        partRemoved[i] = PDE_sum[(i - 1) * 3 + 2] - PDE_sum[i * 3 + 2];
        sum_removed += (int)partRemoved[i];
    }

    FILE *partSD_OUT = fopen(opts->poreSD_Out, "w+");

    fprintf(partSD_OUT, "r,p(r)\n");
    for (int i = 0; i < lastR; i++)
    {
        fprintf(partSD_OUT, "%d,%lf\n", i + 1, (double)partRemoved[i] / sum_removed);
    }

    fclose(partSD_OUT);

    // partial memory management

    free(partRemoved);
    free(PDE_sum);

    // Derive particle labels from R, if applicable

    if (opts->poreLabel)
    {
        ParticleLabel3D(opts->radOff, lastR, R, L, info);
        saveLabels3D(R, L, info, opts->poreLabel_Out);

        free(R);
        free(L);
    }

    // memory management

    free(EDT_D);
    free(EDT_E);

    free(B);
    free(E);
    free(D);

    return 0;
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

    ------------------------------------------------------------------------*/

    // declare structure related variables

    unsigned char *target_img;
    sizeInfo2D imgInfo;

    // set omp options

    omp_set_num_threads(opts->nThreads);

    // read structure

    if (opts->inputType == 0)
    {
        if (readImg_2D(opts->inputFilename, &target_img, &imgInfo) == 1)
            printf("Error, image has wrong number of channels\n");
    }
    else
    {
        printf("Method not implemented yet!\n");
    }

    // Create array to hold structure

    char *P = (char *)malloc(sizeof(char) * imgInfo.nElements);

    memset(P, 0, sizeof(char) * imgInfo.nElements);

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

    if (opts->partSD)
        partSD_2D(opts, &imgInfo, P, 1);

    if (opts->poreSD)
        poreSD_2D(opts, &imgInfo, P, 0);

    // Memory Management
    free(P);

    return 0;
}

int Sim3D(options *opts)
{
    /*---------------------------------------------------------------------

                            Read Input
                                &
                          Declare Arrays

        Input mode flags:
        - Flag = 0 means .csv file with x,y,z coordinates of the particles.
        - Flag = 1 means stack of .jpg files
    ------------------------------------------------------------------------*/

    // Structs

    sizeInfo info;

    // define P

    char *P;

    // read by input type

    if (opts->inputType == 0)
    {
        // Expected structure size

        info.width = opts->width;
        info.height = opts->height;
        info.depth = opts->depth;
        info.nElements = info.depth * info.width * info.height;

        // set omp options

        omp_set_num_threads(opts->nThreads);

        // Declare and Define Phase array

        P = (char *)malloc(sizeof(char) * info.nElements);

        memset(P, 0, sizeof(char) * info.nElements);
        // read csv
        readCSV(opts->inputFilename, P, &info);
    }
    else if (opts->inputType == 1)
    {
        // read one image to get size
        info.depth = opts->stackSize;

        unsigned char *target_practice;

        char test_name[100];

        sprintf(test_name, "%0*d.jpg", opts->LeadZero, 0);

        int channel;

        target_practice = stbi_load(test_name, &info.width, &info.height, &channel, 1);

        free(target_practice);
        // set size of P

        info.nElements = info.width * info.height * info.depth;

        P = (char *)malloc(sizeof(char) * info.nElements);

        memset(P, 0, sizeof(char) * info.nElements);

        // read stack onto P

        readStack(P, opts);
    }

    // Perform selected simulations

    if (opts->partSD)
        partSD_3D(opts, &info, P, 1);

    if (opts->poreSD)
        poreSD_3D(opts, &info, P, 0);

    free(P);

    return 0;
}