/*

    CPU-based Dilation-Erosion algorithm for phase-size distribution.

    Last Change:
    09/17/2024

    Andre Adam.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>

// Define constants

#define MAX_R 65


int main(void){

    /*--------------------------------------------------------
    
            Declare and define useful variables

    --------------------------------------------------------*/
    // OMP options
    int num_threads = 8;
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

    width = 200;
    height = 200;
    depth = 200;

    int max_size = width*height*depth;

    // P holds the structure. E and D are essential for the operations

    char *P = (char *)malloc(sizeof(char)*max_size);
    char *D = (char *)malloc(sizeof(char)*max_size);
    char *E = (char *)malloc(sizeof(char)*max_size);

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

    while(e_sum != 0)
    {
        
        /*--------------------------------------------------------
    
        Step 3: Dilate Interfaces by R

        --------------------------------------------------------*/

        // Copy P into dilation array

        memcpy(D, P, sizeof(char)*max_size);

        // search entire array

        // int row, col, slice;
        // bool interfaceFlag;
        // long int temp_index;
        // #pragma omp private(row, col, slice, interfaceFlag, temp_index)
        #pragma omp parallel for schedule(auto)
        for(long int i = 0; i<max_size; i++)
        {
            int row, col, slice;
            bool interfaceFlag;
            long int temp_index;
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

                // it is an interface, so we must scan a radius around it

                for(int rk = slice - radius; rk <= slice + radius; rk++){
                    if(rk < 0 || rk > depth - 1) continue;
                    for(int ri = row - radius; ri <= row + radius; ri++){
                        if(ri < 0 || ri > height - 1) continue;
                        for(int rj = col - radius; rj <= col + radius; rj++){
                            if(rj < 0 || rj > width - 1) continue;

                            if(pow(rk-slice,2) + pow(rj-col,2) + pow(ri - row,2) <= pow(radius, 2)) D[rk*height*width + ri*width + rj] = 0;
                        }
                    }
                }
            }
        }

        /*--------------------------------------------------------
    
        Step 4: Erode Interfaces by R

        --------------------------------------------------------*/
        // Copy D into E

        memcpy(E, D, sizeof(char)*max_size);

        // Erosion
        #pragma omp parallel for schedule(auto)
        for(long int i = 0; i<max_size; i++)
        {
            int row, col, slice;
            bool interfaceFlag;
            long int temp_index;
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
        radius++;
    }

    fclose(OUT);


    return 0;
}