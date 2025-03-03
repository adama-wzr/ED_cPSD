#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QThread>
#include "ED_PSD_CPU.hpp"

/*

Worker Thread Class Implementation

*/

// constructor

Worker::Worker(QObject *parent)
    : QThread(parent)
{
}

// destructor

Worker::~Worker()
{
    mutex.lock();
    abort = true;
    condition.wakeOne();
    mutex.unlock();
    wait();
}

// runSim function

void Worker::runSim(const QString string)
{
    QMutexLocker locker(&mutex);
    this->foldername = string;
    if(!isRunning())
        start(LowPriority);
    else
    {
        restart = true;
        condition.wakeOne();
    }
    return;
}

int Worker::Sim2D_ui(options *opts)
{
    // Re-Implemented for UI communication purposes
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
        if (readImg_2D(opts->inputFilename, &target_img, &imgInfo, opts) == 1)
        {
            QString imgError = "Error, image has wrong number of channels";
            emit resultReady(&imgError);
            msleep(250);
            return 1;
        }
    }
    else
    {
        QString imgError = "Method not implemented yet! Please enter valid input type.";
        emit resultReady(&imgError);
        msleep(250);
        return 1;
    }

    if (restart)
    {
        QString restartMsg = "Stop button pressed, simulation interrupted.";
        emit resultReady(&restartMsg);
        msleep(250);
        return 1;
    }

    if (abort)
        return 1;

    // Create array to hold structure

    char *P = (char *)malloc(sizeof(char) * imgInfo.nElements);

    memset(P, 0, sizeof(char) * imgInfo.nElements);

    // Cast image into P, free original array

    size_t sCount = 0;

    for (int i = 0; i < imgInfo.nElements; i++)
    {
        if (target_img[i] < opts->TH)
            P[i] = 0;
        else
        {
            sCount++;
            P[i] = 1;
        }
    }

    free(target_img);

    // Perform the selected simulations

    int executionFlag = 0;

    if (opts->partSD)
    {
        QString divider = "\n---------------------------------------------";
        emit resultReady(&divider);
        msleep(250);
        executionFlag = part2D_SD_ui(opts, &imgInfo, P, 1);
    }



    if(executionFlag == 1)
    {
        free(P);
        return 1;
    }

    if (opts->poreSD)
    {
        QString divider = "\n---------------------------------------------";
        emit resultReady(&divider);
        msleep(250);
        executionFlag = pore2D_SD_ui(opts, &imgInfo, P, 0);
    }

    if(executionFlag == 1)
    {
        free(P);
        return 1;
    }

    // Memory management
    free(P);
    // Check to see if any buttons were pressed or simulation was aborted already
    return 0;
}

int Worker::Sim3D_ui(options *opts)
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
        readCSV(opts->inputFilename, P, &info, opts);
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

    if (restart)
    {
        QString restartMsg = "Stop button pressed, simulation interrupted.";
        emit resultReady(&restartMsg);
        msleep(250);
        free(P);
        return 1;
    }

    if (abort)
        return 1;

    // Perform selected simulations

    int executionFlag = 0;

    if (opts->partSD)
    {
        QString divider = "\n---------------------------------------------";
        emit resultReady(&divider);
        msleep(250);
        executionFlag = part3D_SD_ui(opts, &info, P, 1);
    }

    if(executionFlag == 1)
    {
        free(P);
        return 1;
    }


    if (opts->poreSD)
    {
        QString divider = "\n---------------------------------------------";
        emit resultReady(&divider);
        msleep(250);
        executionFlag = pore3D_SD_ui(opts, &info, P, 0);
    }

    if(executionFlag == 1)
    {
        free(P);
        return 1;
    }

    // memory management
    free(P);
    return 0;
}

int Worker::pore2D_SD_ui(options *opts, sizeInfo2D *info, char *P, char POI)
{
    /*
        Function poreSD_2D_ui:
        Inputs:
            - pointer to options struct
            - pointer to structure info struct
            - pointer to phase-array
            - char phase of interest
    */

    if (opts->verbose)
    {
        QString pore2D_runMsg = "Pore-Size Distribution 2D:\n";
        emit resultReady(&pore2D_runMsg);
        msleep(250);
    }

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

    if (opts->poreLabel)
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
        if (restart)
        {
            QString restartMsg = "Stop button pressed, simulation interrupted.";
            emit resultReady(&restartMsg);
            msleep(250);

            free(PDE_sum);
            free(EDT_D);
            free(EDT_E);

            free(E);
            free(B);
            free(D);

            if(opts->poreLabel)
            {
                free(L);
                free(R);
            }

            return 1;
        }
        // copy P into D (probably not necessary)

        memcpy(D, B, sizeof(bool) * info->nElements);

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

        // if (opts->verbose)
        //     qInfo("R = %d, P = %ld, E = %ld, D = %ld\n", radius, p_sum, e_sum, d_sum);
        if (opts->verbose)
        {
            QString pore2D_runMsg = QString().asprintf("R = %d, P = %ld, E = %ld, D = %ld", radius, p_sum, e_sum, d_sum);
            emit resultReady(&pore2D_runMsg);
            msleep(250);
        }
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

    std::filesystem::path dir (opts->folderName);
    std::filesystem::path file (opts->poreSD_Out);
    std::filesystem::path full_path = dir / file;

    FILE *poreSD_OUT = fopen(full_path.generic_string().c_str(), "w+");

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
        saveLabels2D(R, L, info, opts->poreLabel_Out, opts);
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

int Worker::part2D_SD_ui(options *opts, sizeInfo2D *info, char *P, char POI)
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
    {
        QString part2D_runMsg = "Particle-Size Distribution 2D:\n";
        emit resultReady(&part2D_runMsg);
        msleep(250);
    }

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
        if (restart)
        {
            QString restartMsg = "Stop button pressed, simulation interrupted.";
            emit resultReady(&restartMsg);
            msleep(250);

            free(PDE_sum);
            free(EDT_D);
            free(EDT_E);

            free(E);
            free(B);
            free(D);

            if(opts->partLabel)
            {
                free(L);
                free(R);
            }

            return 1;
        }
        // copy P into D (probably not necessary)

        memcpy(D, B, sizeof(bool) * info->nElements);

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
        {
            QString part2D_runMsg = QString().asprintf("R = %d, P = %ld, E = %ld, D = %ld", radius, p_sum, e_sum, d_sum);
            emit resultReady(&part2D_runMsg);
            msleep(250);
        }

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

    std::filesystem::path dir (opts->folderName);
    std::filesystem::path file (opts->partSD_Out);
    std::filesystem::path full_path = dir / file;



    FILE *partSD_OUT = fopen(full_path.generic_string().c_str(), "w+");

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
        saveLabels2D(R, L, info, opts->partLabel_Out, opts);
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

int Worker::pore3D_SD_ui(options *opts, sizeInfo *info, char *P, int POI)
{
    /*
        Function poreSD_3D:
        Inputs:
            - pointer to options struct
            - pointer to structure info struct
            - pointer to phase-array
            - char phase of interest
        Outputs:
            - None.
        Function will calculate pore size distribution of array P.
    */

    if (opts->verbose)
    {
        QString part3D_runMsg = "Pore-Size Distribution 3D:\n";
        emit resultReady(&part3D_runMsg);
        msleep(250);
    }

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

    if (opts->poreLabel)
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

    int radius;

    if(opts->radOff == 0)
    {
        radius = 1;
    } else
    {
        radius = opts->radOff;
    }

    // Main Loop

    while (e_sum != 0 && radius <= opts->maxR)
    {
        if (restart)
        {
            QString restartMsg = "Stop button pressed, simulation interrupted.";
            emit resultReady(&restartMsg);
            msleep(250);

            free(PDE_sum);
            free(EDT_D);
            free(EDT_E);

            free(E);
            free(B);
            free(D);

            if(opts->partLabel)
            {
                free(L);
                free(R);
            }

            return 1;
        }
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

        PDE_sum[(radius - 1 - opts->radOff) * 3 + 0] = p_sum;
        PDE_sum[(radius - 1 - opts->radOff) * 3 + 1] = d_sum;
        PDE_sum[(radius - 1 - opts->radOff) * 3 + 2] = e_sum;

        // print verbose

        if (opts->verbose)
        {
            QString pore3D_runMsg = QString().asprintf("R = %d, P = %ld, E = %ld, D = %ld", radius, p_sum, e_sum, d_sum);
            emit resultReady(&pore3D_runMsg);
            msleep(250);
        }

        // increment radius
        radius++;
    }

    int lastR = radius;

    // calculate partSD and print to output file

    long int sum_removed = 0;
    double *partRemoved = (double *)malloc(sizeof(double) * (lastR - opts->radOff));

    // get particles removed at R = 1

    partRemoved[0] = PDE_sum[0 * 3 + 0] - PDE_sum[0 * 3 + 2];
    sum_removed += (int)partRemoved[0];

    for (int i = 1; i < (lastR - opts->radOff); i++)
    {
        partRemoved[i] = PDE_sum[(i - 1) * 3 + 2] - PDE_sum[i * 3 + 2];
        sum_removed += (int)partRemoved[i];
    }

    // add correction for radius offset

    int correction = 0;
    if (opts->radOff != 0)
        correction = opts->radOff - 1;

    FILE *partSD_OUT = fopen(opts->poreSD_Out, "w+");

    fprintf(partSD_OUT, "r,p(r)\n");
    for (int i = 0; i < (lastR - opts->radOff); i++)
    {
        fprintf(partSD_OUT, "%d,%lf\n", i + 1 + correction, (double)partRemoved[i] / sum_removed);
    }

    fclose(partSD_OUT);

    // partial memory management

    free(partRemoved);
    free(PDE_sum);

    // Derive particle labels from R, if applicable

    if (opts->poreLabel)
    {
        ParticleLabel3D(opts->radOff, lastR, R, L, info);
        saveLabels3D(R, L, info,  opts->poreLabel_Out, opts);

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

int Worker::part3D_SD_ui(options *opts, sizeInfo *info, char *P, int POI)
{
    /*
        Function partSD_3D:
        Inputs:
            - pointer to options struct
            - pointer to structure info struct
            - pointer to phase-array
            - char phase of interest
        Outputs:
            - None.
        Function will calculate particle size distribution of array P.
    */

    if (opts->verbose)
    {
        QString part3D_runMsg = "Particle-Size Distribution 3D:\n";
        emit resultReady(&part3D_runMsg);
        msleep(250);
    }
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

    if(opts->radOff == 0)
    {
        radius = 1;
    } else
    {
        radius = opts->radOff;
    }

    // Main Loop

    while (e_sum != 0 && radius <= opts->maxR)
    {
        if (restart)
        {
            QString restartMsg = "Stop button pressed, simulation interrupted.";
            emit resultReady(&restartMsg);
            msleep(250);

            free(PDE_sum);
            free(EDT_D);
            free(EDT_E);

            free(E);
            free(B);
            free(D);

            if(opts->partLabel)
            {
                free(L);
                free(R);
            }

            return 1;
        }
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

        PDE_sum[(radius - 1 - opts->radOff) * 3 + 0] = p_sum;
        PDE_sum[(radius - 1 - opts->radOff) * 3 + 1] = d_sum;
        PDE_sum[(radius - 1 - opts->radOff) * 3 + 2] = e_sum;

        // print verbose

        if (opts->verbose)
        {
            QString part3D_runMsg = QString().asprintf("R = %d, P = %ld, E = %ld, D = %ld", radius, p_sum, e_sum, d_sum);
            emit resultReady(&part3D_runMsg);
            msleep(250);
        }

        // increment radius
        radius++;
    }

    int lastR = radius;

    // calculate partSD and print to output file

    long int sum_removed = 0;
    double *partRemoved = (double *)malloc(sizeof(double) * (lastR - opts->radOff));

    // get particles removed at R = 1

    partRemoved[0] = PDE_sum[0 * 3 + 0] - PDE_sum[0 * 3 + 2];
    sum_removed += (int)partRemoved[0];

    for (int i = 1; i < (lastR - opts->radOff); i++)
    {
        partRemoved[i] = PDE_sum[(i - 1) * 3 + 2] - PDE_sum[i * 3 + 2];
        sum_removed += (int)partRemoved[i];
    }

    // correction for radius offset
    int correction = 0;
    if (opts->radOff != 0)
        correction = opts->radOff - 1;

    FILE *partSD_OUT = fopen(opts->partSD_Out, "w+");

    fprintf(partSD_OUT, "r,p(r)\n");
    for (int i = 0; i < (lastR - opts->radOff); i++)
    {
        fprintf(partSD_OUT, "%d,%lf\n", i + 1 + correction, (double)partRemoved[i] / sum_removed);
    }

    fclose(partSD_OUT);

    // partial memory management

    free(partRemoved);
    free(PDE_sum);

    // Derive particle labels from R, if applicable

    if (opts->partLabel)
    {
        ParticleLabel3D(opts->radOff, lastR, R, L, info);
        saveLabels3D(R, L, info,  opts->partLabel_Out, opts);
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

// Re-implement run(), which is actually the function that does all of the work

void Worker::run()
{
    QElapsedTimer timer;
    forever
    {
        emit disableButtons();
        options opts;
        opts.folderName = (char *)malloc(sizeof(char)*1000);
        mutex.lock();
        strcpy(opts.folderName, foldername.toStdString().c_str());
        mutex.unlock();

        // now we can start the actual code run
        timer.restart();
        QString started = "Operating Folder:";
        emit resultReady(&started);
        msleep(250);
        QString test = QString(opts.folderName);
        emit resultReady(&test);
        msleep(250);
        // QString test2 = QString().asprintf("%1.3e", 1232845.4);
        // emit resultReady(&test2);
        // msleep(250);

        // first step is to parse informations

        char inputTextFile[100];
        sprintf(inputTextFile, "input.txt");

        bool fileFlag = false;

        std::filesystem::path dir (opts.folderName);
        std::filesystem::path file (inputTextFile);
        std::filesystem::path full_path = dir / file;

        if(FILE *file = fopen(full_path.generic_string().c_str(), "r")){
            fclose(file);
            fileFlag = true;
        }else
            fileFlag = false;

        if (fileFlag == false)
        {
            QString fileError = "Could not locate input file, exiting now.";
            emit resultReady(&fileError);
            msleep(250);
        }
        else
        {
            int executionFlag = 0;
            readInput(inputTextFile, &opts);
            // attempt to run simulation
            QString simRunning = "Running c-PSD Simulation...";
            emit resultReady(&simRunning);
            msleep(250);
            if (opts.nD == 2)
            {
                executionFlag = Sim2D_ui(&opts);
            }else if (opts.nD == 3)
            {
                executionFlag = Sim3D_ui(&opts);
            }
            if (executionFlag == 1)
            {
                QString errorMessage = "Simulation exited with code 1: something stopped execution.";
                emit resultReady(&errorMessage);
                msleep(250);
                emit enableButtons();
                restart = false;
                break;
            }else
            {
                QString doneRunning = "Simulation Finished!";
                emit resultReady(&doneRunning);
                msleep(250);
            }
        }


        // re-enable buttons
        emit enableButtons();
        // abort statement to exit
        if(abort)
            return;

        // work ended
        mutex.lock();
        if(!restart)
            condition.wait(&mutex);
        restart = false;
        mutex.unlock();
    }
}

/*

Main Window Class Implementation

*/

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    // gen button
    connect(ui->GenButton2D,&QPushButton::clicked, this, &MainWindow::updateFileText);
    // gen button 3D
    connect(ui->GenButton3D,&QPushButton::clicked, this, &MainWindow::updateFileText3D);
    // toggle 2D pore/part labels
    connect(ui->partSD,&QCheckBox::checkStateChanged, this, &MainWindow::togglePartLabel2D);
    connect(ui->poreSD,&QCheckBox::checkStateChanged, this, &MainWindow::togglePoreLabel2D);
    // toggle 3D pore/part labels
    connect(ui->partSD_3D, &QCheckBox::checkStateChanged, this, &MainWindow::togglePartLabel3D);
    connect(ui->poreSD_3D, &QCheckBox::checkStateChanged, this, &MainWindow::togglePoreLabel3D);
    // toggle Input Type 3D
    connect(ui->inputType3D, &QComboBox::currentIndexChanged, this, &MainWindow::toggleInput3D);
    // save input 2D
    connect(ui->saveButton2D, &QPushButton::clicked, this, &MainWindow::saveInput2D);
    // save input 3D
    connect(ui->saveButton3D, &QPushButton::clicked, this, &MainWindow::saveInput3D);
    // Find operating folder
    connect(ui->searchFolder, &QPushButton::clicked, this, &MainWindow::findOpFolder);
    // connect clear button 2D
    connect(ui->clearButton2D, &QPushButton::clicked, this, &MainWindow::clearText2D);
    // connect clear button 3D
    connect(ui->clearButton3D, &QPushButton::clicked, this, &MainWindow::clearText3D);
    // connect run button to run function
    connect(ui->runButton, &QPushButton::clicked, this, &MainWindow::runSim);
    // connect clear button Runtime
    connect(ui->clearRunButton, &QPushButton::clicked, this, &MainWindow::clearTextRun);
    // result ready queued connection
    connect(&workerThread, &Worker::resultReady, this, &MainWindow::handleResult);
    // enable buttons
    connect(&workerThread, &Worker::enableButtons, this, &MainWindow::enableButtons);
    // disable buttons
    connect(&workerThread, &Worker::disableButtons, this, &MainWindow::disableButtons);
    // Stop Sim
    connect(ui->stopButton, &QPushButton::clicked, this, &MainWindow::runSim);
}


// print results
void MainWindow::handleResult(const QString *result)
{
    // qInfo() << result;
    ui->runtimeMessages->append(*result);
    return;
}


int simOpts(const QString &string)
{
    options opts;

    opts.folderName = (char *)malloc(sizeof(char)*1000);
    strcpy(opts.folderName, string.toStdString().c_str());

    char inputTextFile[100];
    sprintf(inputTextFile, "input.txt");

    bool fileFlag = false;

    std::filesystem::path dir (opts.folderName);
    std::filesystem::path file (inputTextFile);
    std::filesystem::path full_path = dir / file;

    if(FILE *file = fopen(full_path.generic_string().c_str(), "r")){
        fclose(file);
        fileFlag = true;
    }else
    {
        fileFlag = false;
    }

    if (!fileFlag)
    {
        qInfo("Could not find input file. Exiting now.");
        return 1;
    }

    readInput(inputTextFile, &opts);

    // if(opts.verbose)
    // {
    //     // printOpts(&opts);
    // }

    if(opts.nD == 2)
    {
        Sim2D(&opts);
    }
    else if(opts.nD == 3)
    {
        Sim3D(&opts);
    }

    free(opts.folderName);
    return 0;
}

void MainWindow::disableButtons()
{
    // // disable buttons and text
    ui->runButton->setEnabled(false);
    ui->stopButton->setEnabled(true);

    ui->searchFolder->setEnabled(false);
    ui->opFolder->setReadOnly(true);
    return;
}

void MainWindow::enableButtons()
{
    ui->runButton->setEnabled(true);
    ui->stopButton->setEnabled(false);

    ui->searchFolder->setEnabled(true);
    ui->opFolder->setReadOnly(false);
    return;
}

// Run sim

void MainWindow::runSim()
{
    if (ui->opFolder->text().isEmpty())
    {
        ui->runtimeMessages->append("Error: no operating folder selected!\n");
        return;
    }
    workerThread.runSim(ui->opFolder->text());
    // ui->runtimeMessages->setText("Attempting to run simulation.\n");

    // char* runtimeFolder = (char *)malloc(sizeof(char)*1000);

    // if (ui->opFolder->text().isEmpty())
    // {
    //     ui->runtimeMessages->append("Error: no operating folder selected!\n");
    //     free(runtimeFolder);
    //     return;
    // }else
    // {
    //     strcpy(runtimeFolder, ui->opFolder->text().toStdString().c_str());
    //     ui->runtimeMessages->append("Folder: " + ui->opFolder->text() + "/\n");
    // }

    // ui->runtimeMessages->append("Simulation running, please wait.\n");

    // // disable buttons and text

    // ui->runButton->setEnabled(false);
    // ui->stopButton->setEnabled(true);

    // ui->searchFolder->setEnabled(false);
    // ui->opFolder->setReadOnly(true);

    // // QFutureWatcher<int> watcher;
    // // connect the watcher with handleFinish function
    // // connect(&watcher, &QFutureWatcher<int>::finished, this, &MainWindow::handleFinish);

    // // QFuture<int> future = QtConcurrent::run(simOpts, ui->opFolder->text());
    // future = QtConcurrent::run(simOpts, ui->opFolder->text());
    // // watch the concurrent execution
    // watcher.setFuture(future);

    // free(runtimeFolder);
    return;
}

void MainWindow::handleFinish()
{
    // int setStatus = future.result();
    // ui->runButton->setEnabled(true);
    // ui->stopButton->setEnabled(false);

    // ui->searchFolder->setEnabled(true);
    // ui->opFolder->setReadOnly(false);

    // if (setStatus == 0)
    // {
    //     ui->runtimeMessages->append("Execution Finished Successfully!");
    // }else
    // {
    //     ui->runtimeMessages->append("Execution Finished with error.");
    //     ui->runtimeMessages->append("Check command line output for more details.");
    // }


    return;
}

// Find operating folder

void MainWindow::findOpFolder()
{
    QString opFolderName = QFileDialog::getExistingDirectory(this);
    ui->opFolder->setText(opFolderName);
    return;
}

// save input file 2D

void MainWindow::saveInput2D()
{
    QString folderName = QFileDialog::getExistingDirectory(this);
    QFile myFile(folderName + "\\input.txt");
    if(myFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        myFile.write(ui->textDisplay2D->toPlainText().toStdString().c_str());
    }
    myFile.close();
    return;
}

void MainWindow::clearText2D()
{
    ui->textDisplay2D->clear();
    return;
}

// save input file 3D

void MainWindow::saveInput3D()
{
    QString folderName = QFileDialog::getExistingDirectory(this);
    QFile myFile(folderName + "\\input.txt");
    if(myFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        myFile.write(ui->textDisplay3D->toPlainText().toStdString().c_str());
    }
    myFile.close();
    return;
}

void MainWindow::clearText3D()
{
    ui->textDisplay3D->clear();
    return;
}

void MainWindow::clearTextRun()
{
    ui->runtimeMessages->clear();
    return;
}

// Toggle Input Type 3D

void MainWindow::toggleInput3D()
{
    int index = 0;
    if(ui->inputType3D->currentText() == "Stack .jpg")
        index = 0;
    else if(ui->inputType3D->currentText() == ".csv")
        index = 1;
    ui->stackedWidget_input3D->setCurrentIndex(index);
    return;
}


// Toggle Part/Pore Label 2D

void MainWindow::togglePoreLabel2D()
{
    if(ui->poreSD->isChecked())
    {
        ui->poreLabel->setCheckable(true);
        ui->poreLabel_Out->setReadOnly(false);
        ui->poreSD_Out->setReadOnly(false);
    } else
    {
        ui->poreLabel->setCheckable(false);
        ui->poreSD_Out->clear();
        ui->poreLabel_Out->clear();
        ui->poreLabel_Out->setReadOnly(true);
        ui->poreSD_Out->setReadOnly(true);
    }
    return;
}

void MainWindow::togglePoreLabel3D()
{
    if(ui->poreSD_3D->isChecked())
    {
        ui->poreLabel_3D->setCheckable(true);
        ui->poreLabel_Out3D->setReadOnly(false);
        ui->poreSD_Out3D->setReadOnly(false);
    } else
    {
        ui->poreLabel_3D->setCheckable(false);
        ui->poreLabel_Out3D->clear();
        ui->poreSD_Out3D->clear();
        ui->poreLabel_Out3D->setReadOnly(true);
        ui->poreSD_Out3D->setReadOnly(true);
    }
    return;
}

void MainWindow::togglePartLabel2D()
{
    if(ui->partSD->isChecked())
    {
        ui->partLabel->setCheckable(true);
        ui->partLabel_Out->setReadOnly(false);
        ui->partSD_Out->setReadOnly(false);
    } else
    {
        ui->partLabel->setCheckable(false);
        ui->partSD_Out->clear();
        ui->partLabel_Out->clear();
        ui->partLabel_Out->setReadOnly(true);
        ui->partSD_Out->setReadOnly(true);
    }
    return;
}

void MainWindow::togglePartLabel3D()
{
    if(ui->partSD_3D->isChecked())
    {
        ui->partLabel_3D->setCheckable(true);
        ui->partLabel_Out3D->setReadOnly(false);
        ui->partSD_Out3D->setReadOnly(false);
    } else
    {
        ui->partLabel_3D->setCheckable(false);
        ui->partSD_Out3D->clear();
        ui->partLabel_Out3D->clear();
        ui->partLabel_Out3D->setReadOnly(true);
        ui->partSD_Out3D->setReadOnly(true);
    }
    return;
}


// Update text

void MainWindow::updateFileText3D()
{
    // header
    QString header = "Input Parameters:";
    ui->textDisplay3D->setText(header);

    QString nD = "nD: 3";

    ui->textDisplay3D->append(nD);

    // file extensions

    QString csv = ".csv";

    // InputType (1 = stack, 0 = .csv)

    QString inputType = "inputType: ";
    int index = 0;
    if(ui->inputType3D->currentText() == "Stack .jpg")
        index = 1;
    else if(ui->inputType3D->currentText() == ".csv")
        index = 0;
    ui->textDisplay3D->append(inputType + QString::number(index));

    if(index == 1)
    {
        // stack size
        QString stackSize = "nSlices: ";
        ui->textDisplay3D->append(stackSize + QString::number(ui->StackSize3D->value()));

        // leading zeroes
        QString leadZero = "leadZero: ";
        ui->textDisplay3D->append(leadZero + QString::number(ui->LeadingZeros3D->value()));

        // threshold
        QString TH = "TH: ";
        ui->textDisplay3D->append(TH + QString::number(ui->Gray_TH_3D->value()));
    } else if(index == 0)
    {
        // check if user entered file name
        QString filenameLabel = "inputFilename: ";
        if(ui->FilenameInput3D_csv->text().isEmpty())
        {
            QString error = "Error: please enter input file name.";
            ui->textDisplay3D->setText(error);
            return;
        }else
        {
            ui->textDisplay3D->append(filenameLabel + ui->FilenameInput3D_csv->text() + csv);
            QString width = "width: ";
            QString depth = "depth: ";
            QString height = "height: ";

            ui->textDisplay3D->append(width + QString::number(ui->width3D->value()));
            ui->textDisplay3D->append(height + QString::number(ui->height3D->value()));
            ui->textDisplay3D->append(depth + QString::number(ui->depth3D->value()));
        }
    }

    // check what code modes to run

    if(ui->poreSD_3D->isChecked())
    {
        QString poreOut = "poreOut: ";
        if (ui->poreSD_Out3D->text().isEmpty())
        {
            QString defaultPoreOut = "defaultPoreOut.csv";
            ui->textDisplay3D->append(poreOut + defaultPoreOut);
        }else
        {
            ui->textDisplay3D->append(poreOut + ui->poreSD_Out3D->text() + csv);
        }
    }

    // pore labels

    if(ui->poreLabel_3D->isChecked())
    {
        QString poreLabelOut = "poreLabelOut: ";
        if (ui->poreLabel_Out3D->text().isEmpty())
        {
            QString default_poreLabelOut = "def_poreLabelOut.csv";
            ui->textDisplay3D->append(poreLabelOut + default_poreLabelOut);
        }
        else
        {
            ui->textDisplay3D->append(poreLabelOut + ui->poreLabel_Out3D->text() + csv);
        }
    }

    // particle-size distribution

    if(ui->partSD_3D->isChecked())
    {
        QString partOut = "partOut: ";
        if (ui->partSD_Out3D->text().isEmpty())
        {
            QString default_partSDOut = "def_partSDOut.csv";
            ui->textDisplay3D->append(partOut + default_partSDOut);
        }else
        {
            ui->textDisplay3D->append(partOut + ui->partSD_Out3D->text() + csv);
        }
    }

    // particle labels

    if(ui->partLabel_3D->isChecked())
    {
        QString partLabelOut = "partLabelOut: ";
        if (ui->partLabel_Out3D->text().isEmpty())
        {
            QString default_partLabelOut = "def_partLabelOut.csv";
            ui->textDisplay3D->append(partLabelOut + default_partLabelOut);
        }else
        {
            ui->textDisplay3D->append(partLabelOut + ui->partLabel_Out3D->text() + csv);
        }
    }

    // Check nThreads

    QString nThreadsLabel = "nThreads: ";
    ui->textDisplay3D->append(nThreadsLabel + QString::number(ui->nThreads3D->value()));

    // Max Radius:

    QString MaxR_Label = "maxR: ";
    ui->textDisplay3D->append(MaxR_Label + QString::number(ui->maxR_3D->value()));

    // verbose

    QString verbose_label = "verbose: ";
    ui->textDisplay3D->append(verbose_label + QString::number(ui->verbose3D->isChecked()));

    // Radius Offset

    QString offsetR_3D_label = "offsetR: ";
    ui->textDisplay3D->append(offsetR_3D_label + QString::number(ui->r_off_3D->value()));

    return;
}


void MainWindow::updateFileText()
{
    // header
    QString header = "Input Parameters:";
    ui->textDisplay2D->setText(header);

    QString nD = "nD: 2";

    ui->textDisplay2D->append(nD);

    // file extensions

    QString csv = ".csv";
    QString jpg = ".jpg";

    // Input file name

    QString s = "inputFilename: ";
    if(ui->FilenameInput2D->text().isEmpty())
    {
        QString error = "Error: invalid input file name.";
        ui->textDisplay2D->setText(error);
        return;
    }
    ui->textDisplay2D->append(s + ui->FilenameInput2D->text() + jpg);

    // Thresholding (Grayscale)

    QString TH2D_Label = "TH: ";
    ui->textDisplay2D->append(TH2D_Label + QString::number(ui->Gray_TH->value()));

    // check what code to run

    // pore-size distribution

    if(ui->poreSD->isChecked())
    {
        QString poreOut = "poreOut: ";
        if (ui->poreSD_Out->text().isEmpty())
        {
            QString defaultPoreOut = "defaultPoreOut.csv";
            ui->textDisplay2D->append(poreOut + defaultPoreOut);
        }else
        {
            ui->textDisplay2D->append(poreOut + ui->poreSD_Out->text() + csv);
        }
    }

    // pore labels

    if(ui->poreLabel->isChecked())
    {
        QString poreLabelOut = "poreLabelOut: ";
        if (ui->poreLabel_Out->text().isEmpty())
        {
            QString default_poreLabelOut = "def_poreLabelOut.csv";
            ui->textDisplay2D->append(poreLabelOut + default_poreLabelOut);
        }
        else
        {
            ui->textDisplay2D->append(poreLabelOut + ui->poreLabel_Out->text() + csv);
        }
    }

    // particle-size distribution

    if(ui->partSD->isChecked())
    {
        QString partOut = "partOut: ";
        if (ui->partSD_Out->text().isEmpty())
        {
            QString default_partSDOut = "def_partSDOut.csv";
            ui->textDisplay2D->append(partOut + default_partSDOut);
        }else
        {
            ui->textDisplay2D->append(partOut + ui->partSD_Out->text() + csv);
        }
    }

    // particle labels

    if(ui->partLabel->isChecked())
    {
        QString partLabelOut = "partLabelOut: ";
        if (ui->partLabel_Out->text().isEmpty())
        {
            QString default_partLabelOut = "def_partLabelOut.csv";
            ui->textDisplay2D->append(partLabelOut + default_partLabelOut);
        }else
        {
            ui->textDisplay2D->append(partLabelOut + ui->partLabel_Out->text() + csv);
        }
    }

    // Check nThreads

    QString nThreadsLabel = "nThreads: ";
    ui->textDisplay2D->append(nThreadsLabel + QString::number(ui->nThreads->value()));

    // Max Radius:

    QString MaxR_Label = "maxR: ";
    ui->textDisplay2D->append(MaxR_Label + QString::number(ui->maxR_2D->value()));

    // verbose

    QString verbose_label = "verbose: ";
    ui->textDisplay2D->append(verbose_label + QString::number(ui->verbose2D->isChecked()));

    // Radius Offset

    QString offsetR_2D_label = "offsetR: ";
    ui->textDisplay2D->append(offsetR_2D_label + QString::number(ui->r_off_2D->value()));

    // Batch Flag

    QString batchLabel2D = "BatchSim: ";
    ui->textDisplay2D->append(batchLabel2D + "0");

    return;
}

// MainWindow::~MainWindow()
// {
//     delete ui;
// }


