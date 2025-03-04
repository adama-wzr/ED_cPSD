#ifndef DATA_STRUCTS_HPP
#define DATA_STRUCTS_HPP

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
    char* folderName;
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


#endif // DATA_STRUCTS_HPP
