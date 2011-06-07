#include "cudaInfo.h"

#include <glog/logging.h>
#include <stdio.h>
#include <string>
#include <stdlib.h>

#include "types.h"

cudaDeviceProp *devprops = NULL;

int initCudaInfo()
{
    cudaError_t error;
    int devicecount;
    
    error =  cudaGetDeviceCount(&devicecount);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);    
    CHECK_NE(devicecount, 0) << "Error: no CUDA devices found.";
    
    devprops = (cudaDeviceProp*)malloc(sizeof(cudaDeviceProp)*devicecount);
    CHECK_NOTNULL(devprops);
    
    for( int device = 0; device < devicecount; device++ )
    {
        error =  cudaGetDeviceProperties(&devprops[device], device);
        CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    }
    
    return devicecount;
}

void destroyCudaInfo()
{
    if(devprops != NULL)
    {
        free(devprops);
        devprops = NULL;
    }
}

bool kernelExecutionTimeout(int device)
{
    return devprops[device].kernelExecTimeoutEnabled;
}

bool concurrentCopyAndExecution(int device)
{
    return devprops[device].kernelExecTimeoutEnabled;
}

unsigned int textureAlignment(int device)
{
    return devprops[device].textureAlignment;
}

int numberOfMultiprozessors(int device)
{
    return devprops[device].multiProcessorCount;
}

int clockRate(int device)
{
    return devprops[device].clockRate;
}

int maxThreadsPerBlock(int device)
{
    return devprops[device].maxThreadsPerBlock;
}

int maxBlocksizeX(int device)
{
    return devprops[device].maxThreadsDim[0];
}

int maxBlocksizeY(int device)
{
    return devprops[device].maxThreadsDim[1];
}

int maxBlocksizeZ(int device)
{
    return devprops[device].maxThreadsDim[2];
}

int maxGridsizeX(int device)
{
    return devprops[device].maxGridSize[0];
}

int maxGridsizeY(int device)
{
    return devprops[device].maxGridSize[1];
}

int maxGridsizeZ(int device)
{
    return devprops[device].maxGridSize[2];
}

unsigned int maxMemoryPitch(int device)
{
    return devprops[device].memPitch;
}

int getWarpSize(int device)
{
    return devprops[device].warpSize;
}

unsigned int getTotalGlobalMemory( unsigned int& globalMemory, unsigned int& sharedMemory, int device)
{
    return devprops[device].totalGlobalMem;
}

unsigned int getSharedMemory( unsigned int& globalMemory, unsigned int& sharedMemory, int device)
{
    return devprops[device].sharedMemPerBlock;
}

std::string getName(int device)
{
    return devprops[device].name;
}

float getRevisionNumber(int& major, int& minor, int device)
{
    return devprops[device].major*10 + devprops[device].minor;
}
