#include <glog/logging.h>
#include "types.h"
#include <stdio.h>
#include <string>
#include <cstring>

int getDeviceCount()
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    return devCount;
}

cudaDeviceProp getDevProperties(int device)
{
    cudaDeviceProp devProp;
    if (getDeviceCount() > device && device >= 0 )
    {
        cudaGetDeviceProperties(&devProp, device);
        return devProp;
    }
    return devProp;
}

bool kernelExecutionTimeout(int device)
{
    if (getDeviceCount() > device && device >= 0 )
    {
        cudaDeviceProp devProp = getDevProperties(device);
        return devProp.kernelExecTimeoutEnabled;
    }
}

bool concurrentCopyAndExecution(int device)
{
    if (getDeviceCount() > device && device >= 0 )
    {
        cudaDeviceProp devProp = getDevProperties(device);
        return devProp.kernelExecTimeoutEnabled;
    }
}

unsigned int textureAlignment(int device)
{

    if (getDeviceCount() > device && device >= 0 )
    {
        cudaDeviceProp devProp = getDevProperties(device);
        return devProp.textureAlignment;
    }
}

int numberOfMultiprozessors(int device)
{

    if (getDeviceCount() > device && device >= 0 )
    {
        cudaDeviceProp devProp = getDevProperties(device);
        return devProp.multiProcessorCount;
    }
}

int clockRate(int clock, int device)
{
    if (getDeviceCount() > device && device >= 0 )
    {
        cudaDeviceProp devProp = getDevProperties(device);
        return clock = devProp.clockRate;
    }
}

int maxThreadsPerBlock(int device)
{
    if (getDeviceCount() > device && device >= 0 )
    {
        cudaDeviceProp devProp = getDevProperties(device);
        return devProp.maxThreadsPerBlock;
    }
}

void maxDimBlock(int& threadsx, int& threadsy, int& threadsz, int device)
{
    if (getDeviceCount() > device && device >= 0 )
    {
        cudaDeviceProp devProp = getDevProperties(device);
        threadsx = devProp.maxThreadsDim[0];
        threadsy = devProp.maxThreadsDim[1];
        threadsz = devProp.maxThreadsDim[2];
    }
}

void maxDimGrid(int& blocksx, int& blocksy, int& blocksz, int device)
{

    if (getDeviceCount() > device && device >= 0 )
    {
        cudaDeviceProp devProp = getDevProperties(device);
        blocksx = devProp.maxGridSize[0];
        blocksy = devProp.maxGridSize[1];
        blocksz = devProp.maxGridSize[2];
    }
}

unsigned int maxMemoryPitch(int device)
{

    if (getDeviceCount() > device && device >= 0 )
    {
        cudaDeviceProp devProp = getDevProperties(device);
        return devProp.memPitch;
    }
}

int getWarpSize(int device)
{
    if (getDeviceCount() > device && device >= 0 )
    {
        cudaDeviceProp devProp = getDevProperties(device);
        return devProp.warpSize;
    }
}

void getMemory( unsigned int& globalMemory, unsigned int& sharedMemory,
                unsigned int& constMemory, int& registerPerBlock, int device)
{
    if (getDeviceCount() > device && device >= 0 )
    {
        cudaDeviceProp devProp = getDevProperties(device);
        globalMemory = devProp.totalGlobalMem;
        sharedMemory = devProp.sharedMemPerBlock;
        constMemory = devProp.totalConstMem;
        registerPerBlock = devProp.regsPerBlock;
    }
}

std::string getName(int device)
{
    if (getDeviceCount() > device && device >= 0 )
    {
        cudaDeviceProp devProp = getDevProperties(device);
        return devProp.name;
    }
}

void getRevisionNumber(int& major, int& minor, int device)
{
    if (getDeviceCount() > device && device >= 0 )
    {
        cudaDeviceProp devProp = getDevProperties(device);
        major = devProp.major;
        minor = devProp.minor;
    }
}