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

cudaDeviceProp getDevProperties(const int& device)
	{
	cudaDeviceProp devProp;
	if (getDeviceCount() > device && device >= 0 )
		{
		cudaGetDeviceProperties(&devProp, device);
		return devProp;
		}
	return devProp;
	}

void kernelExecutionTimeout(bool& ket, const int& device)
	{
	cudaDeviceProp devProp = getDevProperties(device);
	if (getDeviceCount() > device && device >= 0 )
		{
		if (ket != NULL)
			ket = devProp.kernelExecTimeoutEnabled ? true : false;
		}
	}

void concurrentCopyAndExecution(bool& ccae, const int& device)
	{
	cudaDeviceProp devProp = getDevProperties(device);
	if (getDeviceCount() > device && device >= 0 )
		{
		if (ccae != NULL)
			ccae = devProp.kernelExecTimeoutEnabled ? true : false;
		}
	}

void textureAlignment(unsigned int& textureAlign, const int& device)
	{
	cudaDeviceProp devProp = getDevProperties(device);
	if (getDeviceCount() > device && device >= 0 )
		{
		if (textureAlign != NULL)
			textureAlign = devProp.textureAlignment;
		}
	}

void numberOfMultiprozessors(int& mp, const int& device)
	{
	cudaDeviceProp devProp = getDevProperties(device);
	if (getDeviceCount() > device && device >= 0 )
		{
		if (mp != NULL)
			mp = devProp.multiProcessorCount;
		}
	}

void clockRate(int& clock, const int& device)
	{
	cudaDeviceProp devProp = getDevProperties(device);
	if (getDeviceCount() > device && device >= 0 )
		{
		if (clock != NULL)
			clock = devProp.clockRate;
		}
	}

void maxThreadsPerBlock(int& threads, const int& device)
	{
	cudaDeviceProp devProp = getDevProperties(device);
	if (getDeviceCount() > device && device >= 0 )
		{
		if (threads != NULL)
			threads = devProp.maxThreadsPerBlock;
		}
	}
	
void maxDimBlock(int& threadsx, int& threadsy, int& threadsz, const int& device)
	{
	cudaDeviceProp devProp = getDevProperties(device);
	if (getDeviceCount() > device && device >= 0 )
		{
		if (threadsx != NULL)
			threadsx = devProp.maxThreadsDim[0];
		if (threadsy != NULL)
			threadsy = devProp.maxThreadsDim[1];
		if (threadsz != NULL)
			threadsz = devProp.maxThreadsDim[2];
		}
	}
	
void maxDimGrid(int& blocksx, int& blocksy, int& blocksz, const int& device)
	{
	cudaDeviceProp devProp = getDevProperties(device);
	if (getDeviceCount() > device && device >= 0 )
		{
		if (blocksx != NULL)
			blocksx = devProp.maxGridSize[0];
		if (blocksy != NULL)
			blocksy = devProp.maxGridSize[1];
		if (blocksz != NULL)
			blocksz = devProp.maxGridSize[2];
		}
	}

void maxMemoryPitch(unsigned int& memoryPitch, const int& device)
	{
	cudaDeviceProp devProp = getDevProperties(device);
	if (getDeviceCount() > device && device >= 0 )
		{
		if (memoryPitch != NULL)
			memoryPitch = devProp.memPitch;
		}
	}

void getWarpSize(int& warpS, const int& device)
	{
	cudaDeviceProp devProp = getDevProperties(device);
	if (getDeviceCount() > device && device >= 0 )
		{
		if (warpS != NULL)
			warpS = devProp.warpSize;
		}
	}

void getMemory(	unsigned int& globalMemory, unsigned int& sharedMemory, 
				unsigned int& constMemory, int& registerPerBlock, const int& device)
	{
	cudaDeviceProp devProp = getDevProperties(device);
	if (getDeviceCount() > device && device >= 0 )
		{
		if (globalMemory != NULL)
			globalMemory = devProp.totalGlobalMem;
		if (sharedMemory != NULL)
			sharedMemory = devProp.sharedMemPerBlock;
		if (constMemory != NULL)
			constMemory = devProp.totalConstMem;
		if (registerPerBlock != NULL)
			registerPerBlock = devProp.regsPerBlock;
		}
	}

void getName(std::string& stream, const int& device)
	{
	cudaDeviceProp devProp = getDevProperties(device);
	if (getDeviceCount() > device && device >= 0 )
		{
		stream << devProp.name.c_str();
		}
	}

void getRevisionNumber(int& major, int& minor, const int& device)
	{
	cudaDeviceProp devProp = getDevProperties(device);
	if (getDeviceCount() > device && device >= 0 )
		{
		if (major != NULL)
			major = devProp.major;
		if (minor != NULL)
			minor = devProp.minor;
		}
	}


	
	
