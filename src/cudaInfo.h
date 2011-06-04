#pragma once

int getDeviceCount();

cudaDeviceProp getDevProperties(int device);

bool kernelExecutionTimeout(int device);

bool concurrentCopyAndExecution(int device);

unsigned int textureAlignment(int device);

int numberOfMultiprozessors(int device);

int clockRate(int clock, int device);

int maxThreadsPerBlock(int device);

void maxDimBlock(int& threadsx, int& threadsy, int& threadsz, int device);

void maxDimGrid(int& blocksx, int& blocksy, int& blocksz, int device);

unsigned int maxMemoryPitch(int device);

int getWarpSize(int device);

void getMemory( unsigned int& globalMemory, unsigned int& sharedMemory,
    unsigned int& constMemory, int& registerPerBlock, int device);

std::string getName(int device);

void getRevisionNumber(int& major, int& minor, int device);
