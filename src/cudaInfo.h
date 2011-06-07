#pragma once

#include <string>

int initCudaInfo();

void destroyCudaInfo();

bool kernelExecutionTimeout(int device);

bool concurrentCopyAndExecution(int device);

unsigned int textureAlignment(int device);

int numberOfMultiprozessors(int device);

int clockRate(int device);

int maxThreadsPerBlock(int device);

int maxBlocksizeX(int device);

int maxBlocksizeY(int device);

int maxBlocksizeZ(int device);

int maxGridsizeX(int device);

int maxGridsizeY(int device);

int maxGridsizeZ(int device);

unsigned int maxMemoryPitch(int device);

int getWarpSize(int device);

unsigned int getTotalGlobalMemory( unsigned int& globalMemory, unsigned int& sharedMemory, int device);

unsigned int getSharedMemory( unsigned int& globalMemory, unsigned int& sharedMemory, int device);

std::string getName(int device);

float getRevisionNumber(int& major, int& minor, int device);
