#pragma once
int getDeviceCount();
cudaDeviceProp getDevProperties(const int& device);
void kernelExecutionTimeout(bool& ket, const int& device);
void concurrentCopyAndExecution(bool& ccae, const int& device);
void textureAlignment(unsigned int& textureAlign, const int& device);
void numberOfMultiprozessors(int& mp, const int& device);
void clockRate(int& clock, const int& device);
void maxThreadsPerBlock(int& threads, const int& device);
void maxDimBlock(int& threadsx, int& threadsy, int& threadsz, const int& device);	
void maxDimGrid(int& blocksx, int& blocksy, int& blocksz, const int& device);
void maxMemoryPitch(unsigned int& memoryPitch, const int& device);
void getWarpSize(int& warpSize, const int& device);
void getMemory(	unsigned int& globalMemory, unsigned int& sharedMemory, 
				unsigned int& constMemory, int& registerPerBlock, const int& device);
void getName(std::string& stream, const int& device);
void getRevisionNumber(int& major, int& minor, const int& device);
