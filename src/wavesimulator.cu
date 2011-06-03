#import "wavesimulator.h"

const int UNINTIALISED = 0;
const int INITIALISED = 1;

int state = UNINTIALISED;

void initWaterSurface(int width, int height, float *&heights, velocity *velocities)
{
    if(state != UNINTIALISED)
    {
        return;
    }
    
    state = INITIALISED;
}

void computeNext(float time)
{
    if(state != INITIALISED)
    {
        return;
    }
}

void destroyWaterSurface()
{
    if(state != INITIALISED)
    {
        return;
    }
    
    state = UNINTIALISED;
}