#pragma once

#include "types.h"

void initWaterSurface(int width, int height, float *&heights, velocity *velocities);

void computeNext(float time);

void destroyWaterSurface();