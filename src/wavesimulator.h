#pragma once

void initWaterSurface(int width, int height, float *&heights, velocity *velocities);

float* computeNext(float time);