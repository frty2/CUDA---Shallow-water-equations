#pragma once

#include "types.h"

void initWaterSurface(int width, int height, vertex* heightmapvertices, float* treshholds);

void computeNext(int width, int height, vertex* watersurfacevertices, rgb* watersurfacecolors);

void destroyWaterSurface();
