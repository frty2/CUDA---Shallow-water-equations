#pragma once

#include "types.h"

void initWaterSurface(int width, int height, vertex* heightmapvertices);

void computeNext(float time, int width, int height, vertex* watersurfacevertices, rgb* watersurfacecolors); 

void destroyWaterSurface();