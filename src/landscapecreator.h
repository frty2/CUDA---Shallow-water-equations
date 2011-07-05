#pragma once

#include "types.h"

void createLandscapeFloat(float *heightmap, int heightmap_width, int heightmap_height,
                          int width, int height, vertex *& vertices);


void createLandscapeFromRGB(rgb *img, int img_width, int img_height,
                            int width, int height, vertex *& vertices);

void createWaveHeights(rgb *img, int img_width, int img_height,
                       int width, int height, float *& heights);

void createLandscapeColors(rgb *img, vertex *vertices, int img_width, int img_height,
                           int width, int height, rgb *& colors);
