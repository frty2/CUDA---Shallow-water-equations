#pragma once

#include "types.h"

void createLandscapeFloat(float *heightmap, int heightmap_width, int heightmap_height,
                     int width, int height, vertex *& vertices);
                     

void createLandscapeRGB(rgb *img, int img_width, int img_height,
                     int width, int height, vertex *& vertices);

void createHeightData(rgb *img, int img_width, int img_height,
                      int width, int height, float *& heights);

void createLandscapeColors(rgb *img, int img_width, int img_height,
                           int width, int height, rgb *& colors);
