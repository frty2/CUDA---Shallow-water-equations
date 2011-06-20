#pragma once

#include "types.h"

void createLandscape(rgb *img, int img_width, int img_height, 
                        int width, int height, vertex *& vertices);
                        
void createLandscapeColors(rgb *img, int img_width, int img_height, 
                        int width, int height, rgb *& colors);