#pragma once

#include "types.h"
//void displayImage(rgb* pixelarray, int width, int height);

void createWindow(int argc, char **argv, int width, int height, void (*callback)(), int vertex_width, int vertex_height, rgb *heightmap_img, rgb *color_img);