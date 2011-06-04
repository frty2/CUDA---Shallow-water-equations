#pragma once

#include "types.h"
//void displayImage(rgb* pixelarray, int width, int height);

void createWindow(int argc, char **argv, int width, int height, void (*callback)(), int vertex_width, int vertex_height, vertex *vertices = NULL);