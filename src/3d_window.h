#pragma once

#include "types.h"

void createWindow(int argc, char **argv, int windowwidth, int windowheight,
                  int gridwidth, int gridheight, vertex *landscape, vertex *wave, rgb *colors, void (*updatefunction) (vertex*, rgb*));
                  
                  
