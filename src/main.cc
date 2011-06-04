#include <iostream>
#include <glog/logging.h>

#include "stdlib.h"
#include "types.h"
#include "ppm_reader.h"
#include "3d_window.h"

rgb *img;
int w;
int h;

int i = 0;
int main(int argc, char ** argv)
{
    google::InitGoogleLogging(argv[0]);
    
    
    readPPM("../res/mandelbrot.ppm", img, w, h);
    
    
    createWindow(argc, argv, w, h, NULL);
    
    free(img);

    return 0;
}