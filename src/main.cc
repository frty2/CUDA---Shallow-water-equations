#include <iostream>
#include <glog/logging.h>
#include <stdio.h>
#include "stdlib.h"
#include "types.h"
#include "ppm_reader.h"
#include "3d_window.h"
#include "math.h"


int main(int argc, char ** argv)
{
    google::InitGoogleLogging(argv[0]);

    rgb *heightmap_img;
    rgb *color_img;
    int img_width;
    int img_height;

    readPPM("../res/heightmap.ppm", heightmap_img, img_width, img_height);
    readPPM("../res/texture.ppm", color_img, img_width, img_height);


    createWindow(argc, argv, 800, 600, NULL, 256, 256, heightmap_img, color_img);

    free(heightmap_img);
    free(color_img);
    return 0;
}