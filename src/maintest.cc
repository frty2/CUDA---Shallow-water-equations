#include <iostream>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <stdio.h>

#include "stdlib.h"
#include "types.h"
#include "ppm_reader.h"
#include "math.h"
#include "wavesimulator.h"

int main(int argc, char ** argv)
{
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    google::ParseCommandLineFlags(&argc, &argv, true);
    rgb *heightmap_img;
    rgb *color_img;
    int img_width;
    int img_height;

    readPPM("../res/heightmap.ppm", heightmap_img, img_width, img_height);
    readPPM("../res/texture.ppm", color_img, img_width, img_height);

    //Calc the heightmap out of the rgb
    int width = img_width;
    int height = img_height;

    vertex *vertices = (vertex *) malloc(width * height * sizeof(vertex));
    CHECK_NOTNULL(vertices);

    for(int y = 0; y < height; y ++)
    {
        for(int x = 0; x < width; x ++)
        {
            vertex v;
            v.x = x * 16.0f / width - 8;
            v.z = y * 16.0f / height - 8;
            v.y = heightmap_img[y * width + x].x / 256.0f + heightmap_img[y * width + x].y / 256.0f + heightmap_img[y * width + x].z / 256.0f;

            vertices[y * width + x] = v;
        }
    }

    initWaterSurface(width, height, vertices);

    vertex *watersurfacevertices = (vertex *) malloc(width * height * sizeof(vertex));
    CHECK_NOTNULL(watersurfacevertices);

    rgb *watersurfacecolors = (rgb *) malloc(width * height * sizeof(rgb));
    CHECK_NOTNULL(watersurfacecolors);

    for(int step = 0; step < 10; step++)
    {
        computeNext(width, height, watersurfacevertices, watersurfacecolors);
    }

    free(heightmap_img);
    free(color_img);
    free(watersurfacecolors);
    free(watersurfacevertices);
    return 0;
}