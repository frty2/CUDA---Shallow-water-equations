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
    
    rgb *img;
    int img_width;
    int img_height;
    
    readPPM("../res/heightmap.ppm", img, img_width, img_height);
    
    vertex *vertices = (vertex *) malloc(img_width*img_height*sizeof(vertex));
    CHECK_NOTNULL(vertices);
    
   
    for(int y = 0;y < img_height;y ++)
    {
        for(int x = 0; x < img_width;x ++)
        {
            vertex v;
            v.x = x*16.0f/img_width-8;
            v.z = y*16.0f/img_height-8;
            v.y = img[y*img_width+x].x/256.0f+img[y*img_width+x].y/256.0f+img[y*img_width+x].z/256.0f;

            vertices[y*img_width+x] = v;
        }
    }
    
    createWindow(argc, argv, 800, 600, NULL, img_width, img_height, vertices);
    
    free(img);

    return 0;
}