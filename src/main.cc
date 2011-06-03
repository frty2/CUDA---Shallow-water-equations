#include <iostream>
#include <glog/logging.h>

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
    
    int vertcount = (w-1)*(h-1)*4;
    vertex *vertices = (vertex *) malloc(vertcount*sizeof(vertex));
    CHECK_NOTNULL(vertices);
    
    for(int y = 0;y < h - 1;y++)
    {
        for(int x = 0;x < w - 1;x++)
        {
            vertex v1;
            v1.x = x/128.0f-2.0f;
            v1.z = -img[y*(w)+x].x/512.0f;
            v1.y = y/128.0f-2.0f;
            
            vertex v2;
            v2.x = x/128.0f-2.0f;
            v2.z = -img[y*(w)+x+1].x/512.0f;
            v2.y = y/128.0f-2.0f;
            
            vertex v3;
            v3.x = x/128.0f-2.0f;
            v3.z = -img[(y+1)*(w)+x].x/512.0f;
            v3.y = y/128.0f-2.0f;
            
            vertex v4;
            v4.x = x/128.0f-2.0f;
            v4.z = -img[(y+1)*(w)+x+1].x/512.0f;
            v4.y = y/128.0f-2.0f;
            
            vertices[4*(y*(w-1)+x)] = v1;
            vertices[4*(y*(w-1)+x)+2] = v3;
            vertices[4*(y*(w-1)+x)+3] = v4;
            vertices[4*(y*(w-1)+x)+1] = v2;
        }
    }
    
    createWindow(argc, argv, w, h, NULL, vertices, vertcount);
    
    free(img);
    free(vertices);
    return 0;
}