#include "landscapecreator.h"

#include <stdlib.h>
#include <glog/logging.h>

#include "types.h"

void createLandscape(rgb *img, int img_width, int img_height, 
                            int width, int height, vertex *& vertices)
{
    vertices = (vertex *) malloc(width*height*sizeof(vertex));
    CHECK_NOTNULL(vertices);
    
    for(int y = 0; y < height; y ++)
    {
        for(int x = 0; x < width; x ++)
        {
            int imgx = x * (img_width - 1) / (width - 1);
            int imgy = y * (img_height - 1) / (height - 1);
            vertex v;
            v.x = x * 16.0f / (width - 1) - 8;
            v.z = y * 16.0f / (height - 1) - 8;
            v.y = img[imgy * img_width + imgx].x / 256.0f + 
                  img[imgy * img_width + imgx].y / 256.0f + 
                  img[imgy * img_width + imgx].z / 256.0f;
            vertices[y * width + x] = v;
        }
    }
}

void createLandscapeColors(rgb *img, int img_width, int img_height, 
                            int width, int height, rgb *& colors)
{
    rgb *vertexcolors = (rgb *) malloc(width * height * sizeof(rgb));
    CHECK_NOTNULL(vertexcolors);
    
    for(int y = 0; y < height; y ++)
    {
        for(int x = 0; x < width; x ++)
        {
            int imgx = x * (img_width - 1) / (width - 1);
            int imgy = y * (img_height - 1) / (height - 1);
            vertexcolors[y * width + x] = img[imgy * img_width + imgx];
        }
    }
}