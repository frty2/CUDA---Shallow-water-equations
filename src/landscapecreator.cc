#include "landscapecreator.h"

#include <stdlib.h>
#include <glog/logging.h>

#include "types.h"

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

float vertexheight(rgb color)
{
    return ((color.x / 255.0f + color.y / 255.0f + color.z / 255.0f) / 1.5f - 1.0f) * 2.0f + 5.0f;
}

float vertexheightf(float f)
{
    return f / 1000.0f + 5.0f;
}

void createLandscapeFloat(float *heightmap, int heightmap_width, int heightmap_height,
                          int width, int height, vertex *& vertices)
{
    vertices = (vertex *) malloc(width * height * sizeof(vertex));
    CHECK_NOTNULL(vertices);

    for(int y = 0; y < height; y ++)
    {
        for(int x = 0; x < width; x ++)
        {
            int imgx = x * (heightmap_width - 1) / (width - 1);
            int imgy = y * (heightmap_height - 1) / (height - 1);
            vertex v;
            v.x = x * 20.0f / (width - 1) - 10;
            v.z = y * 20.0f / (height - 1) - 10;
            v.y = vertexheightf(heightmap[imgy * heightmap_width + imgx]);
            vertices[y * width + x] = v;
        }
    }
}

void createLandscapeFromRGB(rgb *img, int img_width, int img_height,
                            int width, int height, vertex *& vertices)
{
    vertices = (vertex *) malloc(width * height * sizeof(vertex));
    CHECK_NOTNULL(vertices);

    for(int y = 0; y < height; y ++)
    {
        for(int x = 0; x < width; x ++)
        {
            int imgx = x * (img_width - 1) / (width - 1);
            int imgy = y * (img_height - 1) / (height - 1);
            vertex v;
            v.x = x * 20.0f / (width - 1) - 10;
            v.z = y * 20.0f / (height - 1) - 10;
            v.y = vertexheight(img[imgy * img_width + imgx]);
            vertices[y * width + x] = v;
        }
    }
}

void createWaveHeights(rgb *img, int img_width, int img_height,
                       int width, int height, float *& heights)
{
    heights = (float *) malloc(width * height * sizeof(float));
    CHECK_NOTNULL(heights);

    for(int y = 0; y < height; y ++)
    {
        for(int x = 0; x < width; x ++)
        {
            int imgx = x * (img_width - 1) / (width - 1);
            int imgy = y * (img_height - 1) / (height - 1);
            heights[y * width + x] = (vertexheight(img[imgy * img_width + imgx]) - 3.0f) / 5.0f;
        }
    }
}

void createLandscapeColors(rgb *img, vertex *vertices, int img_width, int img_height,
                           int width, int height, rgb *& colors)
{
    colors = (rgb *) malloc(width * height * sizeof(rgb));
    CHECK_NOTNULL(colors);

    for(int y = 0; y < height; y ++)
    {
        for(int x = 0; x < width; x ++)
        {
            int imgx = x * (img_width - 1) / (width - 1);
            int imgy = y * (img_height - 1) / (height - 1);
            rgb color;
            int heightcolor = (vertices[y * width + x].y - 5) * 20.0f;
            color.x = max(min(-heightcolor + img[imgy * img_width + imgx].x, 255), 0);
            color.y = max(min(-heightcolor + img[imgy * img_width + imgx].y, 255), 0);
            color.z = max(min(-heightcolor + img[imgy * img_width + imgx].z, 255), 0);
            colors[y * width + x] = color;
        }
    }
}
