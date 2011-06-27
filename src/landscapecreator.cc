#include "landscapecreator.h"

#include <stdlib.h>
#include <glog/logging.h>

#include "types.h"

float vertexheight(rgb color)
{
    return color.x / 256.0f + color.y / 256.0f + color.z / 256.0f;
}

void createLandscape(rgb *img, int img_width, int img_height,
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
            v.x = x * 16.0f / (width - 1) - 8;
            v.z = y * 16.0f / (height - 1) - 8;
            v.y = v.x/20+0.4f;//vertexheight(img[imgy * img_width + imgx]);
            vertices[y * width + x] = v;
        }
    }
}

void createHeightData(rgb *img, int img_width, int img_height,
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
            heights[y * width + x] = vertexheight(img[imgy * img_width + imgx]);
        }
    }
}

void createLandscapeColors(rgb *img, int img_width, int img_height,
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
            colors[y * width + x] = img[imgy * img_width + imgx];
        }
    }
}
