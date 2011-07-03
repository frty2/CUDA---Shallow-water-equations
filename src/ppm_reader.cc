#include "ppm_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <glog/logging.h>
#include <sstream>

#include "types.h"
#include "stdlib.h"

void readPPM(const char* filename, rgb *&image, int& width, int& height)
{
    std::ifstream ifs(filename);
    LOG_IF(FATAL, ifs.fail()) << "Failed opening file '" << filename << "'";

    std::string ppmFormat;
    ifs >> ppmFormat;
    CHECK_STREQ("P3", ppmFormat.c_str());

    std::string commentcheck;

    ifs >> commentcheck;
    if ( '#' == commentcheck[0] )
    {
        char dummy[256];
        ifs.getline(dummy, 256);
        ifs >> width;
    }
    else
    {
        std::stringstream ss(commentcheck);
        ss >> width;
    }
    CHECK_LT(0, width);
    ifs >> height;
    CHECK_LT(0, height);

    int max;
    ifs >> max;
    CHECK_LE(0, max);
    CHECK_LE(max, 255);

    image = (rgb*) malloc(sizeof(*image) * height * width);
    CHECK_NOTNULL(image);

    int r, g, b;
    size_t i = 0;
    while (ifs >> r && ifs >> g && ifs >> b)
    {
        CHECK_LE(0, r);
        CHECK_LE(r, max);
        CHECK_LE(0, g);
        CHECK_LE(g, max);
        CHECK_LE(0, b);
        CHECK_LE(b, max);
        image[i].x = r;
        image[i].y = g;
        image[i].z = b;
        image[i].w = 0;
        i++;
    }
    CHECK_EQ((unsigned int)height * width, i);
}
