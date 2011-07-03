#include "asc_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <glog/logging.h>

#include "types.h"
#include "ppm_reader.h"

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#define grid2Dread(array, x, y, pitch) array[(y)*pitch+x]

void readASC(const char* filename, int &heightmapwidth, int &heightmapheight, float*& heightmap)
{
    int filewidth;
    int fileheight;

    std::ifstream ifs(filename);
    LOG_IF(FATAL, ifs.fail()) << "Failed opening file '" << filename << "'";

    std::string ncols;
    ifs >> ncols;
    CHECK_STREQ("ncols", ncols.c_str());
    ifs >> filewidth;

    std::string nrows;
    ifs >> nrows;
    CHECK_STREQ("nrows", nrows.c_str());
    ifs >> fileheight;

    std::string dummy;
    // skip next 8 entries
    for(int skip = 0; skip < 8; skip++)
        { ifs >> dummy; }


    int dim = min(fileheight, filewidth);
    heightmapwidth = dim;
    heightmapheight = dim;
    heightmap = (float*)malloc(dim * dim * sizeof(float));

    for(int y = 0; y < dim; y++)
    {
        int x;
        for(x = 0; x < dim; x++)
        {
            ifs >> heightmap[x + y * dim];
        }
        for(; x < filewidth; x++)
        {
            ifs >> dummy;
        }
    }
    ifs.close();
}