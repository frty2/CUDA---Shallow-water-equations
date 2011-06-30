#include "ppm_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <glog/logging.h>

#include "types.h"


#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#define grid2Dread(array, x, y, pitch) array[(y)*pitch+x]

void readASC(const char* filename, int &heightmapwidth, int &heightmapheight,float*& heightmap)
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
    for(int skip=0;skip<8;skip++)
        ifs >> dummy;
    
    
    int dim = min(fileheight,filewidth);
    heightmapwidth=dim;
    heightmapheight=dim;
    heightmap = (float*)malloc(dim*dim*sizeof(float));

    for(int y=0;y<dim;y++)
    {
        int x;
        for(x=0;x<dim;x++)
        {
            ifs >> heightmap[x+y*dim];
        }
        for(;x<filewidth;x++)
        {
            ifs >> dummy;
        }
    }
    ifs.close();   
}

void fitASC(const char* filename, int &heightmapdim, int &fitdim, float*& heightmap)
{
    std::ofstream file(filename);
    file << "ncols         " << fitdim << "\n";
    file << "nrows         " << fitdim << "\n";
    file << "xllcorner     -124.00013888889" << "\n";
    file << "yllcorner     36.99986111111" << "\n";
    file << "cellsize      0.00027777778" << "\n";
    file << "NODATA_value  -9999" << "\n";
    
    int step = max((heightmapdim - 1) / (fitdim - 1), 1);
    
    for(int y = 0; y < fitdim; y++)
    {
        for(int x = 0; x < fitdim; x++)
        {
            int imgx = x * step;
            int imgy = y * step;
            
            float height = 0.0f;
            for(int xx = 0;xx < step;xx++)
            {
                for(int yy = 0;yy < step;yy++)
                {
                    height += grid2Dread(heightmap, imgx+xx, imgy+yy, heightmapdim);
                }
            }
            height = height / (step * step);
            file << height << " ";
        }
    }
    file.close();
}

