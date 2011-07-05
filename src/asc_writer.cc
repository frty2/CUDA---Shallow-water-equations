#include "asc_writer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <gflags/gflags.h>
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

static bool validateNorm(const char* flagname, int value)
{
    if (value >= 1)
        { return true; }
    std::cout << "Invalid value for --" << flagname << ": "
              << value << std::endl;
    return false;
}

DEFINE_int32(norm, 1, "norm factor for height data, see README file");

static const bool norm_dummy = google::RegisterFlagValidator(&FLAGS_norm, &validateNorm);

void writeASC(const char* filename, int &heightmapdim, int &fitdim, float*& heightmap)
{
    std::ofstream file(filename);
    file << "ncols         " << fitdim << "\n";
    file << "nrows         " << fitdim << "\n";
    file << "xllcorner     -124.00013888889" << "\n";
    file << "yllcorner     36.99986111111" << "\n";
    file << "cellsize      0.00027777778" << "\n";
    file << "NODATA_value  -9999" << "\n";


    int step = max((heightmapdim - 1) / (fitdim - 1), 1);
    int norm = FLAGS_norm;
    for(int y = 0; y < fitdim; y++)
    {
        for(int x = 0; x < fitdim; x++)
        {
            int imgx = x * step;
            int imgy = y * step;

            float height = 0.0f;
            for(int xx = 0; xx < step; xx++)
            {
                for(int yy = 0; yy < step; yy++)
                {
                    height += grid2Dread(heightmap, imgx + xx, imgy + yy, heightmapdim);
                }
            }
            height = height / (step * step);
            file << height*norm << " ";
        }
    }
    file.close();
}

