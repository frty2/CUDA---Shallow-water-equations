#include <iostream>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <stdio.h>

#include "stdlib.h"
#include "types.h"
#include "ppm_reader.h"
#include "3d_window.h"
#include "math.h"
#include "wavesimulator.h"
#include "wavesceneparser.h"
#include "landscapecreator.h"

int gridsize;

static bool validateGridsize(const char* flagname, int value)
{
    if (value > 1 && value < 2<<16)
        { return true; }
    std::cout << "Invalid value for --" << flagname << ": "
              << value << std::endl;
    return false;
}

DEFINE_int32(gridsize, 256, "gridsize");

static const bool gridsize_dummy = google::RegisterFlagValidator(&FLAGS_gridsize, &validateGridsize);



void updateFunction(vertex* wave_vertices, rgb* wave_colors)
{
    computeNext(gridsize, gridsize, wave_vertices, wave_colors);
}

int main(int argc, char ** argv)
{
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    google::ParseCommandLineFlags(&argc, &argv, true);
    
    gridsize=FLAGS_gridsize;

    rgb *landscape_img;
    int landscape_width;
    int landscape_height;

    rgb *colors_img;
    int colors_width;
    int colors_height;

    rgb *threshholds_img;
    int threshholds_width;
    int threshholds_height;

    rgb *wave_img;
    int wave_width;
    int wave_height;

    float running_time;
    std::string landscape_filename, landscape_color_filename, threshholds_filename, wave_filename, colors_filename;
    parse_wavescene("../res/wavescene.yaml", landscape_filename, landscape_color_filename, threshholds_filename, wave_filename, running_time);

    readPPM(landscape_filename.c_str(), landscape_img, landscape_width, landscape_height);
/*    
    readPPM(threshholds_filename.c_str(), threshholds_img, threshholds_width, threshholds_height);
    readPPM(wave_filename.c_str(), wave_img, wave_width, wave_height);
    readPPM(landscape_color_filename.c_str(), colors_img, colors_width, colors_height);

    vertex *landscape;
    float *threshholds;
    vertex *wave;
    float *waveheights;
    rgb *colors;



    createLandscape(landscape_img, landscape_width, landscape_height, gridsize, gridsize, landscape);

    createHeightData(threshholds_img, threshholds_width, threshholds_height, gridsize, gridsize, threshholds);

    createLandscape(wave_img, wave_width, wave_height, gridsize, gridsize, wave);

    createHeightData(wave_img, wave_width, wave_height, gridsize, gridsize, waveheights);

    createLandscapeColors(colors_img, colors_width, colors_height, gridsize, gridsize, colors);


    initWaterSurface(gridsize, gridsize, landscape, waveheights, threshholds);

    createWindow(argc, argv, 800, 600, gridsize, gridsize, landscape, wave, colors, &updateFunction);

    free(landscape_img);
    free(threshholds_img);
    free(wave_img);
    free(waveheights);
    free(colors_img);
*/
    return 0;
}
