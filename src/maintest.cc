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

void updateFunction(vertex* wave_vertices, rgb* wave_colors)
{
    computeNext(256, 256, wave_vertices, wave_colors);
}

int main(int argc, char ** argv)
{
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    google::ParseCommandLineFlags(&argc, &argv, true);
    
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
    readPPM(threshholds_filename.c_str(), threshholds_img, threshholds_width, threshholds_height);
    readPPM(wave_filename.c_str(), wave_img, wave_width, wave_height);
    readPPM(landscape_color_filename.c_str(), colors_img, colors_width, colors_height);
    
    vertex *landscape;
    float *threshholds;
    vertex *wave;
    float *waveheights;
    rgb *colors;


    createLandscape(landscape_img, landscape_width, landscape_height, 256, 256, landscape);
    
    createHeightData(threshholds_img, threshholds_width, threshholds_height, 256, 256, threshholds);
    
    createLandscape(wave_img, wave_width, wave_height, 256, 256, wave);
    
    createHeightData(wave_img, wave_width, wave_height, 256, 256, waveheights);
                            
    createLandscapeColors(colors_img, colors_width, colors_height, 256, 256, colors);
  
    
    initWaterSurface(256, 256, landscape, waveheights, threshholds);

    for(int i = 0;i < 10;i++)
        updateFunction(wave, colors);
    
    free(landscape_img);
    free(threshholds_img);
    free(wave_img);
    free(waveheights);
    free(colors_img);
    return 0;
}
