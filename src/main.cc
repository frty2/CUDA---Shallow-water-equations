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

#include "asc_reader.h"
#include "asc_writer.h"
#include "timing.h"

int gridsize;
int simulate;

static bool validateWSPF(const char* flagname, int value)
{
    if (value > 0)
        { return true; }
    std::cout << "Invalid value for --" << flagname << ": "
              << value << std::endl;
    return false;
}

static bool validateGridsize(const char* flagname, int value)
{
    if (value > 1 && value < 2<<16)
        { return true; }
    std::cout << "Invalid value for --" << flagname << ": "
              << value << std::endl;
    return false;
}

static bool validateSimulate(const char* flagname, int value)
{
    if (value >= 0)
        { return true; }
    std::cout << "Invalid value for --" << flagname << ": "
              << value << std::endl;
    return false;
}

DEFINE_int32(gridsize, 256, "resolution of the water wave.");
DEFINE_int32(simulate, 0, "execution steps, without graphic output. (0 = with graphic output)");
DEFINE_int32(wspf, 50, "wavesteps per frame.");
DEFINE_int32(kernelflops, 0, "Flops per kernel");

static const bool gridsize_dummy = google::RegisterFlagValidator(&FLAGS_gridsize, &validateGridsize);
static const bool simulate_dummy = google::RegisterFlagValidator(&FLAGS_simulate, &validateSimulate);
static const bool wspf_dummy = google::RegisterFlagValidator(&FLAGS_wspf, &validateWSPF);

int stepperframe;

void updateFunction(vertex* wave_vertices, rgb* wave_colors)
{
    computeNext(gridsize, gridsize, wave_vertices, wave_colors, stepperframe);
}

int main(int argc, char ** argv)
{
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    google::ParseCommandLineFlags(&argc, &argv, true);
    
    // checking command line arguments
    if (argc < 2)
    {
        std::cerr << "You have to specify a scene file." << std::endl;
        return -1;
    }
    
    CHECK_STRNE(argv[1], "") << "No scene file specified.";
    
    gridsize=FLAGS_gridsize;
    simulate=FLAGS_simulate;
    stepperframe = FLAGS_wspf;
    int kernelflops = FLAGS_kernelflops;

    rgb *colors_img;
    int colors_width;
    int colors_height;

    rgb *wave_img;
    int wave_width;
    int wave_height;

    float running_time;
    std::string landscape_filename, landscape_color_filename, wave_filename, colors_filename;
    parse_wavescene(argv[1], landscape_filename, landscape_color_filename, wave_filename, running_time);

    readPPM(wave_filename.c_str(), wave_img, wave_width, wave_height);
    readPPM(landscape_color_filename.c_str(), colors_img, colors_width, colors_height);
    
    rgb* landscapeheightmap;
    int heightmapheight;
    int heightmapwidth;
    readPPM(landscape_filename.c_str(), landscapeheightmap, heightmapwidth, heightmapheight);
    
    vertex *landscape;

    vertex *wave;
    float *waveheights;
    rgb *colors;



    createLandscapeRGB(landscapeheightmap, heightmapwidth, heightmapheight, gridsize, gridsize, landscape);

    createLandscapeRGB(wave_img, wave_width, wave_height, gridsize, gridsize, wave);

    createHeightData(wave_img, wave_width, wave_height, gridsize, gridsize, waveheights);

    createLandscapeColors(colors_img, colors_width, colors_height, gridsize, gridsize, colors);


    initWaterSurface(gridsize, gridsize, landscape, waveheights);
    
    if (simulate == 0)
    {
        createWindow(argc, argv, 800, 600, gridsize, gridsize, landscape, wave, colors, &updateFunction);
    }
    else
    {
        initTimer();
        for ( int step=0; step < simulate; step++)
        {
           updateFunction(wave, colors);
        }
        float runtime = timeSinceInit() / 1000.0f;
        
        long kernelcalls = simulate*stepperframe;
        long threads = gridsize*gridsize;
        long flops = kernelcalls*threads*kernelflops;
        
        std::cout << "Gridsize:" << gridsize << "x" << gridsize << std::endl;
        std::cout << "Launched the main kernel " << kernelcalls << " times." << std::endl;
        printf("Execution time: %2.4fs\n", runtime);
        if(flops > 0)
        {
            std::cout << "Total computed flops: " << flops << std::endl;
            printf("Speed: %6.2f GFlop/s\n", flops/runtime/1000000000);
        }
        
    }

    free(landscapeheightmap);
    free(wave_img);
    free(waveheights);
    free(colors_img);

    return 0;
}
