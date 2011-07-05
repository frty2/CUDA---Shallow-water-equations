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
    if (value > 1 && value < 2 << 16)
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

int gridsize;
int simulate;
int stepperframe;
int kernelflops;

vertex *landscape;
vertex *wave;
float *waveheights;
rgb *colors;

int argc;
char ** argv;

void update(vertex* wave_vertices, rgb* wave_colors)
{
    computeNext(gridsize, gridsize, wave_vertices, wave_colors, stepperframe);
}

void shutdown()
{
    destroyWaterSurface();
    free(landscape);
    free(wave);
    free(waveheights);
    free(colors);
    exit(0);
}

void restart()
{
    destroyWaterSurface();
    initWaterSurface(gridsize, gridsize, landscape, waveheights);
}

void start()
{
    initWaterSurface(gridsize, gridsize, landscape, waveheights);

    if (simulate == 0)
    {
        createWindow(argc, argv, 800, 600, gridsize, gridsize, landscape, wave, colors, &update, &restart, &shutdown, stepperframe, kernelflops);
    }
    else
    {
        initTimer();
        for ( int step = 0; step < simulate; step++)
        {
            update(wave, colors);
        }
        float runtime = timeSinceInit() / 1000.0f;

        long kernelcalls = simulate * stepperframe;
        long threads = gridsize * gridsize;
        long flops = kernelcalls * threads * kernelflops;

        std::cout << "Gridsize:" << gridsize << "x" << gridsize << std::endl;
        std::cout << "Launched the main kernel " << kernelcalls << " times." << std::endl;
        printf("Execution time: %2.4fs\n", runtime);
        if(flops > 0)
        {
            std::cout << "Total computed flops: " << flops << std::endl;
            printf("Speed: %6.2f GFlop/s\n", flops / runtime / 1000000000);
        }

    }
}

int main(int ac, char ** av)
{
    argc = ac;
    argv = av;
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    google::ParseCommandLineFlags(&argc, &argv, true);

    gridsize = FLAGS_gridsize;
    simulate = FLAGS_simulate;
    stepperframe = FLAGS_wspf;
    kernelflops = FLAGS_kernelflops;

    // checking command line arguments
    if (argc < 2)
    {
        std::cerr << "You have to specify a scene file." << std::endl;
        return -1;
    }

    CHECK_STRNE(argv[1], "") << "No scene file specified.";

    float running_time;
    std::string landscape_filename, landscape_color_filename, wave_filename, colors_filename;
    parse_wavescene(argv[1], landscape_filename, landscape_color_filename, wave_filename, running_time);


    rgb *colors_img;
    int colors_width;
    int colors_height;

    rgb *wave_img;
    int wave_width;
    int wave_height;

    int heightmapheight;
    int heightmapwidth;


    readPPM(wave_filename.c_str(), wave_img, wave_width, wave_height);

    readPPM(landscape_color_filename.c_str(), colors_img, colors_width, colors_height);

    int landscape_filename_lenght = landscape_filename.length();
    std::string filetype;

    if (landscape_filename_lenght > 4)
        { filetype = landscape_filename.substr(landscape_filename_lenght - 4, landscape_filename_lenght - 1); }

    if (filetype.compare(".asc") == 0 )
    {
        float* heightmap;
        readASC(landscape_filename.c_str(), heightmapwidth, heightmapheight, heightmap);
        createLandscapeFloat(heightmap, heightmapwidth, heightmapheight, gridsize, gridsize, landscape);
        free(heightmap);
    }
    if (filetype.compare(".ppm") == 0 )
    {
        rgb* landscape_img;
        readPPM(landscape_filename.c_str(), landscape_img, heightmapwidth, heightmapheight);
        createLandscapeFromRGB(landscape_img, heightmapwidth, heightmapheight, gridsize, gridsize, landscape);
        free(landscape_img);
    }

    createLandscapeFromRGB(wave_img, wave_width, wave_height, gridsize, gridsize, wave);

    createWaveHeights(wave_img, wave_width, wave_height, gridsize, gridsize, waveheights);

    createLandscapeColors(colors_img, landscape, colors_width, colors_height, gridsize, gridsize, colors);

    free(colors_img);
    free(wave_img);

    start();

    return 0;
}
