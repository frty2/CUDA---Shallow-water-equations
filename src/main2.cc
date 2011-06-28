#include <iostream>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <stdio.h>

#include "stdlib.h"
#include "landscapereader.h"

static bool validateFitsize(const char* flagname, int value)
{
    if (value >= 0)
        { return true; }
    std::cout << "Invalid value for --" << flagname << ": "
              << value << std::endl;
    return false;
}

DEFINE_int32(fitsize, 512, "size for new file.");

static const bool fitsize = google::RegisterFlagValidator(&FLAGS_fitsize, &validateFitsize);

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
    
    CHECK_STRNE(argv[1], "") << "No source file specified.";
    CHECK_STRNE(argv[2], "") << "No destination file specified.";
    
    float* landscapeheightmap;
    int heightmapheight;
    int heightmapwidth;
    readASC(argv[1], heightmapwidth, heightmapheight, landscapeheightmap);
    
    int fitsize = FLAGS_fitsize;
    fitASC(argv[2], heightmapwidth, fitsize, landscapeheightmap);
    

    return 0;
}
