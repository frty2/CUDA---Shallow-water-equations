#pragma once

#include <yaml-cpp/yaml.h>

#include "types.h"

void parse_wavescene(const char* filename, std::string &landscape_filename, std::string &landscape_color_filename,
                     std::string &threshhold_filename, std::string &wave_filename, float &running_time);
