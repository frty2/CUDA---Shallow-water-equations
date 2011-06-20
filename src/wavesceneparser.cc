#include <glog/logging.h>
#include <fstream>
#include "wavesceneparser.h"
#include <stdio.h>
#include <stdlib.h>

void parse_wavescene(const char* filename, std::string &landscape_filename, std::string &landscape_color_filename, 
   std::string &threshhold_filename, std::string &wave_filename, float &running_time)
{
    std::ifstream wavescene(filename);
	if ( wavescene.is_open() )
	{
		YAML::Parser parser(wavescene);
		YAML::Node doc;
		parser.GetNextDocument(doc);
		const YAML::Node *running_time_node = doc.FindValue("running_time");
        if ( running_time_node != NULL )
        {
            (*running_time_node) >> running_time;
        }
        else
        {   
            running_time = 30.0f; 
            LOG(INFO) << "No running_time definied. Running_time was set to 30.";
        }	
        const YAML::Node *landscape_node = doc.FindValue("landscape_filename");
        if ( landscape_node != NULL )
        {
            (*landscape_node) >> landscape_filename;
        }
        else
        {   
            landscape_filename = "landscape.ppm"; 
            LOG(INFO) << "No landscape_filename definied. Landscape_filename was set to 'landscape.ppm'.";
        }
        
        const YAML::Node *landscape_color_node = doc.FindValue("landscape_color_filename");
        if ( landscape_color_node != NULL )
        {
            (*landscape_color_node) >> landscape_color_filename;
        }
        else
        {   
            landscape_color_filename = "landscape_color.ppm"; 
            LOG(INFO) << "No landscape_color_filename definied. Landscape_color_filename was set to 'landscape_color.ppm'.";
        }
        
        
        
        
        const YAML::Node *threshhold_node = doc.FindValue("threshhold_filename");
        if ( threshhold_node != NULL )
        {
            (*threshhold_node) >> threshhold_filename;
        }
        else
        {   
            threshhold_filename = "threshhold.ppm"; 
            LOG(INFO) << "No threshhold_filename definied. Threshhold_filename was set to 'threshhold.ppm'.";
        }
        const YAML::Node *wave_node = doc.FindValue("wave_filename");
        if ( wave_node != NULL )
        {
            (*wave_node) >> wave_filename;
        }
        else
        {   
            wave_filename = "wave.ppm"; 
            LOG(INFO) << "No wave_filename definied. Wave_filename was set to 'wave.ppm'.";
        }
	}
	else
	{
		LOG(ERROR) << "Datei nicht gefunden.";
		exit (1);
	}
}
