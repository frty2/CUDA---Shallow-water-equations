#include <glog/logging.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

double getTime()
    {
	timeval start;
	gettimeofday(&start, 0);
	double tS = start.tv_sec*1000000 + (start.tv_usec);
	return tS;
	}
void getDif(double& tS, int& sec, int& millisec)
	{
	//if (tS != NULL && sec != NULL && millisec != NULL)
	//	{
		timeval end;
		double tE = end.tv_sec*1000000  + (end.tv_usec);
		long diff = long(tE - tS);	
		sec = (int)diff/1000000;
		millisec = (int)(diff%1000000)/1000;
	//	}
	}