#include "timing.h"

#include <sys/time.h>

 // in milliseconds
long inittime;
long marktime;

long getTime()
{
    timeval timestamp;
    gettimeofday(&timestamp, 0);
    return (timestamp.tv_sec * 1000000 + timestamp.tv_usec) / 1000;
}

void initTimer()
{
    inittime = getTime();
    marktime = getTime();
}

void markTime()
{
    marktime = getTime();
}

long timeSinceMark()
{
    long currenttime = getTime();
    return currenttime - marktime;
}

long timeSinceInit()
{
    long currenttime = getTime();
    return currenttime - inittime;
}



