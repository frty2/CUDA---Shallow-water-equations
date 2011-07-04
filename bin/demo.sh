#!/bin/bash
./swegpu ../res/landscape.yaml -kernelflops 256
./swegpu ../res/landscape.yaml -gridsize 128 -kernelflops 256
./swegpu ../res/landscape.yaml -wspf 200 -kernelflops 256
./swegpu ../res/ca.yaml -roty 225 -rotx 15 -zoom 15 -wspf 20 -kernelflops 256
./swegpu ../res/ca.yaml -rotx 8 -zoom 8 -roty 350 -wspf 20 -kernelflops 256
./swegpu ../res/dam.yaml -kernelflops 256
./swegpu ../res/dam.yaml -wspf 20 -mode 1 -rotx 25 -zoom 13 -roty 245 -kernelflops 256
./swegpu ../res/dam.yaml -wspf 20 -mode 2 -rotx 11 -zoom 14 -roty 105 -kernelflops 256
./swegpu ../res/dam.yaml -simulate 1 -wspf 2000 -gridsize 768 -kernelflops 256
