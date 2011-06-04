cmake_minimum_required(VERSION 2.8.0)

# for finding GLog
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(GLog REQUIRED)
include_directories(${GLOG_INCLUDE_DIR})

find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIR})

find_package(YAMLCPP REQUIRED)
include_directories(${YAMLCPP_INCLUDE_DIR})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})


IF(APPLE)
    include_directories(/System/Library/Frameworks)
ENDIF(APPLE)

include_directories("src")

set(CPU_SRC 
    src/main.cc
    src/ppm_reader.cc
    src/ppm_writer.cc
    src/3d_window.cc
    src/wavesimulator.cc
	src/timing.cc
    )
    
#cpu version
configure_file(src/wavesimulator.cu src/wavesimulator.cc COPYONLY)
add_executable(swecpu ${CPU_SRC})
set_target_properties(swecpu PROPERTIES COMPILE_FLAGS "-D__CPUVERSION__")
target_link_libraries(swecpu ${GLOG_LIBRARY})
target_link_libraries(swecpu ${GFLAGS_LIBRARY})
target_link_libraries(swecpu ${YAMLCPP_LIBRARY})
target_link_libraries(swecpu ${GLUT_LIBRARY})
target_link_libraries(swecpu ${OPENGL_LIBRARIES})


set(GPU_SRC 
    src/main.cc
    src/ppm_reader.cc
    src/ppm_writer.cc
    src/3d_window.cc
    src/wavesimulator.cu
	src/cudaInfo.cu
	src/timing.cc
    )

#cuda version
cuda_add_executable(swegpu ${GPU_SRC} OPTIONS -D__GPUVERSION__ --ptxas-options=-v -arch=sm_12)
set_target_properties(swegpu PROPERTIES COMPILE_FLAGS -D__GPUVERSION__)
set_target_properties(swegpu PROPERTIES LINKER_LANGUAGE C)
target_link_libraries(swegpu ${GLOG_LIBRARY})
target_link_libraries(swegpu ${GFLAGS_LIBRARY})
target_link_libraries(swegpu ${YAMLCPP_LIBRARY})
target_link_libraries(swegpu ${GLUT_LIBRARY})
target_link_libraries(swegpu ${OPENGL_LIBRARIES})

# common flags
SET(CMAKE_CXX_FLAGS "-Wall -Wno-unknown-pragmas -Wno-long-long -Wno-variadic-macros")
SET(CMAKE_CXX_FLAGS_RELEASE "-DNEXPENSIVE -O2 -g0")
SET(CMAKE_CXX_FLAGS_DEBUG "-DWARNINGS -O0 -g3 -fno-inline")

