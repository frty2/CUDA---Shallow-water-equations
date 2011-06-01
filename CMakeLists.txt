cmake_minimum_required(VERSION 2.8.0)

# for finding GLog
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# find_package(GTest REQUIRED)
# include_directories(${GTEST_INCLUDE_DIR})

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

if(APPLE)
    include_directories(/System/Library/Frameworks)
ENDIF(APPLE)

include_directories("src")

set(CPU_SRC 
    src/main.cc
    src/parser.cc
    src/raytracer.cc
    src/ppm_writer.cc
    src/types.cc
    src/mainwindow.cc
    src/postprocessing.cc)
    
#cpu version
configure_file(src/raytracer.cu src/raytracer.cc COPYONLY)
configure_file(src/postprocessing.cu src/postprocessing.cc COPYONLY)
add_executable(rtcpu ${CPU_SRC})
set_target_properties(rtcpu PROPERTIES COMPILE_FLAGS "-D__CPUVERSION__")
target_link_libraries(rtcpu ${GLOG_LIBRARY})
target_link_libraries(rtcpu ${GFLAGS_LIBRARY})
target_link_libraries(rtcpu ${YAMLCPP_LIBRARY})
target_link_libraries(rtcpu ${GLUT_LIBRARY})
target_link_libraries(rtcpu ${OPENGL_gl_LIBRARY})

set(GPU_SRC 
    src/main.cc
    src/parser.cc
    src/raytracer.cu
    src/ppm_writer.cc
    src/types.cc
    src/mainwindow.cc
    src/postprocessing.cu)

#cuda version
cuda_add_executable(rtgpu ${GPU_SRC} OPTIONS -D__GPUVERSION__ --ptxas-options=-v --maxrregcount 48 -arch=sm_12)
set_target_properties(rtgpu PROPERTIES COMPILE_FLAGS -D__GPUVERSION__)
set_target_properties(rtgpu PROPERTIES LINKER_LANGUAGE C)
target_link_libraries(rtgpu ${GLOG_LIBRARY})
target_link_libraries(rtgpu ${GFLAGS_LIBRARY})
target_link_libraries(rtgpu ${YAMLCPP_LIBRARY})
target_link_libraries(rtgpu ${GLUT_LIBRARY})
target_link_libraries(rtgpu ${OPENGL_gl_LIBRARY})

# common flags
SET(CMAKE_CXX_FLAGS "-Wall -Wno-unknown-pragmas -Wno-long-long -Wno-variadic-macros")
SET(CMAKE_CXX_FLAGS_RELEASE "-DNEXPENSIVE -O2 -g0")
SET(CMAKE_CXX_FLAGS_DEBUG "-DWARNINGS -O0 -g3 -fno-inline")

