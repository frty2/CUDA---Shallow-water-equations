#include "3d_window.h"

#include <iostream>
#include <stdlib.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <string>
#include <sstream>
#include <sys/time.h>

#if __APPLE__
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glut.h>
#include <GL/glext.h>
#endif

#include "wavesimulator.h"
#include "types.h"
#include "math.h"
#include "timing.h"

#define ESC 27
#define KEY_A 97
#define KEY_D 100
#define KEY_W 119
#define KEY_S 115
#define KEY_R 114
#define KEY_F 102
#define KEY_X 120
#define KEY_M 109
#define KEY_H 104


#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

const int SURFACE_MODE = 0;
const int WIREFRAME_MODE = 1;
const int POINT_MODE = 2;

const int MAXMODE = 2;

int mode = SURFACE_MODE;

static bool validateMaxFPS(const char* flagname, int value)
{
    if (value > 0)
        { return true; }
    std::cout << "Invalid value for --" << flagname << ": "
              << value << std::endl;
    return false;
}

static bool validateMode(const char* flagname, int value)
{
    if (value >= 0 && value <= MAXMODE)
        { return true; }
    std::cout << "Invalid value for --" << flagname << ": "
              << value << std::endl;
    return false;
}

static bool validateAngle(const char* flagname, double value)
{
    if (value >= 0 && value <= 360)
        { return true; }
    std::cout << "Invalid value for --" << flagname << ": "
              << value << std::endl;
    return false;
}

static bool validateZoom(const char* flagname, double value)
{
    if (value >= 5 && value <= 50)
        { return true; }
    std::cout << "Invalid value for --" << flagname << ": "
              << value << std::endl;
    return false;
}

DEFINE_double(rotx, 45, "rotation x-axis");
DEFINE_double(roty, 0, "rotation y-axis");
DEFINE_double(zoom, 30, "distance from scene");
DEFINE_int32(maxfps, 30, "maximum frames per second.");
DEFINE_int32(mode, SURFACE_MODE, "display mode: surface = 0, wireframe = 1, points = 2");

static const bool maxfps_dummy = google::RegisterFlagValidator(&FLAGS_maxfps, &validateMaxFPS);
static const bool mode_dummy = google::RegisterFlagValidator(&FLAGS_mode, &validateMode);
static const bool rotx_dummy = google::RegisterFlagValidator(&FLAGS_rotx, &validateAngle);
static const bool roty_dummy = google::RegisterFlagValidator(&FLAGS_roty, &validateAngle);
static const bool zoom_dummy = google::RegisterFlagValidator(&FLAGS_zoom, &validateZoom);

GLuint heightmap[2];
GLuint watersurface[2];
GLuint indexbufferID;

int width, height;

int window_width;
int window_height;

void initScene(vertex *landscape, vertex *wave, rgb *colors, int grid_width, int grid_height);
void initGlut(int argc, char ** argv);
void initGL();

void resize();
void animate(int v);
void keypressed(unsigned char key, int x, int y);
void drawString(int x, int y, const std::string &text, void *font = GLUT_BITMAP_HELVETICA_18);
void drawBorder();


int frame = 0;
float rotationY = 0;
float rotationX = 45;
float zoom = 30;

bool show_landscape = true;

int wavesteps_, kernelflops_;
float fps;
long fps_update_time;

void (*update) (vertex*, rgb*) = NULL;
void (*restart) () = NULL;
void (*shutdown) () = NULL;

long gpumem = 0, memtrans = 0;

void createWindow(int argc, char **argv, int w_width, int w_height,
                  int grid_width, int grid_height, vertex *landscape, vertex *wave, rgb *colors,
                  void (*updatefunction) (vertex*, rgb*),
                  void (*restartfunction) (),
                  void (*shutdownfunction) (),
                  int ws, int kf)
{
    update = updatefunction;
    restart = restartfunction;
    shutdown = shutdownfunction;

    wavesteps_ = ws;
    kernelflops_ = kf;

    CHECK_NOTNULL(update);
    CHECK_NOTNULL(restart);
    CHECK_NOTNULL(shutdown);

    initTimer();
    window_width = w_width;
    window_height = w_height;

    initGlut(argc, argv);
    initGL();

    initScene(landscape, wave, colors, grid_width, grid_height);

    glutMainLoop();
}

void paint()
{
    frame++;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);

    glPushMatrix();
    glLoadIdentity();

    glTranslatef(0.0f, -5.0f, 0.0f);

    glTranslatef(0.0f, 0.0f, -zoom);
    glRotatef(rotationX, 1.0f, 0.0f, 0.0f);
    glRotatef(rotationY, 0.0f, 1.0f, 0.0f);


    glEnableClientState(GL_INDEX_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, heightmap[1]);
    glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(rgb), 0);

    glBindBuffer(GL_ARRAY_BUFFER, heightmap[0]);
    glVertexPointer(3, GL_FLOAT, sizeof(vertex), 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexbufferID);
    glIndexPointer(GL_INT, 0, 0);

    //underground heightmap
    if(show_landscape)
    {
        glDrawElements( GL_QUADS, 4 * (width - 1) * (height - 1), GL_UNSIGNED_INT, 0 );
    }


    glBindBuffer(GL_ARRAY_BUFFER, watersurface[1]);
    glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(rgb), 0);

    glBindBuffer(GL_ARRAY_BUFFER, watersurface[0]);
    glVertexPointer(3, GL_FLOAT, sizeof(vertex), 0);

    glTranslatef(0.0f, -0.12f, 0.0f);
    
    //watersurface
    switch(mode)
    {
    case SURFACE_MODE:
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDrawElements( GL_QUADS, 4 * (width - 1) * (height - 1), GL_UNSIGNED_INT, 0 );
        glDisable(GL_BLEND);
        break;
    case WIREFRAME_MODE:
        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
        glDrawElements( GL_QUADS, 4 * (width - 1) * (height - 1), GL_UNSIGNED_INT, 0 );
        glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
        break;
    case POINT_MODE:
        glDrawElements( GL_POINTS, 4 * (width - 1) * (height - 1), GL_UNSIGNED_INT, 0 );
        break;
    }

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_INDEX_ARRAY);

    glPopMatrix();

    std::stringstream fps_text, mode_text, grid_text, gpumem_text,
        memtrans_text, mflops_text;
    fps_text << "FPS: " << fps << " Frames total: " << frame;
    mode_text << "Display Mode: ";
    switch(mode)
    {
    case SURFACE_MODE:
        mode_text << "surface";
        break;
    case WIREFRAME_MODE:
        mode_text << "wireframes";
        break;
    case POINT_MODE:
        mode_text << "points";
        break;
    }

    grid_text << "Gridsize: " << width << "x" << height;
    if(gpumem >= 10000000)
    {
        gpumem_text << "Cuda memory usgae: " << (gpumem) / 1000000 << " Mb";
    }
    else
    {
        gpumem_text << "Cuda memory usgae: " << (gpumem) / 1000 << " Kb";
    }
    memtrans_text << "Memory transfer: " << int(fps * memtrans) / 1000000 << " Mb/s";

    long mflops = long(fps * width * height * kernelflops_ * wavesteps_ ) / 1000000;

    if(mflops >= 10000)
    {
        mflops_text << "Speed: " << mflops / 1000 << " GFlop/s";
    }
    else
    {
        mflops_text << "Speed: " << mflops << " MFlop/s";
    }
    drawString(5, 20, fps_text.str());
    drawString(5, 40, mode_text.str());
    drawString(5, 60, grid_text.str());
    if(frame > 100)
    {
        drawString(5, 80, gpumem_text.str());
        drawString(5, 100, memtrans_text.str());

        if(mflops)
        {
            drawString(5, 140, mflops_text.str());
        }
    }
    glutSwapBuffers();
}

void initScene(vertex *landscape, vertex *wave, rgb *colors, int grid_width, int grid_height)
{
    width = grid_width;
    height = grid_height;

    //Calc the indices for the QUADS
    int *indices = (int *) malloc(4 * (width - 1) * (height - 1) * sizeof(int));
    CHECK_NOTNULL(indices);

    for(int y = 0; y < grid_height - 1; y++)
    {
        for(int x = 0; x < grid_width - 1; x++)
        {
            indices[4 * ( y * (width - 1) + x ) + 0] = (y + 1) * width + x;
            indices[4 * ( y * (width - 1) + x ) + 1] = (y + 1) * width + x + 1;
            indices[4 * ( y * (width - 1) + x ) + 2] = y * width + x + 1;
            indices[4 * ( y * (width - 1) + x ) + 3] = y * width + x;
        }
    }

    rgb* watercolors = (rgb *) malloc(width * height * sizeof(rgb));
    CHECK_NOTNULL(watercolors);

    glGenBuffers(2, &heightmap[0]);

    glBindBuffer(GL_ARRAY_BUFFER, heightmap[1]);
    glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(rgb), colors, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, heightmap[0]);
    glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(vertex), landscape, GL_STATIC_DRAW);

    glGenBuffers(2, &watersurface[0]);

    glBindBuffer(GL_ARRAY_BUFFER, watersurface[1]);
    glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(rgb), watercolors, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, watersurface[0]);
    glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(vertex), wave, GL_DYNAMIC_COPY);

    glGenBuffers(1, &indexbufferID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexbufferID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * (width - 1) * (height - 1)*sizeof(int), indices, GL_STATIC_DRAW);

    gpumem += 2 * (width + 1) * (height + 1) * sizeof(gridpoint); //device grids
    gpumem += width * height * sizeof(vertex); //topography
    gpumem += width * height * sizeof(vertex); //watersurface
    gpumem += width * height * sizeof(rgb); //watercolors
    gpumem += width * height * sizeof(float); //wave to add

    memtrans += 2 * width * height * sizeof(vertex); //watersurface
    memtrans += 2 * width * height * sizeof(rgb); //watercolors

    free(indices);
    free(watercolors);
}

void animate(int v)
{
    long frametime = timeSinceMark();

    long currenttime = timeSinceInit();
    if(currenttime - fps_update_time > 1000)
    {
        fps_update_time = currenttime;
        fps = 1000 / frametime;
    }

    markTime();

    glBindBuffer(GL_ARRAY_BUFFER, watersurface[0]);
    vertex *watersurfacevertices = (vertex *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    CHECK_NOTNULL(watersurfacevertices);
    glUnmapBuffer(GL_ARRAY_BUFFER);

    glBindBuffer(GL_ARRAY_BUFFER, watersurface[1]);
    rgb *watersurfacecolors = (rgb *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    CHECK_NOTNULL(watersurfacecolors);
    glUnmapBuffer(GL_ARRAY_BUFFER);

    if(frame == 1 || frame > 100)
        { update(watersurfacevertices, watersurfacecolors); }

    glutPostRedisplay();

    long elapsed = timeSinceMark();

    glutTimerFunc( max(0, 1000.0 / FLAGS_maxfps - elapsed) , animate, 0);
}

void keypressed(unsigned char key, int x, int y)
{
    if(key == ESC)
    {
        shutdown();
    }
    if(key == KEY_X)
    {
        frame = 0;
        restart();
    }
    if(key == KEY_M)
    {
        mode = (mode + 1) % (MAXMODE + 1);
    }
    if(key == KEY_H)
    {
        show_landscape = !show_landscape;
    }
    if(key == KEY_A)
    {
        rotationY = ++rotationY > 360 ? 0 : rotationY;
    }
    if(key == KEY_D)
    {
        rotationY = --rotationY < 0 ? 360 : rotationY;
    }
    if(key == KEY_W)
    {
        zoom = --zoom < 5 ? 5 : zoom;
    }
    if(key == KEY_S)
    {
        zoom = ++zoom > 50 ? 50 : zoom;
    }
    if(key == KEY_R)
    {
        rotationX = ++rotationX > 360 ? 0 : rotationX;
    }
    if(key == KEY_F)
    {
        rotationX = --rotationX < 0 ? 360 : rotationX;
    }
}

void initGL()
{
    mode = FLAGS_mode;
    rotationX = FLAGS_rotx;
    rotationY = FLAGS_roty;
    zoom = FLAGS_zoom;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, window_width, window_height);
    gluPerspective(45, 1.0 * window_width / window_height, 1, 1000);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);
    glClearColor (0.0, 0.0, 0.0, 1.0);
    glPointSize(2.0f);
}

void resize(int width, int height)
{
    window_width = width;
    window_height = height;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, window_width, window_height);
    gluPerspective(45, 1.0 * window_width / window_height, 1, 1000);
    glMatrixMode(GL_MODELVIEW);
}

void initGlut(int argc, char ** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode (GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow (argv[0]);

    glutDisplayFunc(paint);
    glutKeyboardFunc(keypressed);
    glutReshapeFunc(resize);
    glutTimerFunc(0, animate, 0);
}

void drawString(int x, int y, const std::string &text, void *font)
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, window_width, window_height, 0);

    glRasterPos2i(x, y);

    glColor3f(1.0f, 1.0f, 1.0f);
    int i = 0;
    while (text[i] != '\0')
    {
        glutBitmapCharacter(font, text[i++]);
    }
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}
