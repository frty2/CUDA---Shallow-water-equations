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


#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

static bool validateMaxFPS(const char* flagname, int value)
{
    if (value > 0)
        { return true; }
    std::cout << "Invalid value for --" << flagname << ": "
              << value << std::endl;
    return false;
}

DEFINE_int32(maxfps, 30, "maximum frames per second.");

static const bool maxfps_dummy = google::RegisterFlagValidator(&FLAGS_maxfps, &validateMaxFPS);

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
void mousepressed(int button, int state, int x, int y);
void drawString(int x, int y, const std::string &text, void *font = GLUT_BITMAP_HELVETICA_12);
void drawBorder();


int frame = 0;
float rotationY = 0;
float rotationX = 45;
float zoom = 30;

float fps;
long fps_update_time;

void (*callback) (vertex*, rgb*) = NULL;

void createWindow(int argc, char **argv, int w_width, int w_height,
                  int grid_width, int grid_height, vertex *landscape, vertex *wave, rgb *colors,
                  void (*updatefunction) (vertex*, rgb*))
{
    callback = updatefunction;
    CHECK_NOTNULL(callback);

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
    std::stringstream fps_text;

    fps_text << "FPS: " << fps << " Frames total: " << frame;
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
    glDrawElements( GL_QUADS, 4 * (width - 1) * (height - 1), GL_UNSIGNED_INT, 0 );


    glBindBuffer(GL_ARRAY_BUFFER, watersurface[1]);
    glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(rgb), 0);

    glBindBuffer(GL_ARRAY_BUFFER, watersurface[0]);
    glVertexPointer(3, GL_FLOAT, sizeof(vertex), 0);


    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //watersurface
    glDrawElements( GL_QUADS, 4 * (width - 1) * (height - 1), GL_UNSIGNED_INT, 0 );

    glDisable(GL_BLEND);

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_INDEX_ARRAY);

    glPopMatrix();

    drawString(5, 25, fps_text.str());
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

    free(indices);
    free(watercolors);
    free(landscape);
    free(colors);
    free(wave);
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
        callback(watersurfacevertices, watersurfacecolors);

    glutPostRedisplay();

    long elapsed = timeSinceMark();

    glutTimerFunc( max(0, 1000.0 / FLAGS_maxfps - elapsed) , animate, 0);
}

void keypressed(unsigned char key, int x, int y)
{
    if(key == ESC)
    {
        exit(0);
    }
    if(key == KEY_A)
    {
        rotationY += 1;
    }
    if(key == KEY_D)
    {
        rotationY -= 1;
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
        rotationX = ++rotationX > 90 ? 90 : rotationX;
    }
    if(key == KEY_F)
    {
        rotationX = --rotationX < -45 ? -45 : rotationX;
    }
}

void initGL()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, window_width, window_height);
    gluPerspective(45, 1.0 * window_width / window_height, 1, 1000);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);
    glClearColor (0.0, 0.0, 0.0, 1.0);
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
    glutMouseFunc(mousepressed);
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
