#include "3d_window.h"

#include <iostream>
#include <stdlib.h>
#include <glog/logging.h>
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

#define ESC 27
#define KEY_A 97
#define KEY_D 100
#define KEY_W 119
#define KEY_S 115
#define KEY_R 114
#define KEY_F 102

GLuint heightmap[2];
GLuint watersurface[2];
GLuint indexbufferID;

void (*callback)() = NULL;

int width, height;

int window_width;
int window_height;


void initScene(int width, int height, rgb *heightmap_img, rgb *color_img);
void initGlut(int argc, char ** argv);
void initGL();

void resize();
void animate(int v);
void keypressed(unsigned char key, int x, int y);
void drawString(int x, int y, const std::string &text, void *font = GLUT_BITMAP_HELVETICA_12);
void drawBorder();


int frame = 0;
float rotationY = 0;
float rotationX = 10;
float zoom = 10;

timeval l_time, c_time;
long current_time, last_time, fpsupdate_time;
float fps, timediff;

void createWindow(int argc, char **argv, int width, int height, void (*cb)(), int vertex_width, int vertex_height, rgb *heightmap_img, rgb *color_img)
{
    callback = cb;
    window_width = width;
    window_height = height;

    initGlut(argc, argv);
    initGL();

    initScene(vertex_width, vertex_height, heightmap_img, color_img);

    glutMainLoop();
}

void paint()
{
    frame++;
    gettimeofday(&c_time, 0);
    long current_time = c_time.tv_sec * 1000000 + c_time.tv_usec;
    long last_time = l_time.tv_sec * 1000000 + l_time.tv_usec;

    timediff = (current_time - last_time ) / 1000.0f;

    l_time = c_time;

    if(current_time - fpsupdate_time > 1000000)
    {
        fps = 1000.0f / timediff;
        fpsupdate_time = current_time;
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    std::stringstream fps_text;

    fps_text << "FPS: " << fps;
    glMatrixMode(GL_MODELVIEW);

    glPushMatrix();
    glLoadIdentity();


    glTranslatef(0.0f, -5.0f, 0.0f);
    glTranslatef(0, -rotationX / 4.0f, -zoom);
    glRotatef(rotationX, 1.0f, 0.0f, 0.0f);

    glTranslatef(0.0f, 0.0f, -20.0f);
    glRotatef(rotationY, 0.0f, 1.0f, 0.0f);


    glEnableClientState(GL_INDEX_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, heightmap[1]);
    glColorPointer(3, GL_UNSIGNED_BYTE, sizeof(rgb), 0);

    glBindBuffer(GL_ARRAY_BUFFER, heightmap[0]);
    glVertexPointer(3, GL_FLOAT, 12, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexbufferID);
    glIndexPointer(GL_INT, 0, 0);

    //glDrawElements( GL_QUADS, 4 * (width - 1) * (height - 1), GL_UNSIGNED_INT, 0 );

    glBindBuffer(GL_ARRAY_BUFFER, watersurface[1]);
    glColorPointer(3, GL_UNSIGNED_BYTE, sizeof(rgb), 0);

    glBindBuffer(GL_ARRAY_BUFFER, watersurface[0]);
    glVertexPointer(3, GL_FLOAT, 12, 0);

    glDrawElements( GL_QUADS, 4 * (width - 1) * (height - 1), GL_UNSIGNED_INT, 0 );

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_INDEX_ARRAY);

    //drawBorder();

    glPopMatrix();

    drawString(5, 25, fps_text.str());
    glutSwapBuffers();
}

void updateScene()
{
    glBindBuffer(GL_ARRAY_BUFFER, watersurface[0]);
    vertex *watersurfacevertices = (vertex *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    CHECK_NOTNULL(watersurfacevertices);
    glUnmapBuffer(GL_ARRAY_BUFFER);
    
    glBindBuffer(GL_ARRAY_BUFFER, watersurface[1]);
    rgb *watersurfacecolors = (rgb *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    CHECK_NOTNULL(watersurfacecolors);
    glUnmapBuffer(GL_ARRAY_BUFFER);
    
    computeNext(0, width, height, watersurfacevertices, watersurfacecolors);
}

void initScene(int w, int h, rgb *heightmap_img, rgb *colors)
{
    width = w;
    height = h;

    //Calc the heightmap out of the rgb
    vertex *vertices = (vertex *) malloc(width * height * sizeof(vertex));
    CHECK_NOTNULL(vertices);

    for(int y = 0; y < height; y ++)
    {
        for(int x = 0; x < width; x ++)
        {
            vertex v;
            v.x = x * 16.0f / width - 8;
            v.z = y * 16.0f / height - 8;
            v.y = heightmap_img[y * width + x].x / 256.0f + heightmap_img[y * width + x].y / 256.0f + heightmap_img[y * width + x].z / 256.0f;

            vertices[y * width + x] = v;
        }
    }


    //Calc the indices for the QUADS
    int *indices = (int *) malloc(4 * (width - 1) * (height - 1) * sizeof(int));
    CHECK_NOTNULL(indices);

    for(int y = 0; y < height - 1; y++)
    {
        for(int x = 0; x < width - 1; x++)
        {
            indices[4 * ( y * (width - 1) + x ) + 0] = (y + 1) * width + x;
            indices[4 * ( y * (width - 1) + x ) + 1] = (y + 1) * width + x + 1;
            indices[4 * ( y * (width - 1) + x ) + 2] = y * width + x + 1;
            indices[4 * ( y * (width - 1) + x ) + 3] = y * width + x;
        }
    }

    vertex *watervertices = (vertex *) malloc(width * height * sizeof(vertex));
    CHECK_NOTNULL(watervertices);

    rgb* watercolors = (rgb *) malloc(width * height * sizeof(rgb));
    CHECK_NOTNULL(watercolors);


    glGenBuffers(2, &heightmap[0]);

    glBindBuffer(GL_ARRAY_BUFFER, heightmap[1]);
    glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(rgb), colors, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, heightmap[0]);
    glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(vertex), vertices, GL_STATIC_DRAW);

    glGenBuffers(2, &watersurface[0]);

    glBindBuffer(GL_ARRAY_BUFFER, watersurface[1]);
    glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(rgb), watercolors, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, watersurface[0]);
    glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(vertex), watervertices, GL_DYNAMIC_COPY);

    glGenBuffers(1, &indexbufferID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexbufferID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * (width - 1) * (height - 1)*sizeof(int), indices, GL_STATIC_DRAW);
    
    /*
     * TEMP
     */
    initWaterSurface(width, height, vertices);

    free(vertices);
    free(indices);
    free(colors);
    free(watercolors);
}

void animate(int v)
{
    if(callback != NULL)
    {
        callback();
    }

    updateScene();

    glutPostRedisplay();

    glutTimerFunc(0, animate, 0);
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
        zoom = --zoom < -10 ? -10 : zoom;
    }
    if(key == KEY_S)
    {
        zoom = ++zoom > 25 ? 25 : zoom;
    }
    if(key == KEY_R)
    {
        rotationX = ++rotationX > 90 ? 90 : rotationX;
    }
    if(key == KEY_F)
    {
        rotationX = --rotationX < 0 ? 0 : rotationX;
    }
}

void initGL()
{
    glMatrixMode(GL_PROJECTION);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glLoadIdentity();
    glViewport(0, 0, window_width, window_height);
    gluPerspective(45, 1.0 * window_width / window_height, 1, 1000);
    glMatrixMode(GL_MODELVIEW);

    glClearColor (0.0, 0.0, 0.0, 0.0);
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

    glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
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


void drawBorder()
{
    glColor3f(1.0f, 1.0f, 1.0f);
    
    glBegin(GL_QUADS);
    glVertex3f(-8.0f, 0.0f, -8.0f);
    glVertex3f(-8.0f, 2.0f, -8.0f);
    glVertex3f(8.0f, 2.0f, -8.0f);
    glVertex3f(8.0f, 0.0f, -8.0f);

    glVertex3f(-8.0f, 0.0f, -8.0f);
    glVertex3f(-8.0f, 2.0f, -8.0f);
    glVertex3f(-8.0f, 2.0f, 8.0f);
    glVertex3f(-8.0f, 0.0f, 8.0f);

    glVertex3f(8.0f, 0.0f, 8.0f);
    glVertex3f(8.0f, 2.0f, 8.0f);
    glVertex3f(8.0f, 2.0f, -8.0f);
    glVertex3f(8.0f, 0.0f, -8.0f);

    glVertex3f(8.0f, 0.0f, 8.0f);
    glVertex3f(8.0f, 2.0f, 8.0f);
    glVertex3f(-8.0f, 2.0f, 8.0f);
    glVertex3f(-8.0f, 0.0f, 8.0f);
    glEnd();
}