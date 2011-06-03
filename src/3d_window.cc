#include "3d_window.h"

#include <iostream>
#include <stdlib.h>
#include <glog/logging.h>

#if __APPLE__
    #include <GLUT/glut.h>
    #include <OpenGL/gl.h>
#else
    #include <GL/gl.h>
    #include <GL/glut.h>
#endif

#include "types.h"
#define ESC 27

int vertcount;
int w1;
int h1;
vertex *vertices;
void (*callback)() = NULL;


void initScene(int width, int height, vertex *heightmap, int vertcount);
void initGlut(int argc, char ** argv, int width, int height);
void initGL(int width, int height);

void resize(int width, int height);
void animate(int v);
void keypressed(unsigned char key, int x, int y);


void paint()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    int w = w1;
    int h = h1;
    glPushMatrix();
        glLoadIdentity();
        //glRotatef(1, 1, 1,0);
        glTranslatef(0, 0, -5);
        
        glColor3f(1,0,0);
        
        glBegin(GL_POINTS);
        for(int y = 0;y < h - 1;y++)
        {
            for(int x = 0;x < w - 1;x++)
            {
                glVertex3f(vertices[4*(y*(w-1)+x)].x, vertices[4*(y*(w-1)+x)].y, vertices[4*(y*(w-1)+x)].z);
                glVertex3f(vertices[4*(y*(w-1)+x)+1].x, vertices[4*(y*(w-1)+x)+1].y, vertices[4*(y*(w-1)+x)+1].z);
                glVertex3f(vertices[4*(y*(w-1)+x)+2].x, vertices[4*(y*(w-1)+x)+2].y, vertices[4*(y*(w-1)+x)+2].z);
                glVertex3f(vertices[4*(y*(w-1)+x)+3].x, vertices[4*(y*(w-1)+x)+3].y, vertices[4*(y*(w-1)+x)+3].z);
            }
        }
        glEnd();
            
    glPopMatrix();

    
    glutSwapBuffers();
}

void createWindow(int argc, char **argv, int width, int height, void (*cb)(), vertex *heightmap, int vertc)
{
    callback = cb;
    vertcount = vertc;
    vertices = heightmap;
    w1 = width;
    h1 = height;
    
    initGlut(argc, argv, width, height);
    initGL(width, height);
    
    initScene(width, height, heightmap, vertcount);
    
    glutMainLoop();
}

void initScene(int width, int height, vertex *heightmap, int vertcount)
{
    
    //glGenBuffersARB(1, &vertexbufferID);
    //glBindBufferARB(GL_ARRAY_BUFFER_ARB, vertexbufferID);
    //glBufferDataARB(GL_ARRAY_BUFFER_ARB, vertcount*sizeof(vertex), heightmap, GL_STATIC_COPY_ARB);
}

void animate(int v)
{
    glutTimerFunc(50, animate, 0);
    callback();
    glutPostRedisplay();
}

void keypressed(unsigned char key, int x, int y)
{
    if(key == ESC)
    {
        exit(0);
    }
}

void initGL(int width, int height)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, width, height);
    gluPerspective(45, 1.0 * width / height, 1, 1000);
    glMatrixMode(GL_MODELVIEW);
}

void resize(int width, int height)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, width, height);
    gluPerspective(45, 1.0* width / height, 1, 1000);
    glMatrixMode(GL_MODELVIEW);
}

void initGlut(int argc, char ** argv, int width, int height)
{
    glutInit(&argc, argv);
    
    glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(width, height);
    glutCreateWindow (argv[0]);
    
    glutDisplayFunc(paint);
    glutKeyboardFunc(keypressed);
    glutReshapeFunc(resize);
}

