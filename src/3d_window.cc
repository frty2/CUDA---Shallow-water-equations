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
    #include <GL/gl.h>
    #include <GL/glext.h>
    #include <GL/glut.h>
#endif

#include "types.h"
#include "math.h"

#define ESC 27
#define KEY_A 97
#define KEY_D 100
#define KEY_W 119
#define KEY_S 115


int vertcount;
vertex *vertices;
GLuint watersurface;
GLuint indexbufferID;

void (*callback)() = NULL;

int width, height;


void initScene(int width, int height);
void initGlut(int argc, char ** argv, int width, int height);
void initGL(int width, int height);

void resize(int width, int height);
void animate(int v);
void keypressed(unsigned char key, int x, int y);
void drawString(int x, int y, const std::string &text, void *font = GLUT_BITMAP_HELVETICA_12);

int frame = 0;
float rotationY = 0;
float zoom = 10;

timeval l_time, c_time;

long current_time, last_time, fpsupdate_time;
float fps;
void paint()
{
    gettimeofday(&c_time, 0);
    long current_time = c_time.tv_sec * 1000000 + c_time.tv_usec;
    long last_time = l_time.tv_sec * 1000000 + l_time.tv_usec;
    
    float timediff = (current_time - last_time ) / 1000.0f;
    
    l_time = c_time;
    
    if(current_time - fpsupdate_time > 1000000)
    {
        fps = 1000.0f/timediff;
        fpsupdate_time = current_time;
    }
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    std::stringstream text;
    
    text << "FPS: " << fps;
    
    drawString(0,25, text.str());
    
    
    glMatrixMode(GL_MODELVIEW);
    
    glPushMatrix();
    glLoadIdentity();
    
    glTranslatef(0, -10, -zoom);
    glRotatef(10, 1, 0, 0);
    
    glTranslatef(0, 0, -20);
    glRotatef(rotationY, 0, 1, 0);
    
    frame++;
    
    int w = width;
    
    
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_INDEX_ARRAY);
    
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, watersurface);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,indexbufferID);
        
        glVertexPointer(3, GL_FLOAT, 0, 0);
        glIndexPointer(GL_INT, 0, 0);
        
        glDrawElements( GL_QUADS, 4*(width-1)*(height-1), GL_UNSIGNED_INT, 0 );
    
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_INDEX_ARRAY);
    
    for(int y = 0;y < height;y++)
    {
        for(int x = 0; x < width;x++)
        {
            vertices[y*w+x].y = 0.3*sin((x+frame)/20.0f)+0.3*cos((y)/20.0f);
        }
    }
            
    glPopMatrix();
    glutSwapBuffers();
}

void createWindow(int argc, char **argv, int width, int height, void (*cb)())
{
    callback = cb;
    
    initGlut(argc, argv, width, height);
    initGL(width, height);
    
    initScene(width, height);
    
    glutMainLoop();
}

void initScene(int w, int h)
{
    width = w;
    height = h;
    vertices = (vertex *) malloc(width*height*sizeof(vertex));
    CHECK_NOTNULL(vertices);
    
    int *indices = (int *) malloc(4*(width-1)*(height-1)*sizeof(int));
    CHECK_NOTNULL(indices);
    
    for(int y = 0;y < height;y++)
    {
        for(int x = 0; x < width;x++)
        {
            vertex v;
            v.x = x*16.0f/width-8;
            v.z = y*16.0f/height-8;
            v.y = 0.3*sin(x/20.0f)+0.3*cos(y/20.0f);
            
            vertices[y*width+x] = v;
        }
    }
    
    for(int y = 0;y < height-1;y++)
    {
        for(int x = 0; x < width-1;x++)
        {      
            indices[4*( y*(width-1)+x )] = y*width+x;
            indices[4*( y*(width-1)+x )+1] = y*width+x+1;
            indices[4*( y*(width-1)+x )+2] = (y+1)*width+x+1;
            indices[4*( y*(width-1)+x )+3] = (y+1)*width+x;
        }
    }
    
    glGenBuffersARB(1, &watersurface);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, watersurface);
    glBufferDataARB(GL_ARRAY_BUFFER_ARB, width*height*sizeof(vertex), vertices, GL_STREAM_COPY_ARB);
    
    //Map the buffer to host mem
    vertices = (vertex *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    
    glGenBuffersARB(1, &indexbufferID);
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, indexbufferID);
    glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 4*(width-1)*(height-1)*sizeof(int), indices, GL_STATIC_READ_ARB);
    
    free(indices);
}

void animate(int v)
{
    if(callback != NULL)
    {
        callback();
    }
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
        zoom = --zoom < 0 ? 0 : zoom;
    }
    if(key == KEY_S)
    {
        zoom = ++zoom > 15 ? 15 : zoom;
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
    glutTimerFunc(50, animate, 0);
}

void drawString(int x, int y, const std::string &text, void *font)
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
        glLoadIdentity();
        gluOrtho2D(0, width, height, 0);
        
        glRasterPos2f(x, y);
        
        glColor3f(1.0f, 1.0f, 1.0f);
        int i = 0;
        while (text[i] != '\0')
        {
            glutBitmapCharacter(font, text[i++]);
        }
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}
