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
    #include <GL/freeglut.h>
#endif

#include "types.h"
#include "math.h"

#define ESC 27
#define KEY_A 97
#define KEY_D 100
#define KEY_W 119
#define KEY_S 115
#define KEY_R 114
#define KEY_F 102


int vertcount;
vertex *vertices;
GLuint underground;
GLuint indexbufferID;

void (*callback)() = NULL;

int width, height;

int window_width;
int window_height;


void initScene(int width, int height, vertex *vertices);
void initGlut(int argc, char ** argv);
void initGL();

void resize();
void animate(int v);
void keypressed(unsigned char key, int x, int y);
void drawString(int x, int y, const std::string &text, void *font = GLUT_BITMAP_HELVETICA_12);

int frame = 0;
float rotationY = 0;
float rotationX = 10;
float zoom = 10;

timeval l_time, c_time;
long current_time, last_time, fpsupdate_time;
float fps, timediff;

void paint()
{
    gettimeofday(&c_time, 0);
    long current_time = c_time.tv_sec * 1000000 + c_time.tv_usec;
    long last_time = l_time.tv_sec * 1000000 + l_time.tv_usec;
    
    timediff = (current_time - last_time ) / 1000.0f;
    
    l_time = c_time;
    
    if(current_time - fpsupdate_time > 1000000)
    {
        fps = 1000.0f/timediff;
        fpsupdate_time = current_time;
    }
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    std::stringstream text;
    
    text << "FPS: " << fps;
    
    
    
    glMatrixMode(GL_MODELVIEW);
    
    glPushMatrix();
    glLoadIdentity();
    
    glTranslatef(0, -rotationX/4, -zoom);
    glRotatef(rotationX, 1, 0, 0);
    
    glTranslatef(0, 0, -20);
    glRotatef(rotationY, 0, 1, 0);
    
    GLfloat light_position[] = { 0.0, 5, 5, 0.5 };
    GLfloat light_direction[] = { rotationX/4, -rotationY, 0.0 };
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, light_direction);
    
    glColor3f(1,0,0);
    glBegin(GL_POINTS);
        glVertex3f(0.0, -5, 5);
    glEnd();
    
    frame++;
    
    
    
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_INDEX_ARRAY);
    
        glBindBuffer(GL_ARRAY_BUFFER_ARB, underground);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,indexbufferID);
        
        glVertexPointer(3, GL_FLOAT, 0, 0);
        glIndexPointer(GL_INT, 0, 0);
        
        glDrawElements( GL_QUADS, 4*(width-1)*(height-1), GL_UNSIGNED_INT, 0 );
    
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_INDEX_ARRAY);
    
    /*
    for(int y = 0;y < height;y++)
    {
        for(int x = 0; x < width;x++)
        {
            vertices[y*width+x].y += 0.1*sin(frame+x/10.0f)+0.1*cos(frame+y/10.0f);
        }
    }*/
            
    glPopMatrix();
    drawString(5, 25, text.str());
    glutSwapBuffers();
}

void createWindow(int argc, char **argv, int width, int height, void (*cb)(), int vertex_width, int vertex_height, vertex *vertices)
{
    callback = cb;
    window_width = width;
    window_height = height;
    
    initGlut(argc, argv);
    initGL();
    
    initScene(vertex_width, vertex_height, vertices);
    
    glutMainLoop();
}

void initScene(int w, int h, vertex *v)
{
    width = w;
    height = h;
    if(v == NULL)
    {
        vertices = (vertex *) malloc(width*height*sizeof(vertex));
        CHECK_NOTNULL(vertices);
        for(int y = 0;y < height;y++)
        {
            for(int x = 0; x < width;x++)
            {
                vertex v;
                v.x = x*16.0f/width-8;
                v.z = y*16.0f/height-8;
                v.y = 0.3*sin(v.x/20.0f)+0.3*cos(v.y/20.0f);

                vertices[y*width+x] = v;
            }
        }
    }
    else
    {
        vertices = v;
    }
    
    int *indices = (int *) malloc(4*(width-1)*(height-1)*sizeof(int));
    CHECK_NOTNULL(indices);
    
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
    
    glGenBuffers(1, &underground);
    glBindBuffer(GL_ARRAY_BUFFER, underground);
    glBufferData(GL_ARRAY_BUFFER, width*height*sizeof(vertex), vertices, GL_STREAM_COPY_ARB);
    
    //Map the buffer to host mem
    vertices = (vertex *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    
    glGenBuffers(1, &indexbufferID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexbufferID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*(width-1)*(height-1)*sizeof(int), indices, GL_STATIC_READ_ARB);
    
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
    glLoadIdentity();
    glViewport(0, 0, window_width, window_height);
    gluPerspective(45, 1.0 * window_width / window_height, 1, 1000);
    glMatrixMode(GL_MODELVIEW);
    
    GLfloat light_ambient[] = {0.8, 0.8, 0.8, 1.0};
    GLfloat light_diffuse[] = {0.8, 0.8, 0.8, 1.0};
    
    glClearColor (0.0, 0.0, 0.0, 0.0);
    glShadeModel (GL_FLAT);
    
    glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
    
    
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
}

void resize(int width, int height)
{
    window_width = width;
    window_height = height;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, window_width, window_height);
    gluPerspective(45, 1.0* window_width / window_height, 1, 1000);
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
    glutTimerFunc(50, animate, 0);
}

void drawString(int x, int y, const std::string &text, void *font)
{
    
    glMatrixMode(GL_PROJECTION);
    glDisable(GL_LIGHTING);
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
    glEnable(GL_LIGHTING);
    glMatrixMode(GL_MODELVIEW);
}
