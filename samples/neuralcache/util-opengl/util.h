//===========================================================================//
//                                                                           //
// Copyright(c) 2018 Qi Wu (Wilson)                                          //
// University of California, Davis                                           //
// MIT Licensed                                                              //
//                                                                           //
//===========================================================================//

#pragma once

#ifdef ENABLE_OPENGL_COMPATIBLE
#ifdef ENABLE_OPENGL_CORE_3_3
#error "Legacy OpenGL is selected, but libraries compiled in OpenGL 3.3 are linked"
#endif
#ifdef ENABLE_OPENGL_CORE_4_1
#error "Legacy OpenGL is selected, but libraries compiled in OpenGL 4.1 are linked"
#endif
#endif

#ifdef ENABLE_OPENGL_CORE_3_3
#ifdef ENABLE_OPENGL_COMPATIBLE
#error "OpenGL 3.3 is selected, but libraries compiled in Legacy OpenGL are linked"
#endif
#ifdef ENABLE_OPENGL_CORE_4_1
#error "OpenGL 3.3 is selected, but libraries compiled in OpenGL 4.1 are linked"
#endif
#endif

#ifdef ENABLE_OPENGL_CORE_4_1
#ifdef ENABLE_OPENGL_COMPATIBLE
#error "OpenGL 4.1 is selected, but libraries compiled in Legacy OpenGL are linked"
#endif
#ifdef ENABLE_OPENGL_CORE_3_3
#error "OpenGL 4.1 is selected, but libraries compiled in OpenGL 3.3 are linked"
#endif
#endif

#define _USE_MATH_DEFINES
#include <cmath>

#include <glad/glad.h>
// it is necessary to include glad before glfw
#include <GLFW/glfw3.h>

// #include <algorithm>
#include <iostream>
#include <vector>

void
_glCheckError(const char* file, int line, const char* comment);

#ifndef NDEBUG
#define check_error_gl(x) _glCheckError(__FILE__, __LINE__, x)
#else
#define check_error_gl(x) ((void)0)
#endif

// void
// read_frame(GLFWwindow* window, std::vector<uint8_t>& buffer /* TODO use raw array*/, int w, int h);

// void
// screen_shot(GLFWwindow* window, const std::string& fname);

// float
// get_framerate();

// void
// save_jpg(const std::string& fname, std::vector<uint8_t>& fb, int w, int h);

/* TODO support other formats */
// void
// save_png(const std::string& fname, std::vector<uint8_t>& fb, int w, int h);

/* TODO support other formats */
// void
// save_bmp(const std::string& fname, std::vector<uint8_t>& fb, int w, int h);
