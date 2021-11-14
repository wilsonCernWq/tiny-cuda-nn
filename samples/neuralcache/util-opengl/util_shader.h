//===========================================================================//
//                                                                           //
// Copyright(c) ECS 175 (2020)                                               //
// University of California, Davis                                           //
// MIT Licensed                                                              //
//                                                                           //
//===========================================================================//

//===========================================================================//
// Header
//===========================================================================//

#ifndef GFX_UTIL_SHADER_H
#define GFX_UTIL_SHADER_H

#if defined(ENABLE_OPENGL_COMPATIBLE)
#error "No shader support for legacy OpenGL API"
#endif // defined(ENABLE_OPENGL_COMPATIBLE)

#include "util.h"

GLuint
load_program_from_files(const char* vshader_fname, const char* fshader_fname);

#if defined(ENABLE_OPENGL_CORE_3_3)
GLuint
load_program_from_embedding(const void* _vshader_text,
                            long int vshader_size,
                            const void* _fshader_text,
                            long int fshader_size,
                            const void* _gshader_text = NULL,
                            long int gshader_size = 0);
#endif

#if defined(ENABLE_OPENGL_CORE_4_1)
GLuint
load_program_from_embedding(const void* _vshader_text,
                            long int vshader_size,
                            const void* _fshader_text,
                            long int fshader_size,
                            const void* _gshader_text = NULL,
                            long int gshader_size = 0,
                            const void* _tcshader_text = NULL,
                            long int tcshader_size = 0,
                            const void* _teshader_text = NULL,
                            long int teshader_size = 0);
#endif

#endif // GFX_UTIL_SHADER_H

//===========================================================================//
// Implementations
//===========================================================================//

#ifdef GFX_UTIL_SHADER_IMPLEMENTATION

#include <fstream>
#include <string>

void
check_shader_compilation_log(GLuint shader, const std::string& fname)
{
  GLint isCompiled = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
  if (isCompiled == GL_FALSE) {
    GLint maxLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);
    // The maxLength includes the NULL character
    std::vector<GLchar> errorLog(maxLength);
    glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]);
    // Provide the infolog in whatever manor you deem best.
    // Exit with failure.
    glDeleteShader(shader); // Don't leak the shader.
    // show the message
    std::cerr << "compilation error for shader: " << fname << std::endl << errorLog.data() << std::endl;
  }
}

void
_attach_shader(GLuint program, GLuint type, const char* ptr, const std::string& msg)
{
  GLuint shader = glCreateShader(type);
  {
    glShaderSource(shader, 1, &ptr, NULL);
    glCompileShader(shader);
    check_shader_compilation_log(shader, msg.c_str());
    check_error_gl(msg.c_str());
  }
  glAttachShader(program, shader);
  check_error_gl("Compile Shaders: Attach");
}

void
_attach_shader(GLuint program, GLuint type, const void* _data, size_t size, const std::string& msg)
{
  const char* data = (const char*)_data;
  std::string str = "";
  for (int i = 0; i < size; ++i) {
    str += data[i];
  }
  const char* ptr = str.c_str();
  // const char* ptr = str.c_str();
  _attach_shader(program, type, ptr, msg);
}

static const char*
_read_shader_file(const char* fname)
{
  std::ifstream file(fname, std::ios::binary | std::ios::ate | std::ios::in);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  char* buffer = new char[size + 1];
  buffer[size] = '\0';
  if (!file.read(const_cast<char*>(buffer), size)) {
    fprintf(stderr, "Error: Cannot read file %s\n", fname);
    exit(-1);
  }
  return buffer;
}

GLuint
load_program_from_files(const char* vshader_fname, const char* fshader_fname)
{
  fprintf(stdout, "[shader] reading vertex shader file %s\n", vshader_fname);
  fprintf(stdout, "[shader] reading fragment shader file %s\n", fshader_fname);
  // Create program
  GLuint program = glCreateProgram();
  if (glCreateProgram == 0)
    throw std::runtime_error("wrong program");
  // Vertex shader
  const char* vshader_text = _read_shader_file(vshader_fname);
  _attach_shader(program, GL_VERTEX_SHADER, vshader_text, "Compile Vertex Shader");
  // Fragment shader
  const char* fshader_text = _read_shader_file(fshader_fname);
  _attach_shader(program, GL_FRAGMENT_SHADER, fshader_text, "Compile Fragment Shader");
  // Link shaders
  glLinkProgram(program);
  check_error_gl("Compile Shaders: Link");
  glUseProgram(program);
  check_error_gl("Compile Shaders: Final");
  return program;
}

#if defined(ENABLE_OPENGL_CORE_3_3)
GLuint
load_program_from_embedding(const void* _vshader_text,
                            long int vshader_size,
                            const void* _fshader_text,
                            long int fshader_size,
                            const void* _gshader_text,
                            long int gshader_size)
{
  // Create program
  GLuint program = glCreateProgram();
  if (glCreateProgram == 0)
    throw std::runtime_error("wrong program");
  // Vertex shader
  _attach_shader(program, GL_VERTEX_SHADER, _vshader_text, vshader_size, "Compile Vertex Shader");
  // Fragment shader
  _attach_shader(program, GL_FRAGMENT_SHADER, _fshader_text, fshader_size, "Compile Fragment Shader");
  // Geometry shader
  if (_gshader_text)
    _attach_shader(program, GL_GEOMETRY_SHADER, _gshader_text, gshader_size, "Compile Geometry Shader");
  // Link shaders
  glLinkProgram(program);
  check_error_gl("Compile Shaders: Link");
  glUseProgram(program);
  check_error_gl("Compile Shaders: Final");
  return program;
}
#endif

#if defined(ENABLE_OPENGL_CORE_4_1)
GLuint
load_program_from_embedding(const void* _vshader_text,
                            long int vshader_size,
                            const void* _fshader_text,
                            long int fshader_size,
                            const void* _gshader_text,
                            long int gshader_size,
                            const void* _tcshader_text,
                            long int tcshader_size,
                            const void* _teshader_text,
                            long int teshader_size)
{
  // Create program
  GLuint program = glCreateProgram();
  if (glCreateProgram == 0)
    throw std::runtime_error("wrong program");
  // Vertex shader
  _attach_shader(program, GL_VERTEX_SHADER, _vshader_text, vshader_size, "Compile Vertex Shader");
  // Fragment shader
  _attach_shader(program, GL_FRAGMENT_SHADER, _fshader_text, fshader_size, "Compile Fragment Shader");
  // Geometry shader
  if (_gshader_text)
    _attach_shader(program, GL_GEOMETRY_SHADER, _gshader_text, gshader_size, "Compile Geometry Shader");
  // Tessellation shaders
  if (_tcshader_text)
    _attach_shader(
      program, GL_TESS_CONTROL_SHADER, _tcshader_text, tcshader_size, "Compile Tessellation Control Shader");
  if (_teshader_text)
    _attach_shader(
      program, GL_TESS_EVALUATION_SHADER, _teshader_text, teshader_size, "Compile Tessellation Evaluation Shader");
  // Link shaders
  glLinkProgram(program);
  check_error_gl("Compile Shaders: Link");
  glUseProgram(program);
  check_error_gl("Compile Shaders: Final");
  return program;
}
#endif

#endif // GFX_UTIL_SHADER_IMPLEMENTATION
