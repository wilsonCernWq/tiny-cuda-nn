//===========================================================================//
//                                                                           //
// Copyright(c) 2018 Qi Wu (Wilson)                                          //
// University of California, Davis                                           //
// MIT Licensed                                                              //
//                                                                           //
//===========================================================================//

#include "util.h"

// #define GFX_UTIL_MESH_IMPLEMENTATION
// #include "util_mesh.h"
// #undef GFX_UTIL_MESH_IMPLEMENTATION

#define GFX_UTIL_CAMERA_IMPLEMENTATION
#include "util_camera.h"
#undef GFX_UTIL_CAMERA_IMPLEMENTATION

#define GFX_UTIL_TEXTURE_IMPLEMENTATION
#include "util_texture.h"
#undef GFX_UTIL_TEXTURE_IMPLEMENTATION

#if !defined(ENABLE_OPENGL_COMPATIBLE) /* Modern OpenGL */
#define GFX_UTIL_SHADER_IMPLEMENTATION
#include "util_shader.h"
#undef GFX_UTIL_SHADER_IMPLEMENTATION
#endif // !defined(ENABLE_OPENGL_COMPATIBLE)

#if !defined(ENABLE_OPENGL_COMPATIBLE) /* Modern OpenGL */
#define GFX_UTIL_FRAMEBUFFER_IMPLEMENTATION
#include "util_framebuffer.h"
#undef GFX_UTIL_FRAMEBUFFER_IMPLEMENTATION
#endif // !defined(ENABLE_OPENGL_COMPATIBLE)

#if !defined(ENABLE_OPENGL_COMPATIBLE) /* Modern OpenGL */
#define GFX_UTIL_PIPELINE_IMPLEMENTATION
#include "util_pipeline.h"
#undef GFX_UTIL_PIPELINE_IMPLEMENTATION
#endif // !defined(ENABLE_OPENGL_COMPATIBLE)

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#undef STB_IMAGE_WRITE_IMPLEMENTATION

// trying this obj loader https://github.com/syoyo/tinyobjloader
#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"
#undef TINYOBJLOADER_IMPLEMENTATION

#include <chrono>
#include <fstream>
#include <string>

//---------------------------------------------------------------------------------------
// error check helper from EPFL ICG class
static inline const char*
ErrorString(GLenum error)
{
  const char* msg;
  switch (error) {
#define Case(Token) \
  case Token:       \
    msg = #Token;   \
    break;
    Case(GL_INVALID_ENUM);
    Case(GL_INVALID_VALUE);
    Case(GL_INVALID_OPERATION);
    Case(GL_INVALID_FRAMEBUFFER_OPERATION);
    Case(GL_NO_ERROR);
    Case(GL_OUT_OF_MEMORY);
#undef Case
  }
  return msg;
}

void
_glCheckError(const char* file, int line, const char* comment)
{
  GLenum error;
  while ((error = glGetError()) != GL_NO_ERROR) {
    fprintf(stderr, "ERROR: %s (file %s, line %i: %s).\n", comment, file, line, ErrorString(error));
  }
}

//---------------------------------------------------------------------------------------

void
read_frame(GLFWwindow* window, std::vector<uint8_t>& buffer, int w, int h)
{
  const size_t nchannel = buffer.size() / ((size_t)w * (size_t)h);
  assert(nchannel == 3);
  // reading from the default framebuffer
  glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, &(buffer[0]));
  check_error_gl("Save a frame");
}

void
screen_shot(GLFWwindow* window, const std::string& fname)
{
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  std::vector<uint8_t> fb(size_t(width) * size_t(height) * size_t(3));
  read_frame(window, fb, width, height);
  save_jpg(fname, fb, width, height);
}

//---------------------------------------------------------------------------------------

float
get_framerate()
{
  static float fps = 0.0f; // measure frame rate
  static size_t frames = 0;
  static auto start = std::chrono::system_clock::now();
  ++frames;
  if (frames % 10 == 0 || frames == 1) { // dont update this too frequently
    std::chrono::duration<double> es = std::chrono::system_clock::now() - start;
    fps = frames / es.count();
  }
  return fps;
}

//---------------------------------------------------------------------------------------

void
save_jpg(const std::string& fname, std::vector<uint8_t>& fb, int w, int h)
{
  const size_t nchannel = fb.size() / ((size_t)w * (size_t)h);
  if (nchannel == 3) {
    stbi_write_jpg(fname.c_str(), w, h, 3, fb.data(), 100);
  }
  else if (nchannel == 4) {
    const int& width = w;
    const int& height = h;
    uint8_t* pixels = new uint8_t[width * height * 3];
    int index = 0;
    for (int j = height - 1; j >= 0; --j) {
      for (int i = 0; i < width; ++i) {
        int ir = int(fb[4 * (i + j * width) + 0]);
        int ig = int(fb[4 * (i + j * width) + 1]);
        int ib = int(fb[4 * (i + j * width) + 2]);
        pixels[index++] = ir;
        pixels[index++] = ig;
        pixels[index++] = ib;
      }
    }
    stbi_write_jpg(fname.c_str(), width, height, 3, pixels, 100);
    delete[] pixels;
  }
  else {
    throw std::runtime_error("Unknown image type");
  }
}

//---------------------------------------------------------------------------------------
