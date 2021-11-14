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

#ifndef GFX_UTIL_FRAMEBUFFER_H
#define GFX_UTIL_FRAMEBUFFER_H

#if defined(ENABLE_OPENGL_COMPATIBLE)
#error "No framebuffer object support for legacy OpenGL API"
#endif // defined(ENABLE_OPENGL_COMPATIBLE)

#include "util.h"

// Reference: https://www.khronos.org/opengl/wiki/Framebuffer_Object
struct FrameBufferObject {
  enum Flags_ {
    NO_DEPTH = 1, // disable depth testing
    NO_COLOR = 1 << 1, // disable color buffer
    DEPTH_TEXTURE = 1 << 2, // use depth texture instead of depth render buffer
    LINEAR = 1 << 3, // enable linear filtering
    COMPARE = 1 << 4 // enable texture compare mode (for shadow map)
  };

  GLuint framebuffer_id = 0;
  // Texture which will contain the RGB output of our shader.
  GLuint color_tex;
  // An optional depth buffer. This enables depth-testing.
  GLuint depth_tex; // depth texture. slower, but you can sample it later in your shader
  GLuint depth_rb; // depth render buffer: faster

  void
  Generate();

  void
  SetupTexture(uint8_t flag = 0);

  void
  Resize(int width, int height, uint8_t flag = 0);

  void
  Clear();

  void
  BindFramebuffer()
  {
    // Switch to the framebuffer object
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_id);
  }

  void
  UnbindFramebuffer()
  {
    // Switch back to the default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  void
  BindColorTexture()
  {
    glBindTexture(GL_TEXTURE_2D, color_tex);
  }

  GLuint
  GetColorTexture()
  {
    return color_tex;
  }

  void
  BindDepthTexture()
  {
    glBindTexture(GL_TEXTURE_2D, depth_tex);
  }

  GLuint
  GetDepthTexture()
  {
    return depth_tex;
  }
};

using FBO = FrameBufferObject;

#endif // GFX_UTIL_FRAMEBUFFER_H

//===========================================================================//
// Implementations
//===========================================================================//

#ifdef GFX_UTIL_FRAMEBUFFER_IMPLEMENTATION

void
FrameBufferObject::Generate()
{
  glGenFramebuffers(1, &framebuffer_id);
  glGenTextures(1, &color_tex);
  glGenTextures(1, &depth_tex);
  glGenRenderbuffers(1, &depth_rb);
}

void
FrameBufferObject::SetupTexture(uint8_t flag)
{
  if (flag & LINEAR) {
    // Poor filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  }
  else {
    // Poor filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  }
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  if (flag & COMPARE) {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
  }
}

void
FrameBufferObject::Resize(int width, int height, uint8_t flag)
{
  BindFramebuffer();

  // color buffer
  if (!(flag & NO_COLOR)) {
    // "Bind" the newly created texture : all future texture functions will modify this texture
    glBindTexture(GL_TEXTURE_2D, color_tex);

    // Give an empty image to OpenGL ( the last "0" means "empty" )
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    SetupTexture(flag);

    // Set "renderedTexture" as our colour attachement #0
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color_tex, 0);

    // Set the list of draw buffers.
    GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers
  }
  else {
    // No color output in the bound framebuffer, only depth.
    glDrawBuffer(GL_NONE);
  }

  // The depth buffer
  if (!(flag & NO_DEPTH)) {
    if (flag & DEPTH_TEXTURE) {
      glBindTexture(GL_TEXTURE_2D, depth_tex);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
      SetupTexture(flag);
      glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth_tex, 0);
    }
    else {
      glBindRenderbuffer(GL_RENDERBUFFER, depth_rb);
      glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb);
    }
  }

  // Always check that our framebuffer is ok
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    throw std::runtime_error("framebuffer object incomplete");

  // Switch back to the default framebuffer
  UnbindFramebuffer();
}

void
FrameBufferObject::Clear()
{
  glDeleteFramebuffers(1, &framebuffer_id);
  glDeleteTextures(1, &color_tex);
  glDeleteTextures(1, &depth_tex);
  glDeleteRenderbuffers(1, &depth_rb);
}

#endif // GFX_UTIL_FRAMEBUFFER_IMPLEMENTATION
