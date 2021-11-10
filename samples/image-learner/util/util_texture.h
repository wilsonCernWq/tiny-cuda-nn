//===========================================================================//
//                                                                           //
// Copyright(c) 2018 Qi Wu (Wilson)                                          //
// University of California, Davis                                           //
// MIT Licensed                                                              //
//                                                                           //
//===========================================================================//

//===========================================================================//
// Header
//===========================================================================//

#ifndef GFX_UTIL_TEXTURE_H
#define GFX_UTIL_TEXTURE_H

#include "util.h"

GLuint
load_texture_from_file(const char* imagepath); // Load a texture from a file

#endif // GFX_UTIL_TEXTURE_H

//===========================================================================//
// Implementations
//===========================================================================//

#ifdef GFX_UTIL_TEXTURE_IMPLEMENTATION

#include "stb_image.h"

GLuint
load_texture_from_file(const char* imagepath)
{
  printf("Reading image %s\n", imagepath);

  int width, height, channels;

  // Actual RGB data
  stbi_set_flip_vertically_on_load(true);
  unsigned char* data = stbi_load(imagepath, &width, &height, &channels, STBI_rgb);

  // Create one OpenGL texture
  GLuint textureID;
  glGenTextures(1, &textureID);

  // "Bind" the newly created texture : all future texture functions will modify
  // this texture
  glBindTexture(GL_TEXTURE_2D, textureID);

  // Give the image to OpenGL
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

  // OpenGL has now copied the data. Free our own version
  delete[] data;

  // Poor filtering, or ...
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  // ... nice trilinear filtering ...
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  // ... which requires mipmaps. Generate them automatically.
  glGenerateMipmap(GL_TEXTURE_2D);

  // Return the ID of the texture we just created
  return textureID;
}

#endif // GFX_UTIL_TEXTURE_IMPLEMENTATION
