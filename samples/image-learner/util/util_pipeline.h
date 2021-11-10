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

#ifndef GFX_UTIL_PIPELINE_H
#define GFX_UTIL_PIPELINE_H

#include "util.h"
#include "util_shader.h"

#include <glm/glm.hpp>

#include <map>
#include <vector>

struct Pipeline {
private:
  GLuint program_id;
  std::map<std::string, GLuint> uniforms; /* name -> ID */
  std::map<GLuint, GLuint> textures; /* ID -> order */

public:
  Pipeline(const void* vs_data, size_t vs_size, const void* fs_data, size_t fs_size);

  void
  Enable()
  {
    glUseProgram(program_id);
  }

  void
  Clear()
  {
    glDeleteProgram(program_id);
  }

  GLuint
  GetUniformId(const std::string& name);

  void
  BindTexture(const std::string& name, GLuint tex);

  template<typename T>
  void
  SetUniform(const std::string& name, const T& value);
};

template<>
inline void
Pipeline::SetUniform<int>(const std::string& name, const int& value)
{
  glUniform1i(GetUniformId(name), value);
}

template<>
inline void
Pipeline::SetUniform<float>(const std::string& name, const float& value)
{
  glUniform1f(GetUniformId(name), value);
}

template<>
inline void
Pipeline::SetUniform<glm::mat4>(const std::string& name, const glm::mat4& value)
{
  glUniformMatrix4fv(GetUniformId(name), 1, GL_FALSE, &value[0][0]);
}

template<>
inline void
Pipeline::SetUniform<glm::vec3>(const std::string& name, const glm::vec3& value)
{
  glUniform3fv(GetUniformId(name), 1, &value.x);
}

#endif // GFX_UTIL_PIPELINE_H

//===========================================================================//
// Implementations
//===========================================================================//

#ifdef GFX_UTIL_PIPELINE_IMPLEMENTATION

Pipeline::Pipeline(const void* vs_data, size_t vs_size, const void* fs_data, size_t fs_size)
{
  program_id = load_program_from_embedding(vs_data, vs_size, fs_data, fs_size);
  Enable();
}

GLuint
Pipeline::GetUniformId(const std::string& name)
{
  if (uniforms.count(name) == 0) {
    uniforms[name] = glGetUniformLocation(program_id, name.c_str());
  }
  return uniforms[name];
}

void
Pipeline::BindTexture(const std::string& name, GLuint tex)
{
  const auto tex_id = GetUniformId(name);
  if (textures.count(tex_id) == 0) {
    const auto n = textures.size();
    textures[tex_id] = n;
  }
  const auto channel = textures[tex_id];
  glActiveTexture(GL_TEXTURE0 + channel);
  glBindTexture(GL_TEXTURE_2D, tex);
  glUniform1i(tex_id, channel);
}

#endif // GFX_UTIL_PIPELINE_IMPLEMENTATION
