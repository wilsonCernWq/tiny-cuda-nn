//===========================================================================//
//                                                                           //
// Copyright(c) ECS 175 (2020)                                               //
// University of California, Davis                                           //
// MIT Licensed                                                              //
//                                                                           //
//===========================================================================//

#ifndef GFX_UTIL_MESH_H
#define GFX_UTIL_MESH_H

//////////////////////////////////////////////////////////////////////////////
// toggles controlled by users
//////////////////////////////////////////////////////////////////////////////
// #define GFX_UTIL_MESH_DISABLE_VERTEX
// #define GFX_UTIL_MESH_DISABLE_NORMAL
// #define GFX_UTIL_MESH_DISABLE_TEXCOORD

#include "util.h"

#include "tiny_obj_loader.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <memory>
#include <string>
#include <vector>

struct Object {
  std::string name;
  glm::vec3 center;
  glm::vec3 lower, upper;
  float scale;

  Object();

  virtual ~Object() = default;

  virtual void
  clear() = 0;

  virtual void
  create() = 0;

  virtual void
  load(const std::string& filename) = 0;

  virtual void
  render(int layout_vertex, int layout_normal, int layout_texcoord) = 0;

  glm::mat4
  get_model_matrix();

protected:
  static void
  enable_vbo(GLuint vbo, int layout, int size);

  static GLuint
  create_vbo(size_t buffer_size, const float* buffer_data);

  static void
  delete_vbo(GLuint& vbo);
};

//////////////////////////////////////////////////////////////////////////////
// Triangle Array
//////////////////////////////////////////////////////////////////////////////

struct MeshFromFile : public Object {
public:
  void
  load(const std::string& filename) override;

private:
  virtual bool
  alloc(tinyobj::attrib_t& attrib) = 0;

  virtual void
  fill(size_t num_of_vertices,
       size_t num_of_faces,
       std::vector<tinyobj::index_t>& indices,
       glm::vec3* attr_vertex,
       glm::vec3* attr_normal,
       glm::vec2* attr_textcoord,
       bool compute_normal) = 0;
};

struct TriangleArray : public MeshFromFile {
  struct Mesh {
  public:
    size_t size_triangles;

#ifndef GFX_UTIL_MESH_DISABLE_VERTEX
    std::unique_ptr<float[]> vertices;
    GLuint vbo_vertex;

    void
    enable_vertex(int layout)
    {
      Object::enable_vbo(this->vbo_vertex, layout, 3);
    }
#endif

#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
    std::unique_ptr<float[]> normals;
    GLuint vbo_normal;

    void
    enable_normal(int layout)
    {
      Object::enable_vbo(this->vbo_normal, layout, 3);
    }
#endif

#ifndef GFX_UTIL_MESH_DISABLE_TEXCOORD
    std::unique_ptr<float[]> texcoords;
    GLuint vbo_texcoord;

    void
    enable_texcoord(int layout)
    {
      Object::enable_vbo(this->vbo_texcoord, layout, 2);
    }
#endif

    size_t
    num_vertices() const
    {
      return this->size_triangles * 3;
    }
  };

#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
  TriangleArray(bool flat_normal = false) : flat_normal(flat_normal) {}
#endif

  void
  clear() override;

  void
  create() override;

  void
  render(int layout_vertex, int layout_normal, int layout_texcoord) override;

public:
  std::vector<Mesh> meshes;

private:
  bool
  alloc(tinyobj::attrib_t& attrib) override;

  void
  fill(size_t num_of_vertices,
       size_t num_of_faces,
       std::vector<tinyobj::index_t>& indices,
       glm::vec3* attr_vertex,
       glm::vec3* attr_normal,
       glm::vec2* attr_textcoord,
       bool compute_normal) override;

#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
  bool flat_normal;
#endif
};

struct TriangleIndex : public MeshFromFile {
  struct Mesh {
    std::unique_ptr<unsigned int> indices;
    size_t size_indices;
    GLuint vbo_index;
  };

  void
  clear() override;

  void
  create() override;

  void
  render(int layout_vertex, int layout_normal, int layout_texcoord) override;

public:
  std::vector<Mesh> meshes;

private:
  bool
  alloc(tinyobj::attrib_t& attrib) override;

  void
  fill(size_t num_of_vertices,
       size_t num_of_faces,
       std::vector<tinyobj::index_t>& indices,
       glm::vec3* attr_vertex,
       glm::vec3* attr_normal,
       glm::vec2* attr_textcoord,
       bool compute_normal) override;

#ifndef GFX_UTIL_MESH_DISABLE_VERTEX
  std::unique_ptr<float> vertices;
  GLuint vbo_vertex;
#endif
#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
  std::unique_ptr<float> normals;
  GLuint vbo_normal;
#endif
#ifndef GFX_UTIL_MESH_DISABLE_TEXCOORD
  std::unique_ptr<float> texcoords;
  GLuint vbo_texcoord;
#endif
  size_t num_of_vertices;
};

//////////////////////////////////////////////////////////////////////////////
// Cube
//////////////////////////////////////////////////////////////////////////////

struct CubeObject : public Object {

  // Our vertices. Tree consecutive floats give a 3D vertex; Three consecutive
  // vertices give a triangle. A cube has 6 faces with 2 triangles each, so this
  // makes 6*2=12 triangles, and 12*3 vertices
  static const GLfloat g_vertex_buffer_data[];

  // One color for each vertex. They were generated randomly.
  static const GLfloat g_color_buffer_data[];

  GLuint vertex_buffer_id;
  GLuint color_buffer_id;

  void
  clear() override;

  void
  create() override;

  void
  render(int layout_vertex, int layout_normal, int layout_texcoord) override;

  void
  load(const std::string& filename) override
  {
  }
};

//////////////////////////////////////////////////////////////////////////////
// Cone
//////////////////////////////////////////////////////////////////////////////

struct ConeObject : public Object {
  GLuint vertex_buffer_id;
  GLuint normal_buffer_id;
  std::vector<float> vertices;
  std::vector<float> normals;

  int resolution;

  ConeObject(int resolution, float radius, float height);

  void
  clear() override;

  void
  create() override;

  void
  load(const std::string& filename) override
  {
  }

  void
  render(int layout_vertex, int layout_normal, int layout_texcoord) override;
};

//////////////////////////////////////////////////////////////////////////////
// Cylinder
//////////////////////////////////////////////////////////////////////////////

struct CylinderObject : public Object {
  GLuint vertex_buffer_id;
  GLuint normal_buffer_id;
  std::vector<float> vertices;
  std::vector<float> normals;

  int resolution;

  CylinderObject(int resolution, float radius, float height);

  void
  clear() override;

  void
  create() override;

  void
  load(const std::string& filename) override
  {
  }

  void
  render(int layout_vertex, int layout_normal, int layout_texcoord) override;
};

#endif // GFX_UTIL_MESH_H

//===========================================================================//
// Implementations
//===========================================================================//

#ifdef GFX_UTIL_MESH_IMPLEMENTATION

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>

#include <cassert>
#include <limits>
#include <numeric>

static_assert(sizeof(GLuint) == sizeof(GLuint), "'GLuint' does not equal to 'GLuint'");

Object::Object()
  : center(0.f), lower(+std::numeric_limits<float>::max()), upper(-std::numeric_limits<float>::max()), scale(1.f)
{
}

glm::mat4
Object::get_model_matrix()
{
  return glm::scale(glm::vec3(scale)) * glm::translate(-center);
}

void
Object::enable_vbo(GLuint vbo, int layout, int size)
{
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexAttribPointer(layout, // The attribute we want to configure
                        size, // size
                        GL_FLOAT, // type
                        GL_FALSE, // not normalized
                        0, // stride
                        (void*)0 // array buffer offset
  );
}

GLuint
Object::create_vbo(size_t buffer_size, const float* buffer_data)
{
  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * buffer_size, buffer_data, GL_STATIC_DRAW);
  return vbo;
}

void
Object::delete_vbo(GLuint& vbo)
{
  glDeleteBuffers(1, &vbo);
}

///////////////////////////////////////////////////////////////////////////////
// TriangleArray
///////////////////////////////////////////////////////////////////////////////

void
MeshFromFile::load(const std::string& inputfile)
{
  // Parse input argument
  //---------------------------------------------------------------------------
  const auto _i = inputfile.find_last_of('/');
  const auto path_name = inputfile.substr(0, _i);
  const auto file_name = inputfile.substr(_i + 1, inputfile.size() - _i - 1);

  // Tiny OBJ Loader
  //---------------------------------------------------------------------------
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn;
  std::string err;
  const bool ret = tinyobj::LoadObj(
    &attrib, &shapes, &materials, &warn, &err, inputfile.c_str(), path_name.c_str(), true /* force triangulate */);
  if (!warn.empty())
    std::cout << warn << std::endl;
  if (!err.empty())
    std::cerr << err << std::endl;
  if (!ret)
    throw std::runtime_error("failed to read file " + inputfile);

  // Setup
  //---------------------------------------------------------------------------
  size_t total_num_of_vertices = 0;
  glm::vec3 total_lower_bound(+std::numeric_limits<float>::max());
  glm::vec3 total_upper_bound(-std::numeric_limits<float>::max());

  bool compute_normal = alloc(attrib);
  glm::vec3* attr_vertex = (glm::vec3*)attrib.vertices.data();
  glm::vec3* attr_normal = (glm::vec3*)attrib.normals.data();
  glm::vec2* attr_textcoord = (glm::vec2*)attrib.texcoords.data();

  // Loop over shapes
  //---------------------------------------------------------------------------
  for (size_t s = 0; s < shapes.size(); s++) {

    // Loop over faces (polygon)
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {

      // Verify if all faces are triangles
      int fv = shapes[s].mesh.num_face_vertices[f];
      if (fv != 3)
        throw std::runtime_error("non-triangle faces are not supported");
      const size_t index_offset = 3 * f;

      // Access to vertex
      auto& idx0 = shapes[s].mesh.indices[index_offset + 0];
      auto& idx1 = shapes[s].mesh.indices[index_offset + 1];
      auto& idx2 = shapes[s].mesh.indices[index_offset + 2];
      const glm::vec3& v0 = attr_vertex[idx0.vertex_index];
      const glm::vec3& v1 = attr_vertex[idx1.vertex_index];
      const glm::vec3& v2 = attr_vertex[idx2.vertex_index];
      // Statistics
      this->center += v0 + v1 + v2;
      total_upper_bound = glm::max(total_upper_bound, v0);
      total_upper_bound = glm::max(total_upper_bound, v1);
      total_upper_bound = glm::max(total_upper_bound, v2);
      total_lower_bound = glm::min(total_lower_bound, v0);
      total_lower_bound = glm::min(total_lower_bound, v1);
      total_lower_bound = glm::min(total_lower_bound, v2);
    }

    // // Compute size
    // const size_t num_of_vertices = std::accumulate(shapes[s].mesh.num_face_vertices.begin(),
    //                                                shapes[s].mesh.num_face_vertices.end(),
    //                                                size_t(0),
    //                                                [](size_t init, uint8_t value) { return init + value; });
    // total_num_of_vertices += num_of_vertices;
    const size_t num_of_vertices = shapes[s].mesh.num_face_vertices.size() * 3;
    total_num_of_vertices += num_of_vertices;

    // Create a geometry (do this before fill!!!)
    fill(num_of_vertices,
         shapes[s].mesh.num_face_vertices.size(),
         shapes[s].mesh.indices,
         attr_vertex,
         attr_normal,
         attr_textcoord,
         compute_normal);
  }

  // Information
  //---------------------------------------------------------------------------
  this->name = file_name;
  this->center /= total_num_of_vertices; // Center of mass
  this->lower = total_lower_bound;
  this->upper = total_upper_bound;
  glm::vec3 s = total_upper_bound - total_lower_bound;
  this->scale = 4.f / glm::max(s.x, glm::max(s.y, s.z));
}

void
TriangleArray::clear()
{
  for (auto& m : meshes) {
#ifndef GFX_UTIL_MESH_DISABLE_VERTEX
    delete_vbo(m.vbo_vertex);
#endif
#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
    delete_vbo(m.vbo_normal);
#endif
#ifndef GFX_UTIL_MESH_DISABLE_TEXCOORD
    delete_vbo(m.vbo_texcoord);
#endif
  }
}

void
TriangleArray::create()
{
  // Generate a buffer for the indices as well
  for (auto& m : meshes) {
#ifndef GFX_UTIL_MESH_DISABLE_VERTEX
    m.vbo_vertex = Object::create_vbo(m.size_triangles * 9, m.vertices.get());
#endif
#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
    m.vbo_normal = Object::create_vbo(m.size_triangles * 9, m.normals.get());
#endif
#ifndef GFX_UTIL_MESH_DISABLE_TEXCOORD
    m.vbo_texcoord = Object::create_vbo(m.size_triangles * 6, m.texcoords.get());
#endif
  }
}

void
TriangleArray::render(int layout_vertex, int layout_normal, int layout_texcoord)
{
  for (auto& m : this->meshes) {
#ifndef GFX_UTIL_MESH_DISABLE_VERTEX
    m.enable_vertex(layout_vertex);
#else
    assert(layout_vertex == -1);
#endif
#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
    m.enable_normal(layout_normal);
#else
    assert(layout_normal == -1);
#endif
#ifndef GFX_UTIL_MESH_DISABLE_TEXCOORD
    m.enable_texcoord(layout_texcoord);
#else
    assert(layout_texcoord == -1);
#endif
    glDrawArrays(GL_TRIANGLES, 0, m.num_vertices());
  }
}

void
TriangleArray::fill(size_t num_of_vertices,
                    size_t num_of_faces,
                    std::vector<tinyobj::index_t>& indices,
                    glm::vec3* attr_vertex,
                    glm::vec3* attr_normal,
                    glm::vec2* attr_textcoord,
                    bool compute_normal)
{
  assert(num_of_vertices != 0);

  this->meshes.emplace_back();
  auto& self = this->meshes.back();

  self.size_triangles = num_of_vertices / 3;

#ifndef GFX_UTIL_MESH_DISABLE_VERTEX
  self.vertices.reset(new float[self.size_triangles * 9]);
  glm::vec3* data_vertex = (glm::vec3*)self.vertices.get();
#endif

#ifndef GFX_UTIL_MESH_DISABLE_TEXCOORD
  self.texcoords.reset(new float[self.size_triangles * 6]);
  glm::vec2* data_uv = (glm::vec2*)self.texcoords.get();
#endif

#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
  self.normals.reset(new float[self.size_triangles * 9]() /* zero initialize */);
  glm::vec3* data_normal = (glm::vec3*)self.normals.get();
#endif

  // Loop over faces (polygon)
  for (size_t f = 0; f < num_of_faces; f++) {
    const size_t index_offset = 3 * f;

    // Access to vertex
    auto& idx0 = indices[index_offset + 0];
    auto& idx1 = indices[index_offset + 1];
    auto& idx2 = indices[index_offset + 2];
    const glm::vec3& v0 = attr_vertex[idx0.vertex_index];
    const glm::vec3& v1 = attr_vertex[idx1.vertex_index];
    const glm::vec3& v2 = attr_vertex[idx2.vertex_index];

#ifndef GFX_UTIL_MESH_DISABLE_VERTEX
    // Fill the vertex buffer
    data_vertex[index_offset + 0] = v0;
    data_vertex[index_offset + 1] = v1;
    data_vertex[index_offset + 2] = v2;
#endif

#ifndef GFX_UTIL_MESH_DISABLE_TEXCOORD
    // Fill texcoord
    assert(idx0.texcoord_index != -1);
    data_uv[index_offset + 0] = attr_textcoord[idx0.texcoord_index];
    assert(idx1.texcoord_index != -1);
    data_uv[index_offset + 1] = attr_textcoord[idx1.texcoord_index];
    assert(idx2.texcoord_index != -1);
    data_uv[index_offset + 2] = attr_textcoord[idx2.texcoord_index];
#endif

#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
    // Fill the normal buffer
    if (!compute_normal) {
      assert(idx0.normal_index != -1);
      data_normal[index_offset + 0] = attr_normal[idx0.normal_index];
      assert(idx1.normal_index != -1);
      data_normal[index_offset + 1] = attr_normal[idx1.normal_index];
      assert(idx2.normal_index != -1);
      data_normal[index_offset + 2] = attr_normal[idx2.normal_index];
    }
    else {
      // Compute Normal (flat shading)
      const glm::vec3 e10 = v1 - v0;
      const glm::vec3 e20 = v2 - v0;
      const glm::vec3 N = glm::cross(e10, e20);
      if (!this->flat_normal) {
        idx0.normal_index = idx0.vertex_index;
        idx1.normal_index = idx1.vertex_index;
        idx2.normal_index = idx2.vertex_index;
        attr_normal[idx0.normal_index] += N;
        attr_normal[idx1.normal_index] += N;
        attr_normal[idx2.normal_index] += N;
      }
      else {
        data_normal[index_offset + 0] = N;
        data_normal[index_offset + 1] = N;
        data_normal[index_offset + 2] = N;
      }
    }
#endif // GFX_UTIL_MESH_DISABLE_NORMAL
  }

#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
  if (compute_normal && !this->flat_normal) { // Loop again to normalize normals
    for (size_t f = 0; f < num_of_faces; f++) {
      const size_t index_offset = 3 * f;
      const tinyobj::index_t& idx0 = indices[index_offset + 0];
      const tinyobj::index_t& idx1 = indices[index_offset + 1];
      const tinyobj::index_t& idx2 = indices[index_offset + 2];
      assert(idx0.normal_index != -1);
      data_normal[index_offset + 0] = glm::normalize(attr_normal[idx0.normal_index]);
      assert(idx1.normal_index != -1);
      data_normal[index_offset + 1] = glm::normalize(attr_normal[idx1.normal_index]);
      assert(idx2.normal_index != -1);
      data_normal[index_offset + 2] = glm::normalize(attr_normal[idx2.normal_index]);
    }
  }
#endif // GFX_UTIL_MESH_DISABLE_NORMAL
}

bool
TriangleArray::alloc(tinyobj::attrib_t& attrib)
{
  bool compute_normal = false;
#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
  if (attrib.normals.empty() || flat_normal) {
    compute_normal = true; // re-compute vertex normal
    attrib.normals = std::vector<tinyobj::real_t>(attrib.vertices.size(), 0.f);
  }
#endif

#ifndef GFX_UTIL_MESH_DISABLE_TEXCOORD
  if (attrib.texcoords.empty()) {
    throw std::runtime_error("texture coordinate does not exist");
  }
#endif
  return compute_normal;
}

void
TriangleIndex::clear()
{
#ifndef GFX_UTIL_MESH_DISABLE_VERTEX
  glDeleteBuffers(1, &vbo_vertex);
#endif
#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
  glDeleteBuffers(1, &vbo_normal);
#endif
#ifndef GFX_UTIL_MESH_DISABLE_TEXCOORD
  glDeleteBuffers(1, &vbo_texcoord);
#endif
  for (auto& m : meshes) {
    glDeleteBuffers(1, &m.vbo_index);
  }
}

void
TriangleIndex::create()
{
#ifndef GFX_UTIL_MESH_DISABLE_VERTEX
  glGenBuffers(1, &vbo_vertex);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_vertex);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * num_of_vertices * 3, vertices.get(), GL_STATIC_DRAW);
#endif
#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
  glGenBuffers(1, &vbo_normal);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * num_of_vertices * 3, normals.get(), GL_STATIC_DRAW);
#endif
#ifndef GFX_UTIL_MESH_DISABLE_TEXCOORD
  glGenBuffers(1, &vbo_texcoord);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_texcoord);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * num_of_vertices * 2, texcoords.get(), GL_STATIC_DRAW);
#endif
  // Generate a buffer for the indices as well
  for (auto& m : meshes) {
    glGenBuffers(1, &m.vbo_index);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.vbo_index);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m.size_indices * sizeof(unsigned int), m.indices.get(), GL_STATIC_DRAW);
  }
}

void
TriangleIndex::render(int layout_vertex, int layout_normal, int layout_texcoord)
{
#ifndef GFX_UTIL_MESH_DISABLE_VERTEX
  glBindBuffer(GL_ARRAY_BUFFER, vbo_vertex);
  glVertexAttribPointer(layout_vertex, // attribute. No particular reason for 0, but must
                                       // match the layout in the shader.
                        3, // size
                        GL_FLOAT, // type
                        GL_FALSE, // normalized?
                        0, // stride
                        (void*)0 // array buffer offset
  );
#else
  assert(layout_vertex == -1);
#endif
#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
  glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
  glVertexAttribPointer(layout_normal, // attribute. No particular reason for 1, but must
                                       // match the layout in the shader.
                        3, // size
                        GL_FLOAT, // type
                        GL_FALSE, // normalized?
                        0, // stride
                        (void*)0 // array buffer offset
  );
#else
  assert(layout_normal == -1);
#endif
#ifndef GFX_UTIL_MESH_DISABLE_TEXCOORD
  glBindBuffer(GL_ARRAY_BUFFER, vbo_texcoord);
  glVertexAttribPointer(layout_texcoord, // attribute. No particular reason for 1, but must
                                         // match the layout in the shader.
                        2, // size
                        GL_FLOAT, // type
                        GL_FALSE, // normalized?
                        0, // stride
                        (void*)0 // array buffer offset
  );
#else
  assert(layout_texcoord == -1);
#endif

  for (auto& m : meshes) {
    // Index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.vbo_index);

    // Draw the triangles !
    glDrawElements(GL_TRIANGLES, // mode
                   m.size_indices, // count
                   GL_UNSIGNED_INT, // type
                   (void*)0 // element array buffer offset
    );
  }
}

void
TriangleIndex::fill(size_t num_of_vertices,
                    size_t num_of_faces,
                    std::vector<tinyobj::index_t>& indices,
                    glm::vec3* attr_vertex,
                    glm::vec3* attr_normal,
                    glm::vec2* attr_textcoord,
                    bool compute_normal)
{
  assert(num_of_vertices != 0);

#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
  glm::vec3* data_normal = (glm::vec3*)this->normals.get();
#endif

#ifndef GFX_UTIL_MESH_DISABLE_TEXCOORD
  glm::vec2* data_texcoord = (glm::vec2*)this->texcoords.get();
#endif

  this->meshes.emplace_back();
  auto& self = this->meshes.back();

  self.size_indices = num_of_vertices;
  self.indices.reset(new unsigned int[self.size_indices]);
  auto* data_indices = self.indices.get();

  // Loop over faces (polygon)
  for (size_t f = 0; f < num_of_faces; f++) {
    const size_t index_offset = 3 * f;

    // Access to vertex
    tinyobj::index_t idx0 = indices[index_offset + 0];
    tinyobj::index_t idx1 = indices[index_offset + 1];
    tinyobj::index_t idx2 = indices[index_offset + 2];

    glm::vec3& v0 = attr_vertex[idx0.vertex_index];
    glm::vec3& v1 = attr_vertex[idx1.vertex_index];
    glm::vec3& v2 = attr_vertex[idx2.vertex_index];

    data_indices[index_offset + 0] = idx0.vertex_index;
    data_indices[index_offset + 1] = idx1.vertex_index;
    data_indices[index_offset + 2] = idx2.vertex_index;

#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
    if (compute_normal) {
      glm::vec3 e10 = v1 - v0;
      glm::vec3 e20 = v2 - v0;
      glm::vec3 N = glm::cross(e10, e20);
      data_normal[idx0.vertex_index] += N;
      data_normal[idx1.vertex_index] += N;
      data_normal[idx2.vertex_index] += N;
    }
    else {
      assert(idx0.normal_index != -1);
      data_normal[idx0.vertex_index] = attr_normal[idx0.normal_index];
      assert(idx1.normal_index != -1);
      data_normal[idx1.vertex_index] = attr_normal[idx1.normal_index];
      assert(idx2.normal_index != -1);
      data_normal[idx2.vertex_index] = attr_normal[idx2.normal_index];
    }
#endif

#ifndef GFX_UTIL_MESH_DISABLE_TEXCOORD
    assert(idx0.texcoord_index != -1);
    assert(idx1.texcoord_index != -1);
    assert(idx2.texcoord_index != -1);
    data_texcoord[idx0.vertex_index] = attr_textcoord[idx0.texcoord_index];
    data_texcoord[idx1.vertex_index] = attr_textcoord[idx1.texcoord_index];
    data_texcoord[idx2.vertex_index] = attr_textcoord[idx2.texcoord_index];
#endif
  }

#ifndef GFX_UTIL_MESH_DISABLE_NORMAL // Normalize normals
  for (size_t i = 0; i < this->num_of_vertices; i++)
    data_normal[i] = glm::normalize(data_normal[i]);
#endif
};

bool
TriangleIndex::alloc(tinyobj::attrib_t& attrib)
{
  this->num_of_vertices = attrib.vertices.size() / 3;

  bool compute_normal = false;

#ifndef GFX_UTIL_MESH_DISABLE_VERTEX
  // Copy vertices
  this->vertices.reset(new float[this->num_of_vertices * 3]);
  std::copy(&(attrib.vertices[0]), &(attrib.vertices[0]) + attrib.vertices.size(), this->vertices.get());
#endif

#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
  // Handle normals
  this->normals.reset(new float[this->num_of_vertices * 3]() /* zero initialization */);
  if (attrib.normals.empty()) {
    compute_normal = true;
  }
#endif

#ifndef GFX_UTIL_MESH_DISABLE_TEXCOORD
  // Handle texcoords
  this->texcoords.reset(new float[this->num_of_vertices * 2]);
  if (attrib.texcoords.empty()) {
    throw std::runtime_error("texture coordinate does not exist");
  }
#endif

  std::cout << "# vertices = " << attrib.vertices.size() << std::endl;
  std::cout << "# normals = " << attrib.normals.size() << std::endl;
  std::cout << "# texcoords = " << attrib.texcoords.size() << std::endl;

  return compute_normal;
}

//////////////////////////////////////////////////////////////////////////////
// Cube
//////////////////////////////////////////////////////////////////////////////

// Our vertices. Tree consecutive floats give a 3D vertex; Three consecutive
// vertices give a triangle. A cube has 6 faces with 2 triangles each, so this
// makes 6*2=12 triangles, and 12*3 vertices
const GLfloat CubeObject::g_vertex_buffer_data[] = {
  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, 1.0f,  1.0f, 1.0f,  1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
  1.0f,  -1.0f, 1.0f,  -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f,  1.0f,  -1.0f, 1.0f,  -1.0f,
  -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,  -1.0f, 1.0f,  -1.0f, 1.0f,  -1.0f, 1.0f,
  -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  1.0f, -1.0f, -1.0f, 1.0f,  1.0f,  -1.0f, 1.0f,  1.0f,
  1.0f,  1.0f,  1.0f,  -1.0f, -1.0f, 1.0f,  1.0f,  -1.0f, 1.0f, -1.0f, -1.0f, 1.0f,  1.0f,  1.0f,  1.0f,  -1.0f,
  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,  1.0f,  1.0f,  -1.0f, 1.0f,  -1.0f,
  -1.0f, 1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  -1.0f, 1.0f,  1.0f, 1.0f,  -1.0f, 1.0f
};

// One color for each vertex. They were generated randomly.
const GLfloat CubeObject::g_color_buffer_data[] = {
  0.583f, 0.771f, 0.014f, 0.609f, 0.115f, 0.436f, 0.327f, 0.483f, 0.844f, 0.822f, 0.569f, 0.201f, 0.435f, 0.602f,
  0.223f, 0.310f, 0.747f, 0.185f, 0.597f, 0.770f, 0.761f, 0.559f, 0.436f, 0.730f, 0.359f, 0.583f, 0.152f, 0.483f,
  0.596f, 0.789f, 0.559f, 0.861f, 0.639f, 0.195f, 0.548f, 0.859f, 0.014f, 0.184f, 0.576f, 0.771f, 0.328f, 0.970f,
  0.406f, 0.615f, 0.116f, 0.676f, 0.977f, 0.133f, 0.971f, 0.572f, 0.833f, 0.140f, 0.616f, 0.489f, 0.997f, 0.513f,
  0.064f, 0.945f, 0.719f, 0.592f, 0.543f, 0.021f, 0.978f, 0.279f, 0.317f, 0.505f, 0.167f, 0.620f, 0.077f, 0.347f,
  0.857f, 0.137f, 0.055f, 0.953f, 0.042f, 0.714f, 0.505f, 0.345f, 0.783f, 0.290f, 0.734f, 0.722f, 0.645f, 0.174f,
  0.302f, 0.455f, 0.848f, 0.225f, 0.587f, 0.040f, 0.517f, 0.713f, 0.338f, 0.053f, 0.959f, 0.120f, 0.393f, 0.621f,
  0.362f, 0.673f, 0.211f, 0.457f, 0.820f, 0.883f, 0.371f, 0.982f, 0.099f, 0.879f
};

void
CubeObject::clear()
{
  delete_vbo(vertex_buffer_id);
  delete_vbo(color_buffer_id);
}

void
CubeObject::create()
{
  vertex_buffer_id = Object::create_vbo(sizeof(g_vertex_buffer_data) / sizeof(float), g_vertex_buffer_data);
  color_buffer_id = Object::create_vbo(sizeof(g_color_buffer_data) / sizeof(float), g_color_buffer_data);

  name = "cube";
  center = glm::vec3(0.f, 0.f, 0.f);
  lower = glm::vec3(-1);
  upper = glm::vec3(1);
  scale = 1.f;
}

void
CubeObject::render(int layout_vertex, int layout_normal, int layout_texcoord)
{
#ifndef GFX_UTIL_MESH_DISABLE_VERTEX
  enable_vbo(vertex_buffer_id, layout_vertex, 3);
#else
  assert(layout_vertex == -1);
#endif
#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
  enable_vbo(color_buffer_id, layout_normal, 3);
#else
  assert(layout_normal == -1);
#endif
  assert(layout_texcoord == -1);
  glDrawArrays(GL_TRIANGLES, 0, 12 * 3);
}

//////////////////////////////////////////////////////////////////////////////
// Cone
//////////////////////////////////////////////////////////////////////////////

ConeObject::ConeObject(int resolution, float radius, float height) : resolution(resolution)
{
  float angle = 2.f * glm::pi<float>() / resolution;
  float cosTheta = radius / glm::sqrt(height * height + radius * radius);
  float sinTheta = glm::sqrt(1 - cosTheta * cosTheta);

  // Top
  for (int r = 1; r <= resolution; ++r) {
    float sinPhi0 = glm::sin(angle * (r - 1));
    float sinPhi1 = glm::sin(angle * r);
    float cosPhi0 = glm::cos(angle * (r - 1));
    float cosPhi1 = glm::cos(angle * r);
    vertices.push_back(0);
    vertices.push_back(0);
    vertices.push_back(height);
    vertices.push_back(radius * cosPhi0);
    vertices.push_back(radius * sinPhi0);
    vertices.push_back(0);
    vertices.push_back(radius * cosPhi1);
    vertices.push_back(radius * sinPhi1);
    vertices.push_back(0);
    normals.push_back(0);
    normals.push_back(0);
    normals.push_back(1);
    normals.push_back(sinTheta * cosPhi0);
    normals.push_back(sinTheta * sinPhi0);
    normals.push_back(cosTheta);
    normals.push_back(sinTheta * cosPhi1);
    normals.push_back(sinTheta * sinPhi1);
    normals.push_back(cosTheta);
  }

  // Bottom
  for (int r = 1; r <= resolution; ++r) {
    float sinPhi0 = glm::sin(angle * (r - 1));
    float sinPhi1 = glm::sin(angle * r);
    float cosPhi0 = glm::cos(angle * (r - 1));
    float cosPhi1 = glm::cos(angle * r);
    vertices.push_back(0);
    vertices.push_back(0);
    vertices.push_back(0);
    vertices.push_back(radius * cosPhi1);
    vertices.push_back(radius * sinPhi1);
    vertices.push_back(0);
    vertices.push_back(radius * cosPhi0);
    vertices.push_back(radius * sinPhi0);
    vertices.push_back(0);
    normals.push_back(0);
    normals.push_back(0);
    normals.push_back(-1);
    normals.push_back(0);
    normals.push_back(0);
    normals.push_back(-1);
    normals.push_back(0);
    normals.push_back(0);
    normals.push_back(-1);
  }

  // Statistics
  name = "cone";
  center = glm::vec3(0.f, 0.f, height / 4.f);
  lower = glm::vec3(-radius, -radius, 0);
  upper = glm::vec3(radius, radius, height);
  scale = 1.f / glm::max(height, radius);
}

void
ConeObject::clear()
{
  delete_vbo(vertex_buffer_id);
  delete_vbo(normal_buffer_id);
}

void
ConeObject::create()
{
  vertex_buffer_id = Object::create_vbo(sizeof(float) * vertices.size(), vertices.data());
  normal_buffer_id = Object::create_vbo(sizeof(float) * normals.size(), normals.data());
}

void
ConeObject::render(int layout_vertex, int layout_normal, int layout_texcoord)
{
#ifndef GFX_UTIL_MESH_DISABLE_VERTEX
  enable_vbo(vertex_buffer_id, layout_vertex, 3);
#else
  assert(layout_vertex == -1);
#endif
#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
  enable_vbo(normal_buffer_id, layout_normal, 3);
#else
  assert(layout_normal == -1);
#endif
  assert(layout_texcoord == -1);

  // Draw the triangle !
  glDrawArrays(GL_TRIANGLES, 0, 2 * resolution * 3);
}

//////////////////////////////////////////////////////////////////////////////
// Cylinder
//////////////////////////////////////////////////////////////////////////////

CylinderObject::CylinderObject(int resolution, float radius, float height) : resolution(resolution)
{
  float angle = 2.f * glm::pi<float>() / resolution;

  // Top
  for (int r = 1; r <= resolution; ++r) {
    float sinPhi0 = glm::sin(angle * (r - 1));
    float sinPhi1 = glm::sin(angle * r);
    float cosPhi0 = glm::cos(angle * (r - 1));
    float cosPhi1 = glm::cos(angle * r);
    vertices.push_back(0);
    vertices.push_back(0);
    vertices.push_back(height);
    vertices.push_back(radius * cosPhi0);
    vertices.push_back(radius * sinPhi0);
    vertices.push_back(height);
    vertices.push_back(radius * cosPhi1);
    vertices.push_back(radius * sinPhi1);
    vertices.push_back(height);
    normals.push_back(0);
    normals.push_back(0);
    normals.push_back(1);
    normals.push_back(0);
    normals.push_back(0);
    normals.push_back(1);
    normals.push_back(0);
    normals.push_back(0);
    normals.push_back(1);
  }

  // Boundary
  for (int r = 1; r <= resolution; ++r) {
    float sinPhi0 = glm::sin(angle * (r - 1));
    float sinPhi1 = glm::sin(angle * r);
    float cosPhi0 = glm::cos(angle * (r - 1));
    float cosPhi1 = glm::cos(angle * r);

    // face 1
    vertices.push_back(radius * cosPhi0);
    vertices.push_back(radius * sinPhi0);
    vertices.push_back(0);

    vertices.push_back(radius * cosPhi1);
    vertices.push_back(radius * sinPhi1);
    vertices.push_back(0);

    vertices.push_back(radius * cosPhi1);
    vertices.push_back(radius * sinPhi1);
    vertices.push_back(height);

    // face 2
    vertices.push_back(radius * cosPhi0);
    vertices.push_back(radius * sinPhi0);
    vertices.push_back(0);

    vertices.push_back(radius * cosPhi1);
    vertices.push_back(radius * sinPhi1);
    vertices.push_back(height);

    vertices.push_back(radius * cosPhi0);
    vertices.push_back(radius * sinPhi0);
    vertices.push_back(height);

    // normal for face 1
    normals.push_back(cosPhi0);
    normals.push_back(sinPhi0);
    normals.push_back(0);

    normals.push_back(cosPhi1);
    normals.push_back(sinPhi1);
    normals.push_back(0);

    normals.push_back(cosPhi1);
    normals.push_back(sinPhi1);
    normals.push_back(0);

    // normal for face 2
    normals.push_back(cosPhi0);
    normals.push_back(sinPhi0);
    normals.push_back(0);

    normals.push_back(cosPhi1);
    normals.push_back(sinPhi1);
    normals.push_back(0);

    normals.push_back(cosPhi0);
    normals.push_back(sinPhi0);
    normals.push_back(0);
  }

  // Bottom
  for (int r = 1; r <= resolution; ++r) {
    float sinPhi0 = glm::sin(angle * (r - 1));
    float sinPhi1 = glm::sin(angle * r);
    float cosPhi0 = glm::cos(angle * (r - 1));
    float cosPhi1 = glm::cos(angle * r);
    vertices.push_back(0);
    vertices.push_back(0);
    vertices.push_back(0);
    vertices.push_back(radius * cosPhi1);
    vertices.push_back(radius * sinPhi1);
    vertices.push_back(0);
    vertices.push_back(radius * cosPhi0);
    vertices.push_back(radius * sinPhi0);
    vertices.push_back(0);
    normals.push_back(0);
    normals.push_back(0);
    normals.push_back(-1);
    normals.push_back(0);
    normals.push_back(0);
    normals.push_back(-1);
    normals.push_back(0);
    normals.push_back(0);
    normals.push_back(-1);
  }

  // Statistics
  name = "cylinder";
  center = glm::vec3(0.f, 0.f, height / 2.f);
  lower = glm::vec3(-radius, -radius, 0);
  upper = glm::vec3(radius, radius, height);
  scale = 1.f / glm::max(height, radius);
}

void
CylinderObject::clear()
{
  delete_vbo(vertex_buffer_id);
  delete_vbo(normal_buffer_id);
}

void
CylinderObject::create()
{
  vertex_buffer_id = create_vbo(sizeof(float) * vertices.size(), vertices.data());
  normal_buffer_id = create_vbo(sizeof(float) * normals.size(), normals.data());
}

void
CylinderObject::render(int layout_vertex, int layout_normal, int layout_texcoord)
{
#ifndef GFX_UTIL_MESH_DISABLE_VERTEX
  enable_vbo(vertex_buffer_id, layout_vertex, 3);
#else
  assert(layout_vertex == -1);
#endif
#ifndef GFX_UTIL_MESH_DISABLE_NORMAL
  enable_vbo(normal_buffer_id, layout_normal, 3);
#else
  assert(layout_normal == -1);
#endif
  assert(layout_texcoord == -1);

  // Draw the triangle !
  glDrawArrays(GL_TRIANGLES, 0, 4 * resolution * 3);
}

#endif // GFX_UTIL_MESH_IMPLEMENTATION
