set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
if(TARGET OpenGL::OpenGL)
  list(APPEND GFX_LIBRARIES OpenGL::OpenGL)
else()
  list(APPEND GFX_LIBRARIES OpenGL::GL)
endif()
if(TARGET OpenGL::GLU)
  list(APPEND GFX_LIBRARIES OpenGL::GLU)
endif()
if(TARGET OpenGL::GLX)
  list(APPEND GFX_LIBRARIES OpenGL::GLX)
endif()

set(GLFW_USE_OSMESA OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_INSTALL OFF)
add_subdirectory(glfw-3.3.2)
mark_as_advanced(GLFW_INSTALL 
  GLFW_BUILD_DOCS GLFW_BUILD_TESTS GLFW_BUILD_EXAMPLES
  GLFW_USE_OSMESA GLFW_USE_WAYLAND GLFW_VULKAN_STATIC)
list(APPEND GFX_LIBRARIES glfw)

set(ALL_GL_VERSIONS core-3.3)
add_subdirectory(glad-core-3.3)

add_subdirectory(imgui-1.79)

add_subdirectory(util)

add_executable(d11_render_to_texture main.cpp)
target_include_directories(d11_render_to_texture PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(d11_render_to_texture PUBLIC util-core-3.3)
set_target_properties(d11_render_to_texture PROPERTIES CXX_STANDARD 11)
target_add_embeded_shaders(d11_render_to_texture 
  OUTPUT_NAME ${CMAKE_CURRENT_BINARY_DIR}/shaders.h 
  SHADERS
  vshader_rtt=shader_rtt_vertex.glsl
  fshader_rtt=shader_rtt_fragment.glsl
  vshader_quad=shader_passthrough.glsl
  fshader_quad=shader_wobbly_texture.glsl)
