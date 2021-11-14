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

add_executable(bin2c bin2c/bin2c.c)
include(bin2c/target_add_embeded_shaders.cmake)

include_directories(${CMAKE_CURRENT_LIST_DIR})

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
add_subdirectory(glm-0.9.9.8)
add_subdirectory(imgui-1.79)
add_subdirectory(util-opengl)

add_subdirectory(util-cuda)

add_library(neuralcache 
  neuralcache.cu 
  tinyexr_wrapper.cpp)
target_link_libraries(neuralcache PUBLIC 
  ${CUDA_LIBRARIES} 
  tiny-cuda-nn curand 
  util-cuda
  util-opengl-core-3.3)
target_compile_options(neuralcache PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)


set(thisapp neural_image_learner)
add_executable(${thisapp} main.cpp)
set_target_properties(${thisapp} PROPERTIES CXX_STANDARD 17)
target_include_directories(${thisapp} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${thisapp} PUBLIC 
  util-cuda
  util-opengl-core-3.3
  neuralcache)
target_compile_options(${thisapp} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_add_embeded_shaders(${thisapp} 
  OUTPUT_NAME ${CMAKE_CURRENT_BINARY_DIR}/shaders.h 
  SHADERS
  vshader_quad=shader_passthrough.glsl
  fshader_quad=shader_wobbly_texture.glsl)
