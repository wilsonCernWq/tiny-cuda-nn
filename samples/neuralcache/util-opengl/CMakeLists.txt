# here we are building one target for each OpenGL version
foreach(suffix ${ALL_GL_VERSIONS})

  # create the library
  add_library(util-opengl-${suffix} util.cpp)

  # add library dependencies
  target_link_libraries(util-opengl-${suffix} PUBLIC
    glad-${suffix} imgui-${suffix} glm ${GFX_LIBRARIES})

  # setup include directries for this target (modern cmake)
  target_include_directories(util-opengl-${suffix} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

  # and other properties
  set_target_properties(util-opengl-${suffix} PROPERTIES CXX_STANDARD 11)

  # add definition
  string(REPLACE "." "_" str "${suffix}")
  string(REPLACE "-" "_" str "${str}")
  string(TOUPPER "${str}" str)
  target_compile_definitions(util-opengl-${suffix} PUBLIC ENABLE_OPENGL_${str})

endforeach()
