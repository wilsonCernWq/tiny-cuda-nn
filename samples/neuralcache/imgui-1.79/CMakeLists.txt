foreach(suffix ${ALL_GL_VERSIONS})

  add_library(imgui-${suffix}
    imgui.cpp
    imgui_draw.cpp
    imgui_demo.cpp
    imgui_widgets.cpp
    examples/imgui_impl_glfw.cpp
    examples/imgui_impl_opengl3.cpp)
  target_link_libraries(imgui-${suffix} glad-${suffix} ${GFX_LIBRARIES})
  target_include_directories(imgui-${suffix} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/examples/opengl3_example>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/examples>)
  target_compile_definitions(imgui-${suffix} PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)
  
endforeach()
