//===========================================================================//
//                                                                           //
// Copyright(c) ECS 175 (2020)                                               //
// University of California, Davis                                           //
// MIT Licensed                                                              //
//                                                                           //
//===========================================================================//

// Include standard headers
#include <chrono>
#include <string>
#include <cstdio>
#include <cstdlib>

#include "shaders.h"
#include "util.h"
#include "util_shader.h"
GLFWwindow* window;

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
bool imgui_enabled = true;

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
using namespace glm;

#include "util_camera.h"
ArcballCamera camera(vec3(4, 3, -3), vec3(0, 0, 0), vec3(0, 1, 0));

#include "neuralcache.hpp"
size_t steps = 0;
float tloss = 0.f;
float gloss = 0.f;

int leve_of_detail = 0;
int tile_size_rank = 8;

bool control_pause_training = false;
int control_training_mode = (int)NeuralImageCache::UNIFORM_RANDOM;

static void
error(int error, const char* description)
{
  fprintf(stderr, "Error: %s\n", description);
}

static void
keyboard(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) 
  {
    glfwSetWindowShouldClose(window, GLFW_TRUE); // close window
  }
  else if (key == GLFW_KEY_G && action == GLFW_PRESS) 
  {
    imgui_enabled = !imgui_enabled;
  }
}

static void
cursor(GLFWwindow* window, double xpos, double ypos)
{
  ImGuiIO& io = ImGui::GetIO();
  if (!io.WantCaptureMouse) 
  {
    int left_state  = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
    int right_state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);

    int width, height;
    glfwGetWindowSize(window, &width, &height);

    static vec2 prev_cursor;
    vec2 cursor((xpos / width - 0.5f) * 2.f, (0.5f - ypos / height) * 2.f);

    // right click -> zoom
    if (right_state == GLFW_PRESS || right_state == GLFW_REPEAT) 
    {
      camera.zoom(cursor.y - prev_cursor.y);
    }

    // left click -> rotate
    if (left_state == GLFW_PRESS || left_state == GLFW_REPEAT) 
    {
      camera.rotate(prev_cursor, cursor);
    }

    prev_cursor = cursor;
  }
}

void
gui(bool* p_open)
{
  // measure frame rate
  static float fps = 0.0f;
  {
    static bool opened = false;
    static int frames = 0;
    static auto start = std::chrono::system_clock::now();
    if (!opened) 
    {
      start = std::chrono::system_clock::now();
      frames = 0;
    }
    std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start;
    ++frames;
    if (frames % 10 == 0 || frames == 1) // dont update this too frequently
      fps = frames / elapsed_seconds.count();
    opened = *p_open;
  }

  ivec2 window_size, framebuffer_size;
  glfwGetWindowSize(window, &window_size.x, &window_size.y);
  glfwGetFramebufferSize(window, &framebuffer_size.x, &framebuffer_size.y);

  // draw a fixed GUI window
  const float distance = 10.0f;
  static int corner = 0;
  ImVec2 window_pos = ImVec2((corner & 1) ? ImGui::GetIO().DisplaySize.x - distance : distance,
                             (corner & 2) ? ImGui::GetIO().DisplaySize.y - distance : distance);
  ImVec2 window_pos_pivot = ImVec2((corner & 1) ? 1.0f : 0.0f, (corner & 2) ? 1.0f : 0.0f);
  ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.3f)); // Transparent background

  const auto flags = 
    ImGuiWindowFlags_NoTitleBar | 
    ImGuiWindowFlags_NoResize | 
    ImGuiWindowFlags_AlwaysAutoResize |
    ImGuiWindowFlags_NoMove | 
    ImGuiWindowFlags_NoSavedSettings;

  if (ImGui::Begin("Information", NULL, flags)) 
  {
    ImGui::Text("FPS (Hz): %.f\n", fps);
    ImGui::Text("Training Loss: %.7f\n", tloss);
    ImGui::Text("Groundtruth Loss: %.7f\n", gloss);
    ImGui::Text("Steps: %llu\n", steps);
    ImGui::SliderInt("Level Of Detail", &leve_of_detail, 0, 10);
    ImGui::SliderInt("Tile Size Rank", &tile_size_rank, 0, 10);

    ImGui::Checkbox("Control Pause Training", &control_pause_training);

    const char* items[] = { 
      "UNIFORM_RANDOM", 
      "UNIFORM_RANDOM_QUANTIZED", 
      "TILE_BASED_SIMPLE", 
      "TILE_BASED_MIXTURE", 
      "TILE_BASED_EVENLY" 
    };
    ImGui::Combo("Training Mode", &control_training_mode, items, IM_ARRAYSIZE(items));

    ImGui::End();
  }

  ImGui::PopStyleColor();
}

void
init()
{
  // Initialise GLFW
  if (!glfwInit()) 
  {
    throw std::runtime_error("Failed to initialize GLFW");
  }

  const char* glsl_version = "#version 150"; // GL 3.3 + GLSL 150
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  // Open a window and create its OpenGL context
  window = glfwCreateWindow(1200, 600, "ECS 175 (press 'g' to display GUI)", NULL, NULL);
  if (window == NULL) 
  {
    glfwTerminate();
    throw std::runtime_error("Failed to open GLFW window. If you have a GPU that is "
                             "not 3.3 compatible, try a lower OpenGL version.");
  }

  glfwSetKeyCallback(window, keyboard);
  glfwSetCursorPosCallback(window, cursor);

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // Load GLAD symbols
  int err = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) == 0;
  if (err)
    throw std::runtime_error("Failed to initialize OpenGL loader!");

  // Ensure we can capture the escape key being pressed below
  glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

  // ImGui
  {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    // Setup Dear ImGui style
    ImGui::StyleColorsDark(); // or ImGui::StyleColorsClassic();
    // Initialize Dear ImGui
    ImGui_ImplGlfw_InitForOpenGL(window, true /* 'true' -> allow imgui to capture keyboard inputs */);
    ImGui_ImplOpenGL3_Init(glsl_version);
  }

  // Dark blue background (avoid using black)
  glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

  // Enable depth test
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS); // Accept fragment if it closer to the camera than the former one
}

int
main(const int argc, const char** argv)
{
  if (argc > 1)
    control_training_mode = std::stoi(std::string(argv[1]));

  init();

  NeuralImageCache cache("../data/images/albert.exr");
  // NeuralImageCache cache("../data/images/W8B0747.exr");

  ivec2 window_size, framebuffer_size;
  glfwGetWindowSize(window, &window_size.x, &window_size.y);
  glfwGetFramebufferSize(window, &framebuffer_size.x, &framebuffer_size.y);

  // Create and compile our GLSL program from the shaders
  GLuint program_quad = load_program_from_embedding(vshader_quad, vshader_quad_size, fshader_quad, fshader_quad_size);
  GLuint rendered_texture_id = glGetUniformLocation(program_quad, "rendered_texture");
  GLuint time_id = glGetUniformLocation(program_quad, "time");

  // A shared vertex array
  GLuint vertex_array_id;
  glGenVertexArrays(1, &vertex_array_id);
  glBindVertexArray(vertex_array_id);

  // The fullscreen quad
  static const GLfloat g_quad_vertex_buffer_data[] = {
    -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f,
  };
  GLuint vertex_buffer_quad;
  glGenBuffers(1, &vertex_buffer_quad);
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_quad);
  glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

  // ---------------------------------------------
  // Rendering loop
  // ---------------------------------------------
  auto draw = [&](bool draw_inference) 
  {
      // Use our shader
      glUseProgram(program_quad);
      glActiveTexture(GL_TEXTURE0); // Bind our texture in Texture Unit 0
      glUniform1f(time_id, (float)(glfwGetTime() * 10.0f));
      glUniform1i(rendered_texture_id, 0); // Set our sampler to use Texture Unit 0
      if (draw_inference) {
        cache.bindInferenceTexture();
      }
      else {
        cache.bindReferenceTexture();
      }

      // 1st attribute buffer : vertices
      glEnableVertexAttribArray(0);
      glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_quad);
      glVertexAttribPointer(0, // attribute 0. No particular reason for 0, but must match the layout in the shader.
                            3, // size
                            GL_FLOAT, // type
                            GL_FALSE, // normalized?
                            0, // stride
                            (void*)0 // array buffer offset
      );

      // Draw the triangles !
      glDrawArrays(GL_TRIANGLES, 0, 6); // 2*3 indices starting at 0 -> 2 triangles
      glDisableVertexAttribArray(0);
  };

  glDisable(GL_DEPTH_TEST);

  do {

    // Render to the screen
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, framebuffer_size.x/2, framebuffer_size.y);
    {
        draw(false);
    }
    glViewport(framebuffer_size.x/2, 0, framebuffer_size.x/2, framebuffer_size.y);
    {
        draw(true);
    }

    cache.setLod(leve_of_detail);
    cache.setTileSize(1 << tile_size_rank);

    if (!control_pause_training) {
      static int frames = 0;

      if (frames % 10 == 0) { // dont update this too frequently
        cache.trainingStats(steps, tloss, gloss);
      }

      cache.train(2, (NeuralImageCache::SamplingMode)control_training_mode);
      cache.renderInference();
      cache.renderReference();

      ++frames;

      if (steps > 20000) 
        break;
    }

    // Draw GUI
    {
      // Initialization
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      // - Uncomment below to show ImGui demo window
      if (imgui_enabled) gui(&imgui_enabled);

      // Render GUI
      ImGui::Render();
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    // Swap buffers
    glfwSwapBuffers(window);
    glfwPollEvents();

  }
  // Check if the ESC key was pressed or the window was closed
  while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

  // Cleanup
  glDeleteBuffers(1, &vertex_buffer_quad);
  glDeleteVertexArrays(1, &vertex_array_id);
  glDeleteProgram(program_quad);

  // Cleanup ImGui
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  // Close OpenGL window and terminate GLFW
  glfwTerminate();

  return 0;
}
