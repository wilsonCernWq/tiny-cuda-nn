add_library(util-cuda
  helper_cuda_texture.cu 
  helper_cuda_texture.h
  helper_math.h)
target_link_libraries(util-cuda PUBLIC ${CUDA_LIBRARIES})
target_compile_options(util-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_include_directories(util-cuda PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
