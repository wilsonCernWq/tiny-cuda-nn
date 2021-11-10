#include "neural_cache.hpp"

#include "../tinyexr_wrapper.h"

#include "util.h"

#include <tiny-cuda-nn/misc_kernels.h>
#include <tiny-cuda-nn/config.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace tcnn;
using precision_t = network_precision_t;

static GLuint view_opengl_texture;
static cudaGraphicsResource_t view_cuda_resource;

GPUMemory<float> load_image(const std::string &filename, int &width, int &height)
{
	float *out; // width * height * RGBA
	load_exr(&out, &width, &height, filename.c_str());

	GPUMemory<float> result(width * height * 4);
	result.copy_from_host(out);
	free(out); // release memory of image data

	return result;
}

NeuralImageCache::NeuralImageCache(std::string filename)
{
    // First step: load an image that we'd like to learn
	GPUMemory<float> image = load_image(filename.c_str(), width, height);
	std::cout << "Image size: " << width << " " << height << std::endl;

    initialize();

    // We want to copy cuda_dest_resource data to the texture
    // map buffer objects to get CUDA device pointers
    CUDA_CHECK_THROW(cudaGraphicsMapResources(1, &view_cuda_resource, 0));

    cudaArray_t texture_ptr;
    CUDA_CHECK_THROW(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, view_cuda_resource, 0, 0));

    int size_tex_data = sizeof(float) * width * height * 4;
    CUDA_CHECK_THROW(cudaMemcpyToArray(texture_ptr, 0, 0, image.data(), size_tex_data, cudaMemcpyDeviceToDevice));

    CUDA_CHECK_THROW(cudaGraphicsUnmapResources(1, &view_cuda_resource, 0));
}

void NeuralImageCache::initialize() 
{
    check_error_gl("NeuralImageCache Create OpenGL Texture Source");
    glGenTextures(1, &view_opengl_texture);
    glBindTexture(GL_TEXTURE_2D, view_opengl_texture); 
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    check_error_gl("NeuralImageCache Create OpenGL Texture Source ... OK");

    CUDA_CHECK_THROW(cudaGraphicsGLRegisterImage(&view_cuda_resource, view_opengl_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
}

void NeuralImageCache::resize(int width, int height) 
{
    CUDA_CHECK_THROW(cudaGraphicsUnregisterResource(view_cuda_resource));
    {
        check_error_gl("NeuralImageCache Resize OpenGL Texture Source");
        glBindTexture(GL_TEXTURE_2D, view_opengl_texture);    
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
        check_error_gl("NeuralImageCache Resize OpenGL Texture Source ... OK");
    }    
    CUDA_CHECK_THROW(cudaGraphicsGLRegisterImage(&view_cuda_resource, view_opengl_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
}

void NeuralImageCache::bindTexture()
{
    glBindTexture(GL_TEXTURE_2D, view_opengl_texture);
    check_error_gl("NeuralImageCache Bind OpenGL Texture");
}
