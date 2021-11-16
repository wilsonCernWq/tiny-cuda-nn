#ifndef TEXTURE_INTEROP_H
#define TEXTURE_INTEROP_H

#include "util.h"

#include <tiny-cuda-nn/common.h>
#include <cuda_gl_interop.h>

struct OpenGLTexture {
private:
    GLuint                 opengl_texture;
    cudaGraphicsResource_t cuda_resource_view;

public:
    OpenGLTexture(int width, int height)
    {
        check_error_gl("Create OpenGL Texture");
        glGenTextures(1, &opengl_texture);
        glBindTexture(GL_TEXTURE_2D, opengl_texture); 
        {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        check_error_gl("Create OpenGL Texture ... OK");

        resize(width, height);
    }

    ~OpenGLTexture()
    {
        glDeleteTextures(1, &opengl_texture);
    }

    void resize(int width, int height)
    {
        check_error_gl("Resize OpenGL Texture");
        glBindTexture(GL_TEXTURE_2D, opengl_texture); 
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
        check_error_gl("Resize OpenGL Texture ... OK");

        CUDA_CHECK_THROW(cudaGraphicsGLRegisterImage(&cuda_resource_view, opengl_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
    }

    void bindOpenGLTexture()
    {
        glBindTexture(GL_TEXTURE_2D, opengl_texture);
        check_error_gl("Bind OpenGL Texture");
    }

    void unbindOpenGLTexture()
    {
        glBindTexture(GL_TEXTURE_2D, 0);
        check_error_gl("Unbind OpenGL Texture");
    }

    void mapCudaArray(cudaArray_t& array)
    {
        // We want to copy cuda_dest_resource data to the texture
        // map buffer objects to get CUDA device pointers
        CUDA_CHECK_THROW(cudaGraphicsMapResources(1, &cuda_resource_view, 0));
        CUDA_CHECK_THROW(cudaGraphicsSubResourceGetMappedArray(&array, cuda_resource_view, 0, 0));
    }

    void unmapCudaArray(cudaArray_t&)
    {
        CUDA_CHECK_THROW(cudaGraphicsUnmapResources(1, &cuda_resource_view, 0));
    }
};

#endif // TEXTURE_INTEROP_H
