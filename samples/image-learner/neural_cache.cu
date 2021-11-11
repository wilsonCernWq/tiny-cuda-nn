#include "neural_cache.hpp"

#include "util.h"

#include "../tinyexr_wrapper.h"
#include "../helper_math.h"

#include <tiny-cuda-nn/misc_kernels.h>
#include <tiny-cuda-nn/config.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cassert>

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

using namespace tcnn;
using precision_t = network_precision_t;

static GPUMemory<float> load_image(const std::string &filename, int &width, int &height)
{
	float *out; // width * height * RGBA
	load_exr(&out, &width, &height, filename.c_str());

	GPUMemory<float> result(width * height * 4);
	result.copy_from_host(out);
	free(out); // release memory of image data

	return result;
}

//  A key benefit of using the new surface objects is that we don't need any global
//  binding points anymore. We can directly pass them as function arguments.

__global__ void
d_mipmap(cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, uint32_t imageW, uint32_t imageH)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    float px = 1.0/float(imageW);
    float py = 1.0/float(imageH);

    if ((x < imageW) && (y < imageH))
    {
        // take the average of 4 samples

        // we are using the normalized access to make sure non-power-of-two textures
        // behave well when downsized.
        float4 color =
            (tex2D<float4>(mipInput,(x + 0) * px, (y + 0) * py)) +
            (tex2D<float4>(mipInput,(x + 1) * px, (y + 0) * py)) +
            (tex2D<float4>(mipInput,(x + 1) * px, (y + 1) * py)) +
            (tex2D<float4>(mipInput,(x + 0) * px, (y + 1) * py));

        color /= 4.0;

        surf2Dwrite(color, mipOutput, x * sizeof(float4), y);
    }
}

void generate_mipmaps(cudaMipmappedArray_t mipmapArray, cudaExtent size)
{
    size_t width    = size.width;
    size_t height   = size.height;

    uint32_t level = 0;

    while (width != 1 || height != 1)
    {
        width     /= 2;
        width      = MAX((size_t)1, width);
        height    /= 2;
        height     = MAX((size_t)1, height);

        cudaArray_t levelFrom;
        CUDA_CHECK_THROW(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));
        cudaArray_t levelTo;
        CUDA_CHECK_THROW(cudaGetMipmappedArrayLevel(&levelTo,   mipmapArray, level + 1));

        cudaExtent levelToSize;
        CUDA_CHECK_THROW(cudaArrayGetInfo(NULL, &levelToSize, NULL, levelTo));
        assert(levelToSize.width  == width);
        assert(levelToSize.height == height);
        assert(levelToSize.depth  == 0);

        // generate texture object for reading
        cudaResourceDesc texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));
        texRes.resType         = cudaResourceTypeArray;
        texRes.res.array.array = levelFrom;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.normalizedCoords = true;
        texDesc.addressMode[0]   = cudaAddressModeClamp;
        texDesc.addressMode[1]   = cudaAddressModeClamp;
        texDesc.addressMode[2]   = cudaAddressModeClamp;

        cudaTextureObject_t texInput;
        CUDA_CHECK_THROW(cudaCreateTextureObject(&texInput, &texRes, &texDesc, NULL));

        // generate surface object for writing
        cudaResourceDesc surfRes;
        memset(&surfRes,0,sizeof(cudaResourceDesc));
        surfRes.resType = cudaResourceTypeArray;
        surfRes.res.array.array = levelTo;

        cudaSurfaceObject_t surfOutput;
        CUDA_CHECK_THROW(cudaCreateSurfaceObject(&surfOutput, &surfRes));

        // run mipmap kernel
        dim3 blockSize(16,16,1);
        dim3 gridSize(((uint32_t)width+blockSize.x-1)/blockSize.x, ((uint32_t)height+blockSize.y-1)/blockSize.y,1);
        d_mipmap<<<gridSize, blockSize>>>(surfOutput, texInput, (uint32_t)width, (uint32_t)height);

        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        CUDA_CHECK_THROW(cudaGetLastError());
        CUDA_CHECK_THROW(cudaDestroySurfaceObject(surfOutput));
        CUDA_CHECK_THROW(cudaDestroyTextureObject(texInput));

        level++;
    }
}

uint32_t get_mipmap_levels(cudaExtent size)
{
    size_t sz = MAX(MAX(size.width, size.height), size.depth);
    uint32_t levels = 0;
    while (sz)
    {
        sz /= 2;
        levels++;
    }
    return levels;
}

cudaTextureObject_t create_mipmap_rgba32f_texture(void* data, int width, int height)
{
    // how many mipmaps we need
    cudaExtent extent;
    extent.width  = width;
    extent.height = height;
    extent.depth = 0;
    uint32_t levels = get_mipmap_levels(extent);
        
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaMipmappedArray_t mipmapArray;
    CUDA_CHECK_THROW(cudaMallocMipmappedArray(&mipmapArray, &desc, extent, levels));

    // upload level 0
    cudaArray_t level0;
    CUDA_CHECK_THROW(cudaGetMipmappedArrayLevel(&level0, mipmapArray, 0));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr       = make_cudaPitchedPtr(data, width * 4 * sizeof(float), width, height);
    copyParams.dstArray     = level0;
    copyParams.extent       = extent;
    copyParams.extent.depth = 1;
    copyParams.kind         = cudaMemcpyHostToDevice;
    CUDA_CHECK_THROW(cudaMemcpy3D(&copyParams));

    // compute rest of mipmaps based on level 0
    generate_mipmaps(mipmapArray, extent);

    // generate bindless texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType            = cudaResourceTypeMipmappedArray;
    resDesc.res.mipmap.mipmap  = mipmapArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.mipmapFilterMode = cudaFilterModeLinear;
    texDesc.normalizedCoords = true;
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.addressMode[2]   = cudaAddressModeClamp;
    texDesc.maxMipmapLevelClamp = float(levels - 1);

    cudaTextureObject_t texture;
    CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &resDesc, &texDesc, NULL));
    return texture;
}

cudaTextureObject_t create_pitch2d_rgba32f_texture(void* data, int width, int height)
{
    // Second step: create a cuda texture out of this image. It'll be used to generate training data efficiently on the fly
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = data;
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	resDesc.res.pitch2D.width = width;
	resDesc.res.pitch2D.height = height;
	resDesc.res.pitch2D.pitchInBytes = width * 4 * sizeof(float);

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.normalizedCoords = true;
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;

	cudaResourceViewDesc viewDesc;
	memset(&viewDesc, 0, sizeof(viewDesc));
	viewDesc.format = cudaResViewFormatFloat4;
	viewDesc.width = width;
	viewDesc.height = height;

	cudaTextureObject_t texture;
	CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &resDesc, &texDesc, &viewDesc));
    return texture;
}

static std::tuple<GPUMemory<float>, cudaTextureObject_t> generate_image_texture(std::string filename, int& width, int& height)
{
    // First step: load an image that we'd like to learn
	GPUMemory<float> image = load_image(filename.c_str(), width, height);
    std::cout << "image size: " << width << " " << height << std::endl;

    // Second step: create a cuda texture out of this image. It'll be used to generate training data efficiently on the fly
	// cudaTextureObject_t texture = create_pitch2d_rgba32f_texture(image.data(), width, height);
    cudaTextureObject_t texture = create_mipmap_rgba32f_texture(image.data(), width, height);

    return std::make_tuple(std::move(image), texture);
}

template <uint32_t stride>
__global__ void eval_image(uint32_t n_elements, cudaTextureObject_t texture, float *__restrict__ xs_and_ys, float *__restrict__ result)
{
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	uint32_t output_idx = i * stride;
	uint32_t input_idx  = i * 2;

	float4 val = tex2D<float4>(texture, xs_and_ys[input_idx], xs_and_ys[input_idx + 1]);
    
	result[output_idx + 0] = val.x;
	result[output_idx + 1] = val.y;
	result[output_idx + 2] = val.z;

	for (uint32_t i = 3; i < stride; ++i) result[output_idx + i] = 1;
}

__global__ void eval_image_lod(uint32_t n_elements, cudaTextureObject_t texture, cudaSurfaceObject_t output, int width, int height, float lod)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    float4 color = tex2DLod<float4>(texture, x/(float)width, y/(float)height, lod);
    surf2Dwrite(color, output, x * sizeof(float4), y);
}

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

const uint32_t batch_size = 1 << 12;
const uint32_t n_input_dims  = 2; // 2-D image coordinate
const uint32_t n_output_dims = 4; // RGBA color

struct NeuralImageCache::Impl
{
    typedef std::shared_ptr<OpenGLTexture> openglTextureObject_t;
    typedef GPUMatrix<float, MatrixLayout::ColumnMajor> GPUColumnMatrix;
    
    int width;
    int height;

    GPUMemory<float> xs_and_ys;

    GPUMemory<float> groundtruth_data;
    cudaTextureObject_t groundtruth;

    openglTextureObject_t reference_opengl_texture;
    openglTextureObject_t inference_opengl_texture;

    std::unique_ptr<GPUColumnMatrix> inference_input; // Auxiliary matrices for evaluation
    std::unique_ptr<GPUColumnMatrix> inference_result;
    std::unique_ptr<GPUColumnMatrix> training_input;  // Auxiliary matrices for training
    std::unique_ptr<GPUColumnMatrix> training_target;
    cudaStream_t inference_stream;
    cudaStream_t training_stream;

    std::shared_ptr<Loss<precision_t>> loss;
    std::shared_ptr<Optimizer<precision_t>> optimizer;
    std::shared_ptr<NetworkWithInputEncoding<precision_t>> network;
    std::shared_ptr<Trainer<float, precision_t, precision_t>> trainer;
    
    curandGenerator_t rng;

    float    tmp_loss = 0;
    uint32_t tmp_loss_counter = 0;
    uint64_t total_steps = 0;

    Impl(std::string filename) 
    {
        // Load the image & network configuration
        json config = {
            {
                "loss", {{"otype", "RelativeL2"}}
            }, {
                "optimizer", {
                    {"otype", "Adam"},
                    {"learning_rate", 1e-2},
                    {"beta1", 0.9f},
                    {"beta2", 0.99f},
                    {"epsilon", 1e-8f},
                    {"l2_reg", 1e-8f},
                }
            }, {
                "encoding", {
                    {"otype", "OneBlob"},
                    {"n_bins", 64},
                }
            }, {
                "network", {
                    {"otype", "FullyFusedMLP"},
                    {"n_neurons", 128},
                    {"n_layers", 5},
                    {"activation", "ReLU"},
                    {"output_activation", "None"},
                }
            }
        };
        std::tie(groundtruth_data, groundtruth) = generate_image_texture(filename, width, height);

        uint32_t n_coords = width * height;
        uint32_t n_coords_padded = (n_coords + 255) / 256 * 256;

        // Create the OpenGL texture
        initialize();

        // Generate the coordinates of the image
        std::vector<float> host_xs_and_ys(n_coords * n_input_dims);
        xs_and_ys = GPUMemory<float>(n_coords_padded * n_input_dims);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int idx = (y * width + x) * n_input_dims;
                host_xs_and_ys[idx + 0] = (float)(x + 0.5) / (float)width;
                host_xs_and_ys[idx + 1] = (float)(y + 0.5) / (float)height;
            }
        }
        xs_and_ys.copy_from_host(host_xs_and_ys.data());

        // Allocate matrices for training and evaluation
        inference_input  = std::make_unique<GPUColumnMatrix>(xs_and_ys.data(), n_input_dims, n_coords_padded);
        inference_result = std::make_unique<GPUColumnMatrix>(n_output_dims, n_coords_padded);
        training_input    = std::make_unique<GPUColumnMatrix>(n_input_dims, batch_size);
        training_target   = std::make_unique<GPUColumnMatrix>(n_output_dims, batch_size);

        // Input & corresponding RNG
        CURAND_CHECK_THROW(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK_THROW(curandSetPseudoRandomGeneratorSeed(rng, 1337ULL));

        inference_stream = 0;
        CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
        training_stream = inference_stream;
        CURAND_CHECK_THROW(curandSetStream(rng, training_stream));

        // Create the network
        json encoding_opts  = config.value("encoding", json::object());
        json loss_opts      = config.value("loss", json::object());
        json optimizer_opts = config.value("optimizer", json::object());
        json network_opts   = config.value("network", json::object());

        loss = std::shared_ptr<Loss<precision_t>>{create_loss<precision_t>(loss_opts)};
        optimizer = std::shared_ptr<Optimizer<precision_t>>{create_optimizer<precision_t>(optimizer_opts)};
        network = std::make_shared<NetworkWithInputEncoding<precision_t>>(n_input_dims, 0, n_output_dims, encoding_opts, network_opts);
        trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(network, optimizer, loss);

        // Initialize values
        {
            cudaArray_t array;
            reference_opengl_texture->mapCudaArray(array);
            // {
            //     CUDA_CHECK_THROW(cudaMemcpyToArray(array, 0, 0, groundtruth_data.data(), sizeof(float) * n_coords * 4, cudaMemcpyDeviceToDevice));
            // }
            {
                cudaResourceDesc surfRes;
                memset(&surfRes,0,sizeof(cudaResourceDesc));
                surfRes.resType = cudaResourceTypeArray;
                surfRes.res.array.array = array;

                cudaSurfaceObject_t surf;
                CUDA_CHECK_THROW(cudaCreateSurfaceObject(&surf, &surfRes));

                dim3 blockSize(16,16,1);
                dim3 gridSize(((uint32_t)width+blockSize.x-1)/blockSize.x, ((uint32_t)height+blockSize.y-1)/blockSize.y,1);
                eval_image_lod<<<gridSize, blockSize>>>(n_coords, groundtruth, surf, width, height, 4.f);
            }
            reference_opengl_texture->unmapCudaArray(array);
        }

        linear_kernel(eval_image<4>, 0, inference_stream, n_coords, groundtruth, inference_input->data(), inference_result->data());
        synchronize(inference_result->data());
    }

    ~Impl()
    {
        reference_opengl_texture.reset();
        inference_opengl_texture.reset();
    }

    void train(size_t steps)
    {
        /* now randomly sample some data */
        for (int i = 0; i < steps; ++i)
        {
            // Third step: sample a reference image to dump to disk. Visual comparison of this reference image and the learned
            //             function will be eventually possible.
            {
                CURAND_CHECK_THROW(curandGenerateUniform(rng, training_input->data(), batch_size * n_input_dims));
                linear_kernel(eval_image<n_output_dims>, 0, training_stream, batch_size, groundtruth, training_input->data(), training_target->data());
            }

            float loss_value;
            {
                trainer->training_step(training_stream, *training_input, *training_target, &loss_value);
            }
            tmp_loss += loss_value;
            ++tmp_loss_counter;
        }
        total_steps += steps;
    }
    
    void renderInference()
    {
        network->inference(inference_stream, *inference_input, *inference_result);
        synchronize(inference_result->data());
    }
    
    float currentLoss()
    {
        float ret = tmp_loss / (float)tmp_loss_counter;
        std::cout << "step=" << total_steps << "\tloss=" << ret << std::endl;
        tmp_loss = 0;
        tmp_loss_counter = 0;
        return ret;
    }

    void initialize() 
    {
        reference_opengl_texture = std::make_shared<OpenGLTexture>(width, height);
        inference_opengl_texture = std::make_shared<OpenGLTexture>(width, height);
    }
    
    void synchronize(void* device_ptr)
    {
        // We want to copy cuda_dest_resource data to the texture
        // map buffer objects to get CUDA device pointers
        cudaArray_t texture_ptr;
        inference_opengl_texture->mapCudaArray(texture_ptr);
    
        static_assert(n_output_dims == 4);
        int num_of_bytes = sizeof(float) * width * height * 4;
        CUDA_CHECK_THROW(cudaMemcpyToArray(texture_ptr, 0, 0, device_ptr, num_of_bytes, cudaMemcpyDeviceToDevice));

        inference_opengl_texture->unmapCudaArray(texture_ptr);
    }
};

NeuralImageCache::~NeuralImageCache()
{
    pimpl.reset();
}

NeuralImageCache::NeuralImageCache(std::string filename)
    : pimpl(new Impl(filename))
{
}

void NeuralImageCache::bindInferenceTexture()
{
    pimpl->inference_opengl_texture->bindOpenGLTexture();
}

void NeuralImageCache::bindReferenceTexture()
{
    pimpl->reference_opengl_texture->bindOpenGLTexture();
}

void NeuralImageCache::train(size_t steps)
{
    pimpl->train(steps);
}

void NeuralImageCache::renderInference()
{
    pimpl->renderInference();
}

float NeuralImageCache::currentLoss()
{
    return pimpl->currentLoss();
}
