#include "neural_cache.hpp"

#include "../tinyexr_wrapper.h"

#include "util.h"

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

using namespace tcnn;
using precision_t = network_precision_t;

static GLuint                 view_opengl_texture;
static cudaGraphicsResource_t view_cuda_resource;

const uint32_t batch_size = 1 << 16;
const uint32_t n_input_dims  = 2; // 2-D image coordinate
const uint32_t n_output_dims = 4; // RGBA color

static GPUMemory<float>    reference_image;
static cudaTextureObject_t reference_texture;

static GPUMemory<float> xs_and_ys;

static GPUMatrix<float, MatrixLayout::ColumnMajor> prediction_input; // Auxiliary matrices for evaluation
static GPUMatrix<float, MatrixLayout::ColumnMajor> prediction_result;
static GPUMatrix<float, MatrixLayout::ColumnMajor> training_input;   // Auxiliary matrices for training
static GPUMatrix<float, MatrixLayout::ColumnMajor> training_target;

static cudaStream_t prediction_stream;
static cudaStream_t training_stream;

static curandGenerator_t rng;

static std::shared_ptr<Loss<precision_t>> loss;
static std::shared_ptr<Optimizer<precision_t>> optimizer;
static std::shared_ptr<NetworkWithInputEncoding<precision_t>> network;
static std::shared_ptr<Trainer<float, precision_t, precision_t>> trainer;

static float    tmp_loss = 0;
static uint32_t tmp_loss_counter = 0;

static GPUMemory<float> load_image(const std::string &filename, int &width, int &height)
{
	float *out; // width * height * RGBA
	load_exr(&out, &width, &height, filename.c_str());

	GPUMemory<float> result(width * height * 4);
	result.copy_from_host(out);
	free(out); // release memory of image data

	return result;
}

static cudaTextureObject_t generate_image_texture(std::string filename, int& width, int& height)
{
    // First step: load an image that we'd like to learn
	reference_image = load_image(filename.c_str(), width, height);
    std::cout << "Image size: " << width << " " << height << std::endl;

    // Second step: create a cuda texture out of this image. It'll be used to generate training data efficiently on the fly
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = reference_image.data();
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

template <uint32_t stride>
__global__ void eval_image(uint32_t n_elements, cudaTextureObject_t texture, float *__restrict__ xs_and_ys, float *__restrict__ result)
{
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements)
		return;

	uint32_t output_idx = i * stride;
	uint32_t input_idx = i * 2;

	float4 val = tex2D<float4>(texture, xs_and_ys[input_idx], xs_and_ys[input_idx + 1]);
	result[output_idx + 0] = val.x;
	result[output_idx + 1] = val.y;
	result[output_idx + 2] = val.z;

	for (uint32_t i = 3; i < stride; ++i)
	{
		result[output_idx + i] = 1;
	}
}

NeuralImageCache::~NeuralImageCache()
{

}

NeuralImageCache::NeuralImageCache(std::string filename)
{
    // Load the image & network configuration
    reference_texture = generate_image_texture(filename, width, height);
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

    // Create the OpenGL texture
    initialize();
    // synchronize(reference_image.data());
    // return;

    // Generate the coordinates of the image.
    uint32_t n_coords = width * height;
    uint32_t n_coords_padded = (n_coords + 255) / 256 * 256;

    xs_and_ys = GPUMemory<float>(n_coords_padded * n_input_dims);
    std::vector<float> host_xs_and_ys(n_coords * n_input_dims);
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
    new (&prediction_input)  GPUMatrix<float, MatrixLayout::ColumnMajor>(xs_and_ys.data(), n_input_dims, n_coords_padded);
    new (&prediction_result) GPUMatrix<float, MatrixLayout::ColumnMajor>(n_output_dims, n_coords_padded);
    new (&training_input)    GPUMatrix<float, MatrixLayout::ColumnMajor>(n_input_dims, batch_size);
    new (&training_target)   GPUMatrix<float, MatrixLayout::ColumnMajor>(n_output_dims, batch_size);

	// Input & corresponding RNG
    CURAND_CHECK_THROW(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK_THROW(curandSetPseudoRandomGeneratorSeed(rng, 1337ULL));

    prediction_stream = 0;
    CUDA_CHECK_THROW(cudaStreamCreate(&prediction_stream));
    training_stream = prediction_stream;
    CURAND_CHECK_THROW(curandSetStream(rng, training_stream));

    // Create the network
    json encoding_opts = config.value("encoding", json::object());
    json loss_opts = config.value("loss", json::object());
    json optimizer_opts = config.value("optimizer", json::object());
    json network_opts = config.value("network", json::object());

    loss = std::shared_ptr<Loss<precision_t>>{create_loss<precision_t>(loss_opts)};
    optimizer = std::shared_ptr<Optimizer<precision_t>>{create_optimizer<precision_t>(optimizer_opts)};
    network = std::make_shared<NetworkWithInputEncoding<precision_t>>(n_input_dims, 0, n_output_dims, encoding_opts, network_opts);
    trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(network, optimizer, loss);

    // Initialize values
    // linear_kernel(eval_image<4>, 0, nullptr, n_coords, reference_texture, prediction_input.data(), prediction_result.data());
    synchronize(prediction_result.data());
}

void NeuralImageCache::bindTexture()
{
    glBindTexture(GL_TEXTURE_2D, view_opengl_texture);
    check_error_gl("NeuralImageCache Bind OpenGL Texture");
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

void NeuralImageCache::synchronize(void* device_ptr)
{
    // We want to copy cuda_dest_resource data to the texture
    // map buffer objects to get CUDA device pointers
    CUDA_CHECK_THROW(cudaGraphicsMapResources(1, &view_cuda_resource, 0));

    cudaArray_t view_cuda_array_pointer;
    CUDA_CHECK_THROW(cudaGraphicsSubResourceGetMappedArray(&view_cuda_array_pointer, view_cuda_resource, 0, 0));

    static_assert(n_output_dims == 4);
    int num_of_bytes = sizeof(float) * width * height * 4;
    CUDA_CHECK_THROW(cudaMemcpyToArray(view_cuda_array_pointer, 0, 0, device_ptr, num_of_bytes, cudaMemcpyDeviceToDevice));

    CUDA_CHECK_THROW(cudaGraphicsUnmapResources(1, &view_cuda_resource, 0));
}

void NeuralImageCache::train(size_t steps)
{
    /* now randomly sample some data */
    for (int i = 0; i < steps; ++i)
    {
        // Third step: sample a reference image to dump to disk. Visual comparison of this reference image and the learned
        //             function will be eventually possible.
        {
            CURAND_CHECK_THROW(curandGenerateUniform(rng, training_input.data(), batch_size * n_input_dims));
            linear_kernel(eval_image<n_output_dims>, 0, training_stream, batch_size, reference_texture, training_input.data(), training_target.data());
        }
        // Training step
        float loss_value;
        {
            trainer->training_step(training_stream, training_input, training_target, &loss_value);
        }
        tmp_loss += loss_value;
        ++tmp_loss_counter;
    }
}

void NeuralImageCache::render()
{
    network->inference(prediction_stream, prediction_input, prediction_result);
    synchronize(prediction_result.data());
}

float NeuralImageCache::pull_loss()
{
    float ret = tmp_loss / (float)tmp_loss_counter;
    std::cout << "loss=" << ret << std::endl;
    tmp_loss = 0;
    tmp_loss_counter = 0;
    return ret;
}
