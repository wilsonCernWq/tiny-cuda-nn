#include "neuralcache.hpp"

#include "tinyexr_wrapper.h"
#include "helper_cuda_texture.h"
#include "texture_interop.h"

#include <tiny-cuda-nn/misc_kernels.h>
#include <tiny-cuda-nn/config.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif

using namespace tcnn;
using precision_t = network_precision_t;

//  Work with OpenEXR images

static GPUMemory<float> load_image(const std::string &filename, int &width, int &height)
{
	float *out; // width * height * RGBA
	load_exr(&out, &width, &height, filename.c_str());

	GPUMemory<float> result(width * height * 4);
	result.copy_from_host(out);
	free(out); // release memory of image data

	return result;
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

__device__ inline void quantize_sampling_inputs_inner(uint32_t i, uint32_t width, uint32_t height, int lod, float *__restrict__ inputs)
{
	const uint32_t r = 1 << lod;
	const uint32_t w = width / r;
	const uint32_t h = height / r;

	const uint32_t idx = i * 2;

	const uint32_t x = (1.f - inputs[idx + 0]) * (float)w;
	const uint32_t y = (1.f - inputs[idx + 1]) * (float)h;
	inputs[idx + 0] = (x + 0.5) / (float)w;
	inputs[idx + 1] = (y + 0.5) / (float)h;
}

__global__ void quantize_sampling_inputs_fixed_lod(uint32_t n_elements, uint32_t width, uint32_t height, int lod, float *__restrict__ inputs)
{
    if (lod == 0) return;

	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

    quantize_sampling_inputs_inner(i, width, height, lod, inputs);
}

__global__ void quantize_sampling_inputs_variable_lod(uint32_t n_elements, uint32_t width, uint32_t height, int max_lod, float *__restrict__ lods, float *__restrict__ inputs)
{
    if (max_lod == 0) return;

	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	int lod = (1.f - lods[i]) * max_lod;
    quantize_sampling_inputs_inner(i, width, height, lod, inputs);
}

template <uint32_t stride>
__global__ void sample_groundtruth(uint32_t n_elements, cudaTextureObject_t groundtruth, float *__restrict__ xs_and_ys, float *__restrict__ result)
{
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	uint32_t output_idx = i * stride;
	uint32_t input_idx  = i * 2;

	float4 sample = tex2D<float4>(groundtruth, xs_and_ys[input_idx], xs_and_ys[input_idx + 1]);

	result[output_idx + 0] = sample.x;
	result[output_idx + 1] = sample.y;
	result[output_idx + 2] = sample.z;
    result[output_idx + 3] = sample.w;

	for (uint32_t i = 4; i < stride; ++i) result[output_idx + i] = 1;
}

__global__ void resample_texture_with_lod(uint32_t width, uint32_t height, cudaTextureObject_t texture, cudaSurfaceObject_t output, int lod)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    float4 color = tex2DLod<float4>(texture, x/(float)width, y/(float)height, (float)lod);
    surf2Dwrite(color, output, x * sizeof(float4), y);
}

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

    int level_of_detail = 0;

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
        training_input   = std::make_unique<GPUColumnMatrix>(n_input_dims, batch_size);
        training_target  = std::make_unique<GPUColumnMatrix>(n_output_dims, batch_size);

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
        renderInference();
        renderReference();
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
                linear_kernel(quantize_sampling_inputs_fixed_lod, 0, training_stream, batch_size, width, height, level_of_detail, training_input->data());
                linear_kernel(sample_groundtruth<n_output_dims>, 0, training_stream, batch_size, groundtruth, training_input->data(), training_target->data());
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

        // linear_kernel(sample_groundtruth<n_output_dims>, 0, inference_stream, width * height, groundtruth, inference_input->data(), inference_result->data());

        // We want to copy cuda_dest_resource data to the texture
        // map buffer objects to get CUDA device pointers
        cudaArray_t array;
        inference_opengl_texture->mapCudaArray(array);
        {
            CUDA_CHECK_THROW(cudaMemcpyToArray(array, 0, 0, inference_result->data(), sizeof(float) * width * height * 4, cudaMemcpyDeviceToDevice));
        }
        inference_opengl_texture->unmapCudaArray(array);
    }

    void renderReference()
    {
        // We want to copy cuda_dest_resource data to the texture
        // map buffer objects to get CUDA device pointers
        cudaArray_t array;
        reference_opengl_texture->mapCudaArray(array);
        /* show ground truth at the full resolution */
        // {
        //     CUDA_CHECK_THROW(cudaMemcpyToArray(array, 0, 0, groundtruth_data.data(), sizeof(float) * width * height * 4, cudaMemcpyDeviceToDevice));
        // }
        /* show ground truth with lod */
        {
            cudaResourceDesc resDesc;
            memset(&resDesc,0,sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = array;
            cudaSurfaceObject_t arraySurf;
            CUDA_CHECK_THROW(cudaCreateSurfaceObject(&arraySurf, &resDesc));
            bilinear_kernel(resample_texture_with_lod, 0, /*stream=*/inference_stream, width, height, groundtruth, arraySurf, /*lod=*/level_of_detail);
        }
        reference_opengl_texture->unmapCudaArray(array);
    }

    float currentLoss()
    {
        float ret = tmp_loss / (float)tmp_loss_counter;
        // std::cout << "step=" << total_steps << "\tloss=" << ret << std::endl;
        tmp_loss = 0;
        tmp_loss_counter = 0;
        return ret;
    }

    void initialize() 
    {
        reference_opengl_texture = std::make_shared<OpenGLTexture>(width, height);
        inference_opengl_texture = std::make_shared<OpenGLTexture>(width, height);
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

void NeuralImageCache::setLod(int lod)
{
    pimpl->level_of_detail = lod;
}

void NeuralImageCache::train(size_t steps)
{
    pimpl->train(steps);
}

void NeuralImageCache::renderInference()
{
    pimpl->renderInference();
}

void NeuralImageCache::renderReference()
{
    pimpl->renderReference();
}

float NeuralImageCache::currentLoss()
{
    return pimpl->currentLoss();
}
