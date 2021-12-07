#ifndef NEURAL_CACHE_HPP
#define NEURAL_CACHE_HPP

#include <tuple>
#include <memory>

class NeuralImageCache {
public:
    enum SamplingMode {
        UNIFORM_RANDOM = 0,
        UNIFORM_RANDOM_QUANTIZED,
        TILE_BASED_SIMPLE,
        TILE_BASED_MIXTURE,
        TILE_BASED_EVENLY,
    };

    ~NeuralImageCache();

    NeuralImageCache(std::string filename);

    void bindInferenceTexture();    
    void bindReferenceTexture();

    void setLod(int level_of_detail);
    void setTileSize(int tile_size);

    // trigger an evaluation step
    void renderInference();
    void renderReference();

    // trigger a training step
    void train(size_t steps, SamplingMode mode);

    // get current training statistics
    void trainingStats(size_t steps, float& training_loss, float& groundtruth_loss);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;  // pointer to the internal implementation
};

#endif // NEURAL_CACHE_HPP
