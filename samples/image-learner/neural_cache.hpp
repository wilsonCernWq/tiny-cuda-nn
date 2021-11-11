#ifndef NEURAL_CACHE_HPP
#define NEURAL_CACHE_HPP

#include <tuple>
#include <memory>

class NeuralImageCache
{
    struct Impl;
public:
    ~NeuralImageCache();

    NeuralImageCache(std::string filename);

    void bindInferenceTexture();    
    void bindReferenceTexture();

    // trigger an actual data access, domain: [0, 1]
    // void focus(float x, float y);

    // trigger a training step
    void train(size_t steps);

    // trigger an evaluation step
    void render();

    // get current loss
    float currentLoss();

private:
    std::unique_ptr<Impl> pimpl;  // pointer to the internal implementation
};

#endif // NEURAL_CACHE_HPP
