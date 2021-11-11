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

    // trigger an evaluation step
    void renderInference();
    void renderReference();

    // trigger a training step
    void train(size_t steps);

    // trigger an actual data access, domain: [0, 1]
    void access(float x, float y);

    // get current loss
    float currentLoss();

private:
    std::unique_ptr<Impl> pimpl;  // pointer to the internal implementation
};

#endif // NEURAL_CACHE_HPP
