#ifndef NEURAL_CACHE_HPP
#define NEURAL_CACHE_HPP

// #include <tuple>

class NeuralImageCache
{
public:
    ~NeuralImageCache();

    NeuralImageCache(std::string filename);

    // bind to the current OpenGL texture
    void bindTexture();    

    // trigger an actual data access, domain: [0, 1]
    // void focus(float x, float y);

    // trigger a training step
    void train(size_t steps);

    // trigger an evaluation step
    void render();

    // get current loss
    float pull_loss();

private:
    int width;
    int height;
    void initialize();
    void synchronize(void* device_ptr);
};

#endif // NEURAL_CACHE_HPP
