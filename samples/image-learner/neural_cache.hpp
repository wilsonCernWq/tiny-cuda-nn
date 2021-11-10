#pragma once

#include <tuple>

class NeuralImageCache
{
public:
    using color_t = std::tuple<float, float, float>;

    ~NeuralImageCache() = default;

    NeuralImageCache(std::string filename);

    // bind to the current OpenGL texture
    void bindTexture();    

    // domain to access = [0, 1]
    // color_t access(float x, float y);

    // trigger a training step
    // void train();

private:
    int width;
    int height;

    void initialize();
    void resize(int width, int height);
};
