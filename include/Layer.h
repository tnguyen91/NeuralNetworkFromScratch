#ifndef LAYER_H
#define LAYER_H

#include <vector>

class Layer {
public:
    virtual std::vector<double> forward(const std::vector<double>& input, bool training) = 0;

    virtual std::vector<double> backward(const std::vector<double>& grad_output) = 0;

    virtual ~Layer() = default;
};

#endif
