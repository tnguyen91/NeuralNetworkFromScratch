#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void updateWeights(std::vector<std::vector<double>>& weights,
                               const std::vector<std::vector<double>>& weightGradients,
                               double learningRate) = 0;

    virtual void updateBiases(std::vector<double>& biases,
                              const std::vector<double>& biasGradients,
                              double learningRate) = 0;
};

#endif