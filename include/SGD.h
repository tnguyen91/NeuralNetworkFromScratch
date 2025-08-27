#ifndef SGD_H
#define SGD_H

#include "Optimizer.h"
#include <vector>


class SGD : public Optimizer {
public:
    explicit SGD(double l2_lambda = 0.0) : Optimizer(l2_lambda) {}

    void updateWeights(std::vector<std::vector<double>>& weights,
                       const std::vector<std::vector<double>>& weightGradients,
                       double learningRate) override;

    void updateBiases(std::vector<double>& biases,
                      const std::vector<double>& biasGradients,
                      double learningRate) override;
};

#endif