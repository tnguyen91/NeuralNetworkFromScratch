#ifndef MOMENTUM_H
#define MOMENTUM_H

#include "Optimizer.h"

class Momentum : public Optimizer {
public:
    Momentum(double momentum);

    void updateWeights(std::vector<std::vector<double>>& weights,
                       const std::vector<std::vector<double>>& weightGradients,
                       double learningRate) override;

    void updateBiases(std::vector<double>& biases,
                      const std::vector<double>& biasGradients,
                      double learningRate) override;

private:
    double momentum;
    std::vector<std::vector<double>> weightVelocities;
    std::vector<double> biasVelocities;
};

#endif