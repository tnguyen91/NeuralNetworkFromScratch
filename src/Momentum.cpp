#include "Momentum.h"

Momentum::Momentum(double momentum, double l2_lambda)
    : Optimizer(l2_lambda), momentum(momentum) {}

void Momentum::updateWeights(std::vector<std::vector<double>>& weights,
                             const std::vector<std::vector<double>>& weightGradients,
                             double learningRate) {
    if (weights.empty()) return;
    
    if (weightVelocities.size() != weights.size()) {
        weightVelocities.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            weightVelocities[i].resize(weights[i].size(), 0.0);
        }
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            double grad = weightGradients[i][j] + l2_lambda_ * weights[i][j];
            weightVelocities[i][j] = momentum * weightVelocities[i][j] - learningRate * grad;
            weights[i][j] += weightVelocities[i][j];
        }
    }
}

void Momentum::updateBiases(std::vector<double>& biases,
                             const std::vector<double>& biasGradients,
                             double learningRate) {
    if (biases.empty()) return;
    
    if (biasVelocities.size() != biases.size()) {
        biasVelocities.resize(biases.size(), 0.0);
    }

    for (size_t i = 0; i < biases.size(); ++i) {
        biasVelocities[i] = momentum * biasVelocities[i] - learningRate * biasGradients[i];
        biases[i] += biasVelocities[i];
    }
}
