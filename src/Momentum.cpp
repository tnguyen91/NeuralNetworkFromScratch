#include "../include/Momentum.h"

Momentum::Momentum(double momentum)
    : momentum(momentum) {}

void Momentum::updateWeights(std::vector<std::vector<double>>& weights,
                             const std::vector<std::vector<double>>& weightGradients,
                             double learningRate) {
    if (weightVelocities.empty()) {
        weightVelocities.resize(weights.size(), std::vector<double>(weights[0].size(), 0.0));
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            weightVelocities[i][j] = momentum * weightVelocities[i][j] - learningRate * weightGradients[i][j];
            weights[i][j] += weightVelocities[i][j];
        }
    }
}

void Momentum::updateBiases(std::vector<double>& biases,
                             const std::vector<double>& biasGradients,
                             double learningRate) {
    if (biasVelocities.empty()) {
        biasVelocities.resize(biases.size(), 0.0);
    }

    for (size_t i = 0; i < biases.size(); ++i) {
        biasVelocities[i] = momentum * biasVelocities[i] - learningRate * biasGradients[i];
        biases[i] += biasVelocities[i];
    }
}
