#include "../include/Adam.h"
#include <cmath>

Adam::Adam(double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
    : beta1(beta1), beta2(beta2), epsilon(epsilon) {}

void Adam::updateWeights(std::vector<std::vector<double>>& weights,
                         const std::vector<std::vector<double>>& weightGradients,
                         double learningRate) {
    if (mWeights.empty()) {
        mWeights.resize(weights.size(), std::vector<double>(weights[0].size(), 0.0));
        vWeights.resize(weights.size(), std::vector<double>(weights[0].size(), 0.0));
    }

    timeStep++;

    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            mWeights[i][j] = beta1 * mWeights[i][j] + (1 - beta1) * weightGradients[i][j];
            vWeights[i][j] = beta2 * vWeights[i][j] + (1 - beta2) * weightGradients[i][j] * weightGradients[i][j];

            double mHat = mWeights[i][j] / (1 - std::pow(beta1, timeStep));
            double vHat = vWeights[i][j] / (1 - std::pow(beta2, timeStep));

            weights[i][j] -= learningRate * mHat / (std::sqrt(vHat) + epsilon);
        }
    }
}

void Adam::updateBiases(std::vector<double>& biases,
                        const std::vector<double>& biasGradients,
                        double learningRate) {
    if (mBiases.empty()) {
        mBiases.resize(biases.size(), 0.0);
        vBiases.resize(biases.size(), 0.0);
    }

    timeStep++;

    for (size_t i = 0; i < biases.size(); ++i) {
        mBiases[i] = beta1 * mBiases[i] + (1 - beta1) * biasGradients[i];
        vBiases[i] = beta2 * vBiases[i] + (1 - beta2) * biasGradients[i] * biasGradients[i];

        double mHat = mBiases[i] / (1 - std::pow(beta1, timeStep));
        double vHat = vBiases[i] / (1 - std::pow(beta2, timeStep));

        biases[i] -= learningRate * mHat / (std::sqrt(vHat) + epsilon);
    }
}