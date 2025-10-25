#include "Adam.h"
#include <cmath>

Adam::Adam(double beta1, double beta2, double epsilon, double l2_lambda)
    : beta1(beta1), beta2(beta2), epsilon(epsilon), Optimizer(l2_lambda), timeStep(0) {}

void Adam::updateWeights(std::vector<std::vector<double>>& weights,
                         const std::vector<std::vector<double>>& weightGradients,
                         double learningRate) {
    if (weights.empty() || weightGradients.empty()) return;
    
    if (mWeights.size() != weights.size()) {
        mWeights.resize(weights.size());
        vWeights.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            mWeights[i].resize(weights[i].size(), 0.0);
            vWeights[i].resize(weights[i].size(), 0.0);
        }
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            double grad = weightGradients[i][j] + l2_lambda_ * weights[i][j];
            mWeights[i][j] = beta1 * mWeights[i][j] + (1.0 - beta1) * grad;
            vWeights[i][j] = beta2 * vWeights[i][j] + (1.0 - beta2) * grad * grad;
            double mHat = mWeights[i][j] / (1.0 - std::pow(beta1, timeStep + 1));
            double vHat = vWeights[i][j] / (1.0 - std::pow(beta2, timeStep + 1));
            weights[i][j] -= learningRate * mHat / (std::sqrt(vHat) + epsilon);
        }
    }
}

void Adam::updateBiases(std::vector<double>& biases,
                        const std::vector<double>& biasGradients,
                        double learningRate) {
    if (biases.empty() || biasGradients.empty()) return;
    
    if (mBiases.size() != biases.size()) {
        mBiases.resize(biases.size(), 0.0);
        vBiases.resize(biases.size(), 0.0);
    }

    for (size_t i = 0; i < biases.size(); ++i) {
        mBiases[i] = beta1 * mBiases[i] + (1.0 - beta1) * biasGradients[i];
        
        vBiases[i] = beta2 * vBiases[i] + (1.0 - beta2) * biasGradients[i] * biasGradients[i];

        double mHat = mBiases[i] / (1.0 - std::pow(beta1, timeStep + 1));
        
        double vHat = vBiases[i] / (1.0 - std::pow(beta2, timeStep + 1));

        biases[i] -= learningRate * mHat / (std::sqrt(vHat) + epsilon);
    }
    
    timeStep++;
}