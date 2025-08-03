#include "../include/Optimizer.h"

class SGD : public Optimizer {
public:
    void updateWeights(std::vector<std::vector<double>>& weights,
                       const std::vector<std::vector<double>>& weightGradients,
                       double learningRate) override {
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weights[i][j] -= learningRate * weightGradients[i][j];
            }
        }
    }

    void updateBiases(std::vector<double>& biases,
                      const std::vector<double>& biasGradients,
                      double learningRate) override {
        for (size_t i = 0; i < biases.size(); ++i) {
            biases[i] -= learningRate * biasGradients[i];
        }
    }
};