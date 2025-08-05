#ifndef ADAM_H
#define ADAM_H

#include "Optimizer.h"

class Adam : public Optimizer {
public:
    Adam(double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

    void updateWeights(std::vector<std::vector<double>>& weights,
                       const std::vector<std::vector<double>>& weightGradients,
                       double learningRate) override;

    void updateBiases(std::vector<double>& biases,
                      const std::vector<double>& biasGradients,
                      double learningRate) override;

private:
    double beta1, beta2, epsilon;
    int timeStep;
    std::vector<std::vector<double>> mWeights, vWeights;
    std::vector<double> mBiases, vBiases;
};

#endif