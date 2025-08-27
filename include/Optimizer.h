#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>

class Optimizer {
public:
    explicit Optimizer(double l2_lambda = 0.0) : l2_lambda_(l2_lambda) {}
    virtual ~Optimizer() = default;

    void setL2Lambda(double l2_lambda) { l2_lambda_ = l2_lambda; }
    double getL2Lambda() const { return l2_lambda_; }

    virtual void updateWeights(std::vector<std::vector<double>>& weights,
                               const std::vector<std::vector<double>>& weightGradients,
                               double learningRate) = 0;

    virtual void updateBiases(std::vector<double>& biases,
                              const std::vector<double>& biasGradients,
                              double learningRate) = 0;
protected:
    double l2_lambda_ = 0.0;
};

#endif