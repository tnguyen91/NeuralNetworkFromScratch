#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "Layer.h"
#include <vector>
#include <functional>
#include <string>

class DenseLayer : public Layer {
public:
    DenseLayer(int inputSize, int outputSize, unsigned int seed = 0);
    DenseLayer(int inputSize, int outputSize, std::function<double(double)> activation,
               std::function<double(double)> activationDerivative, unsigned int seed = 0);
    DenseLayer(int inputSize, int outputSize, std::function<double(double)> activation,
               std::function<double(double)> activationDerivative, const std::string& activationName,
               unsigned int seed = 0);
    DenseLayer(int inputSize, int outputSize, bool useSoftmax, unsigned int seed = 0);

    std::vector<double> forward(const std::vector<double>& input, bool training) override;
    std::vector<double> backward(const std::vector<double>& grad_output) override;

    int getInputSize() const;
    int getOutputSize() const;
    std::vector<std::vector<double>>& getWeights();
    std::vector<double>& getBiases();
    const std::vector<std::vector<double>>& getWeightGradients() const;
    const std::vector<double>& getBiasGradients() const;

private:
    int inputSize;
    int outputSize;
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> weightsGradients;
    std::vector<double> biases;
    std::vector<double> biasGradients;
    std::vector<double> inputs;
    std::vector<double> outputs;
    std::function<double(double)> activation;
    std::function<double(double)> activationDerivative;
    bool isSoftmax = false;
    void initializeWeights(unsigned int seed, const std::string& activationName);
};

#endif
