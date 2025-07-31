#ifndef LAYER_H
#define LAYER_H

#include <vector>

class Layer {
public:
    Layer(int inputSize, int outputSize);

    std::vector<double> forward(const std::vector<double>& inputs);

    std::vector<double> backward(const std::vector<double>& gradients);

    int getInputSize() const { return inputSize; }
    int getOutputSize() const { return outputSize; }
    const std::vector<std::vector<double>>& getWeights() const { return weights; }
    const std::vector<double>& getBiases() const { return biases; }
    const std::vector<std::vector<double>>& getWeightGradients() const { return weightGradients; }
    const std::vector<double>& getBiasGradients() const { return biasGradients; }

private:
    int inputSize;
    int outputSize;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    std::vector<double> inputs;
    std::vector<double> outputs;
    std::vector<std::vector<double>> weightGradients;
    std::vector<double> biasGradients;
};

#endif
