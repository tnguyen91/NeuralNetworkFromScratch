#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <functional>

class Layer {
public:
    Layer(int inputSize, int outputSize);
    Layer(int inputSize, int outputSize, std::function<double(double)> activation,
          std::function<double(double)> activationDerivative);

    std::vector<double> forward(const std::vector<double>& inputs);

    std::vector<double> backward(const std::vector<double>& gradients, double learningRate);

    int getInputSize() const;
    int getOutputSize() const;
    const std::vector<std::vector<double>>& getWeights() const;
    void updateWeights(int inputIndex, int outputIndex, double value);
    const std::vector<double>& getBiases() const;
    void updateBiases(int outputIndex, double value);
    const std::vector<std::vector<double>>& getWeightGradients() const;
    const std::vector<double>& getBiasGradients() const;

private:
    int inputSize;
    int outputSize;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    std::vector<double> inputs;
    std::vector<double> outputs;
    std::vector<std::vector<double>> weightGradients;
    std::vector<double> biasGradients;

    std::function<double(double)> activation;
    std::function<double(double)> activationDerivative;
};

#endif
