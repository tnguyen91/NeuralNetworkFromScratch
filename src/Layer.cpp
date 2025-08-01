#include "../include/Layer.h"
#include "../include/ActivationFunctions.h"
#include <random>
#include <numeric>

Layer::Layer(int inputSize, int outputSize)
    : inputSize(inputSize), outputSize(outputSize) {
    
    activation = [](double x) { return ActivationFunctions::sigmoid(x); };
    activationDerivative = [](double x) { return ActivationFunctions::sigmoidDerivative(x); };
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    weights.resize(outputSize, std::vector<double>(inputSize));
    biases.resize(outputSize);

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weights[i][j] = dis(gen);
        }
        biases[i] = dis(gen);
    }
}

Layer::Layer(int inputSize, int outputSize, std::function<double(double)> activation,
             std::function<double(double)> activationDerivative)
    : inputSize(inputSize), outputSize(outputSize), activation(activation), activationDerivative(activationDerivative) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    weights.resize(outputSize, std::vector<double>(inputSize));
    biases.resize(outputSize);

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weights[i][j] = dis(gen);
        }
        biases[i] = dis(gen);
    }
}

std::vector<double> Layer::forward(const std::vector<double>& inputs) {
    this->inputs = inputs;
    outputs.resize(outputSize);
    for (int i = 0; i < outputSize; ++i) {
        outputs[i] = std::inner_product(inputs.begin(), inputs.end(), weights[i].begin(), biases[i]);
        outputs[i] = activation(outputs[i]);
    }
    return outputs;
}

std::vector<double> Layer::backward(const std::vector<double>& gradients, double learningRate) {
    std::vector<double> inputGradients(inputSize, 0.0);
    for (int i = 0; i < outputSize; ++i) {
        double delta = gradients[i] * activationDerivative(outputs[i]);
        for (int j = 0; j < inputSize; ++j) {
            inputGradients[j] += delta * weights[i][j];
            weights[i][j] -= learningRate * delta * inputs[j];
        }
        biases[i] -= learningRate * delta;
    }
    return inputGradients;
}

int Layer::getInputSize() const {
    return inputSize;
}

int Layer::getOutputSize() const {
    return outputSize;
}

const std::vector<std::vector<double>>& Layer::getWeights() const {
    return weights;
}

const std::vector<double>& Layer::getBiases() const {
    return biases;
}

const std::vector<std::vector<double>>& Layer::getWeightGradients() const {
    return weightGradients;
}

const std::vector<double>& Layer::getBiasGradients() const {
    return biasGradients;
}

void Layer::updateWeights(int inputIndex, int outputIndex, double value) {
    weights[outputIndex][inputIndex] += value;
}

void Layer::updateBiases(int outputIndex, double value) {
    biases[outputIndex] += value;
}
