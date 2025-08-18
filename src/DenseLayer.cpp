#include "DenseLayer.h"
#include "ActivationFunctions.h"
#include <random>
#include <cmath>
#include <cassert>

DenseLayer::DenseLayer(int inputSize, int outputSize, unsigned int seed)
    : inputSize(inputSize), outputSize(outputSize), isSoftmax(false), activationName("relu") {
    initializeWeights(seed, "relu");
    activation = [](double x) { return ActivationFunctions::relu(x); };
    activationDerivative = [](double x) { return ActivationFunctions::reluDerivative(x); };
}

DenseLayer::DenseLayer(int inputSize, int outputSize, std::function<double(double)> activation,
                       std::function<double(double)> activationDerivative, unsigned int seed)
    : inputSize(inputSize), outputSize(outputSize), activation(activation), activationDerivative(activationDerivative), isSoftmax(false), activationName("custom") {
    initializeWeights(seed, "");
}

DenseLayer::DenseLayer(int inputSize, int outputSize, std::function<double(double)> activation,
                       std::function<double(double)> activationDerivative, const std::string& activationName,
                       unsigned int seed)
    : inputSize(inputSize), outputSize(outputSize), activation(activation), activationDerivative(activationDerivative), isSoftmax(false), activationName(activationName) {
    initializeWeights(seed, activationName);
}

DenseLayer::DenseLayer(int inputSize, int outputSize, bool useSoftmax, unsigned int seed)
    : inputSize(inputSize), outputSize(outputSize), isSoftmax(useSoftmax), activationName(useSoftmax ? "softmax" : "relu") {
    initializeWeights(seed, useSoftmax ? "softmax" : "");
}

std::vector<double> DenseLayer::forward(const std::vector<double>& input, bool training) {
    this->inputs = input;
    outputs.resize(outputSize);
    std::vector<double> logits(outputSize, 0.0);
    for (int i = 0; i < outputSize; ++i) {
        double sum = biases[i];
        for (int j = 0; j < inputSize; ++j) {
            sum += weights[j][i] * input[j];
        }
        logits[i] = sum;
    }
    if (isSoftmax) {
        outputs = ActivationFunctions::softmax(logits);
    } else {
        for (int i = 0; i < outputSize; ++i) {
            outputs[i] = activation(logits[i]);
        }
    }
    return outputs;
}

std::vector<double> DenseLayer::backward(const std::vector<double>& gradients) {
    std::vector<double> inputGradients(inputSize, 0.0);
    if (isSoftmax) {
        const std::vector<double>& delta = gradients;
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                inputGradients[j] += weights[j][i] * delta[i];
                weightsGradients[j][i] = delta[i] * inputs[j];
            }
            biasGradients[i] = delta[i];
        }
    } else {
        for (int i = 0; i < outputSize; ++i) {
            double activationGrad = activationDerivative(outputs[i]);
            double delta = gradients[i] * activationGrad;

            for (int j = 0; j < inputSize; ++j) {
                inputGradients[j] += weights[j][i] * delta;
            }

            for (int j = 0; j < inputSize; ++j) {
                weightsGradients[j][i] = delta * inputs[j];
            }

            biasGradients[i] = delta;
        }
    }
    return inputGradients;
}

int DenseLayer::getInputSize() const { return inputSize; }
int DenseLayer::getOutputSize() const { return outputSize; }
std::vector<std::vector<double>>& DenseLayer::getWeights() { return weights; }
std::vector<double>& DenseLayer::getBiases() { return biases; }
const std::vector<std::vector<double>>& DenseLayer::getWeightGradients() const { return weightsGradients; }
const std::vector<double>& DenseLayer::getBiasGradients() const { return biasGradients; }
const std::string& DenseLayer::getActivationName() const { return activationName; }

void DenseLayer::initializeWeights(unsigned int seed, const std::string& activationName) {
    std::mt19937 gen;
    if (seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        gen.seed(seed);
    }
    double limit;
    if (isSoftmax || activationName == "softmax") {
        limit = std::sqrt(6.0 / static_cast<double>(inputSize + outputSize));
    } else if (activationName == "relu") {
        limit = std::sqrt(6.0 / static_cast<double>(inputSize));
    } else if (activationName == "sigmoid" || activationName == "linear") {
        limit = std::sqrt(6.0 / static_cast<double>(inputSize + outputSize));
    } else {
        limit = std::sqrt(6.0 / static_cast<double>(inputSize + outputSize));
    }
    std::uniform_real_distribution<> dis(-limit, limit);

    weights.resize(inputSize, std::vector<double>(outputSize));
    weightsGradients.resize(inputSize, std::vector<double>(outputSize, 0.0));
    biases.resize(outputSize);
    biasGradients.resize(outputSize, 0.0);

    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            weights[i][j] = dis(gen);
        }
    }
    for (int j = 0; j < outputSize; ++j) {
        biases[j] = 0.0;
    }
}