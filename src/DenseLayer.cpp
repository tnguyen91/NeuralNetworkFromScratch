#include "DenseLayer.h"
#include "ActivationFunctions.h"
#include <random>
#include <cmath>
#include <cassert>
#include <stdexcept>

DenseLayer::DenseLayer(int inputSize, int outputSize, unsigned int seed)
    : inputSize(inputSize), outputSize(outputSize), isSoftmax(false), activationName("relu") {
    initializeWeights(seed, "relu");
    activation = [](double x) { return ActivationFunctions::relu(x); };
    activationDerivative = [](double x) { return ActivationFunctions::reluDerivative(x); };
}

DenseLayer::DenseLayer(int inputSize, int outputSize, const std::string& activationName, unsigned int seed)
    : inputSize(inputSize), outputSize(outputSize), isSoftmax(false), activationName(activationName) {
    initializeWeights(seed, activationName);
    
    if (activationName == "relu") {
        activation = [](double x) { return ActivationFunctions::relu(x); };
        activationDerivative = [](double x) { return ActivationFunctions::reluDerivative(x); };
    } else if (activationName == "sigmoid") {
        activation = [](double x) { return ActivationFunctions::sigmoid(x); };
        activationDerivative = [](double x) { 
            double sigmoid_out = ActivationFunctions::sigmoid(x);
            return ActivationFunctions::sigmoidDerivative(sigmoid_out);
        };
    } else if (activationName == "linear") {
        activation = [](double x) { return x; };
        activationDerivative = [](double /*x*/) { return 1.0; };
    } else {
        throw std::invalid_argument("Unsupported activation function: " + activationName);
    }
}

DenseLayer::DenseLayer(int inputSize, int outputSize, bool useSoftmax, unsigned int seed)
    : inputSize(inputSize), outputSize(outputSize), isSoftmax(useSoftmax), activationName(useSoftmax ? "softmax" : "relu") {
    initializeWeights(seed, useSoftmax ? "" : "relu");
    
    if (useSoftmax) {
        activation = [](double x) { return x; }; 
        activationDerivative = [](double /*x*/) { return 1.0; };
    } else {
        activation = [](double x) { return ActivationFunctions::relu(x); };
        activationDerivative = [](double x) { return ActivationFunctions::reluDerivative(x); };
    }
}

std::vector<double> DenseLayer::forward(const std::vector<double>& input, bool training) {
    this->inputs = input;
    outputs.resize(outputSize);
    logits.resize(outputSize, 0.0);
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
            double activationGrad = activationDerivative(logits[i]);
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
    
    weights.resize(inputSize, std::vector<double>(outputSize));
    weightsGradients.resize(inputSize, std::vector<double>(outputSize, 0.0));
    biases.resize(outputSize);
    biasGradients.resize(outputSize, 0.0);

    if (activationName == "relu") {
        double stddev = std::sqrt(2.0 / static_cast<double>(inputSize));
        std::normal_distribution<> dis(0.0, stddev);
        
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                weights[i][j] = dis(gen);
            }
        }
        
        std::uniform_real_distribution<> bias_dis(0.01, 0.1);
        for (int j = 0; j < outputSize; ++j) {
            biases[j] = bias_dis(gen);
        }
    } else {
        double stddev = std::sqrt(1.0 / static_cast<double>(inputSize));
        std::normal_distribution<> dis(0.0, stddev);
        
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                weights[i][j] = dis(gen);
            }
        }
        
        for (int j = 0; j < outputSize; ++j) {
            biases[j] = 0.0;
        }
    }
}