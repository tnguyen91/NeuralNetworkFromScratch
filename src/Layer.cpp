#include "../include/Layer.h"
#include "../include/ActivationFunctions.h"
#include <random>
#include <numeric>

Layer::Layer(int inputSize, int outputSize)
    : inputSize(inputSize), outputSize(outputSize) {
    
    activation = [](double x) { 
        return ActivationFunctions::sigmoid(x); 
    };
    activationDerivative = [](double x) { 
        return ActivationFunctions::sigmoidDerivative(x); 
    };
    
    initializeWeights();
}

Layer::Layer(int inputSize, int outputSize, 
             std::function<double(double)> activation,
             std::function<double(double)> activationDerivative)
    : inputSize(inputSize), outputSize(outputSize), 
      activation(activation), activationDerivative(activationDerivative) {
    
    initializeWeights();
}

void Layer::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    double limit = std::sqrt(2.0 / (inputSize + outputSize)) * 0.5;
    std::uniform_real_distribution<> dis(-limit, limit);

    weights.resize(outputSize, std::vector<double>(inputSize));
    weightsGradients.resize(outputSize, std::vector<double>(inputSize, 0.0));
    biases.resize(outputSize);
    biasGradients.resize(outputSize, 0.0);

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
        double sum = biases[i];
        for (int j = 0; j < inputSize; ++j) {
            sum += weights[i][j] * inputs[j];
        }
        outputs[i] = activation(sum);
    }
    return outputs;
}

std::vector<double> Layer::backward(const std::vector<double>& gradients) {
    std::vector<double> inputGradients(inputSize, 0.0);
    for (int i = 0; i < outputSize; ++i) {
        double activationGrad = activationDerivative(outputs[i]);
        double delta = gradients[i] * activationGrad;

        for (int j = 0; j < inputSize; ++j) {
            inputGradients[j] += weights[i][j] * delta;
        }

        for (int j = 0; j < inputSize; ++j) {
            weightsGradients[i][j] = delta * inputs[j];
        }

        biasGradients[i] = delta;
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
    return weightsGradients;
}

const std::vector<double>& Layer::getBiasGradients() const {
    return biasGradients;
}