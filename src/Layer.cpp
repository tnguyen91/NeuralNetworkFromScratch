#include "../include/Layer.h"
#include "../include/ActivationFunctions.h"
#include <random>

Layer::Layer(int inputSize, int outputSize)
    : inputSize(inputSize), outputSize(outputSize) {
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
    this->inputs = inputs; // Store inputs for use in backward propagation
    std::vector<double> outputs(outputSize, 0.0);

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            outputs[i] += weights[i][j] * inputs[j];
        }
        outputs[i] += biases[i]; // Add bias
        outputs[i] = ActivationFunctions::sigmoid(outputs[i]); // Apply activation function
    }

    this->outputs = outputs; // Store outputs for use in backward propagation
    return outputs;
}

std::vector<double> Layer::backward(const std::vector<double>& gradients) {
    std::vector<double> inputGradients(inputSize, 0.0);
    weightGradients.resize(outputSize, std::vector<double>(inputSize, 0.0));
    biasGradients.resize(outputSize, 0.0);

    for (int i = 0; i < outputSize; ++i) {
        double delta = gradients[i] * (outputs[i] * (1 - outputs[i])); // Derivative of sigmoid

        for (int j = 0; j < inputSize; ++j) {
            weightGradients[i][j] += delta * inputs[j];
            inputGradients[j] += delta * weights[i][j];
        }

        biasGradients[i] += delta; // Gradient for bias
    }

    return inputGradients;
}
