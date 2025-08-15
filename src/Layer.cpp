#include "../include/Layer.h"
#include "../include/ActivationFunctions.h"
#include <random>
#include <numeric>
#include <string>

Layer::Layer(int inputSize, int outputSize, unsigned int seed)
    : inputSize(inputSize), outputSize(outputSize) {

    activation = [](double x) {
        return ActivationFunctions::relu(x);
    };
    activationDerivative = [](double x) {
        return ActivationFunctions::reluDerivative(x);
    };
    initializeWeights(seed, std::string("relu"));
}

Layer::Layer(int inputSize, int outputSize, 
             std::function<double(double)> activation,
             std::function<double(double)> activationDerivative,
             unsigned int seed)
    : inputSize(inputSize), outputSize(outputSize), 
      activation(activation), activationDerivative(activationDerivative) {
    initializeWeights(seed, "");
}

Layer::Layer(int inputSize, int outputSize,
             std::function<double(double)> activation,
             std::function<double(double)> activationDerivative,
             const std::string& activationName,
             unsigned int seed)
    : inputSize(inputSize), outputSize(outputSize),
      activation(activation), activationDerivative(activationDerivative) {
    initializeWeights(seed, activationName);
}

Layer::Layer(int inputSize, int outputSize, bool useSoftmax, unsigned int seed)
    : inputSize(inputSize), outputSize(outputSize), isSoftmax(useSoftmax) {
    initializeWeights(seed, useSoftmax ? std::string("softmax") : std::string(""));
}

void Layer::initializeWeights(unsigned int seed, const std::string& activationName) {
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

    weights.resize(outputSize, std::vector<double>(inputSize));
    weightsGradients.resize(outputSize, std::vector<double>(inputSize, 0.0));
    biases.resize(outputSize);
    biasGradients.resize(outputSize, 0.0);

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weights[i][j] = dis(gen);
        }
        biases[i] = 0.0;
    }
}

std::vector<double> Layer::forward(const std::vector<double>& inputs) {
    this->inputs = inputs;
    outputs.resize(outputSize);
    std::vector<double> logits(outputSize, 0.0);
    for (int i = 0; i < outputSize; ++i) {
        double sum = biases[i];
        for (int j = 0; j < inputSize; ++j) {
            sum += weights[i][j] * inputs[j];
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

std::vector<double> Layer::backward(const std::vector<double>& gradients) {
    std::vector<double> inputGradients(inputSize, 0.0);
    if (isSoftmax) {
        const std::vector<double>& delta = gradients;
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                inputGradients[j] += weights[i][j] * delta[i];
                weightsGradients[i][j] = delta[i] * inputs[j];
            }
            biasGradients[i] = delta[i];
        }
    } else {
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
    }
    return inputGradients;
}

std::vector<std::vector<double>> Layer::computeWeightGradients(const std::vector<double>& gradients) {
    std::vector<std::vector<double>> weightGradients(outputSize, std::vector<double>(inputSize, 0.0));
    if (isSoftmax) {
        for (int i = 0; i < outputSize; ++i) {
            double delta = gradients[i];
            for (int j = 0; j < inputSize; ++j) {
                weightGradients[i][j] = delta * inputs[j];
            }
        }
    } else {
        for (int i = 0; i < outputSize; ++i) {
            double activationGrad = activationDerivative(outputs[i]);
            double delta = gradients[i] * activationGrad;
            for (int j = 0; j < inputSize; ++j) {
                weightGradients[i][j] = delta * inputs[j];
            }
        }
    }
    return weightGradients;
}

std::vector<double> Layer::computeBiasGradients(const std::vector<double>& gradients) {
    std::vector<double> biasGradients(outputSize, 0.0);
    if (isSoftmax) {
        for (int i = 0; i < outputSize; ++i) {
            biasGradients[i] = gradients[i];
        }
    } else {
        for (int i = 0; i < outputSize; ++i) {
            double activationGrad = activationDerivative(outputs[i]);
            biasGradients[i] = gradients[i] * activationGrad;
        }
    }
    return biasGradients;
}

int Layer::getInputSize() const {
    return inputSize;
}

int Layer::getOutputSize() const {
    return outputSize;
}

std::vector<std::vector<double>>& Layer::getWeights() {
    return weights;
}

std::vector<double>& Layer::getBiases() {
    return biases;
}

const std::vector<std::vector<double>>& Layer::getWeightGradients() const {
    return weightsGradients;
}

const std::vector<double>& Layer::getBiasGradients() const {
    return biasGradients;
}