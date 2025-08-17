#include "../include/NeuralNetwork.h"
#include "../include/ActivationFunctions.h"
#include "SGD.h"
#include "Momentum.h"
#include "Adam.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>

NeuralNetwork::NeuralNetwork() {
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes,
                             const std::string& activationFunction,
                             const std::string& lossFunction,
                             const std::string& optimizer,
                             unsigned int seed)
    : NeuralNetwork(
        layerSizes,
        activationFunction == "softmax" ? "relu" : activationFunction,
        activationFunction == "softmax" ? "softmax" : activationFunction,
        lossFunction,
        optimizer,
        seed) {}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes,
                             const std::string& hiddenAct,
                             const std::string& outputAct,
                             const std::string& lossFunction,
                             const std::string& optimizer,
                             unsigned int seed) {
    
    if (layerSizes.size() < 2) {
        throw std::invalid_argument("Network must have at least 2 layers (input and output)");
    }
    
    for (int size : layerSizes) {
        if (size <= 0) {
            throw std::invalid_argument("All layer sizes must be positive");
        }
    }
    
    if (outputAct == "softmax" && lossFunction != "crossEntropy") {
        std::cerr << "Warning: using softmax output with non-crossEntropy loss.\n";
    }

    createLayers(layerSizes, hiddenAct, outputAct, seed);
    
    setupLossFunction(lossFunction);
    
    setupOptimizer(optimizer);
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs,
                          const std::vector<std::vector<double>>& targets,
                          int epochs, double learningRate) {
    
    if (inputs.empty() || targets.empty()) {
        throw std::invalid_argument("Training data cannot be empty");
    }
    
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs must match number of targets");
    }
    
    if (epochs <= 0) {
        throw std::invalid_argument("Number of epochs must be positive");
    }
    
    if (learningRate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    
    if (layers.empty()) {
        throw std::runtime_error("Network has no layers");
    }
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<double> output = inputs[i];
            for (auto& layer : layers) {
                output = layer->forward(output);
            }

            totalLoss += lossFunction(output, targets[i]);
            std::vector<double> gradients = lossDerivative(output, targets[i]);

            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                gradients = (*it)->backward(gradients);
                
                optimizer->updateWeights((*it)->getWeights(), (*it)->getWeightGradients(), learningRate);
                optimizer->updateBiases((*it)->getBiases(), (*it)->getBiasGradients(), learningRate);
            }
        }

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << totalLoss / inputs.size() << std::endl;
        }
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    if (layers.empty()) {
        throw std::runtime_error("Network has no layers");
    }
    
    if (input.empty()) {
        throw std::invalid_argument("Input cannot be empty");
    }
    
    std::vector<double> output = input;
    for (auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void NeuralNetwork::addLayer(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

double NeuralNetwork::evaluate(const std::vector<std::vector<double>>& inputs,
                                const std::vector<std::vector<double>>& targets,
                                double tolerance) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Inputs and targets must have the same number of samples.");
    }
    int correctCount = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<double> output = predict(inputs[i]);
        
        bool isOneHot = false;
        if (!targets[i].empty()) {
            int oneCount = 0;
            bool hasNonBinary = false;
            for (double val : targets[i]) {
                if (std::abs(val - 1.0) < 1e-9) oneCount++;
                else if (std::abs(val) > 1e-9) hasNonBinary = true;
            }
            isOneHot = (oneCount == 1 && !hasNonBinary && targets[i].size() > 1);
        }
        
        if (isOneHot) {
            int predictedClass = std::max_element(output.begin(), output.end()) - output.begin();
            int actualClass = std::max_element(targets[i].begin(), targets[i].end()) - targets[i].begin();
            if (predictedClass == actualClass) {
                correctCount++;
            }
        } else if (targets[i].size() == 1) {
            double predicted = output[0];
            double actual = targets[i][0];
            
            if (std::abs(actual) < 1e-9 || std::abs(actual - 1.0) < 1e-9) {
                if ((predicted >= 0.5 && std::abs(actual - 1.0) < 1e-9) || 
                    (predicted < 0.5 && std::abs(actual) < 1e-9)) {
                    correctCount++;
                }
            } else {
                if (std::abs(predicted - actual) <= tolerance) {
                    correctCount++;
                }
            }
        } else {
            bool allWithinTolerance = true;
            for (size_t j = 0; j < output.size() && j < targets[i].size(); ++j) {
                if (std::abs(output[j] - targets[i][j]) > tolerance) {
                    allWithinTolerance = false;
                    break;
                }
            }
            if (allWithinTolerance) {
                correctCount++;
            }
        }
    }
    return static_cast<double>(correctCount) / inputs.size();
}

void NeuralNetwork::createLayers(const std::vector<int>& layerSizes,
                                const std::string& hiddenActivation,
                                const std::string& outputActivation,
                                unsigned int seed) {
    for (size_t i = 1; i < layerSizes.size(); ++i) {
        unsigned int layerSeed = (seed == 0) ? 0 : seed + static_cast<unsigned int>(i);
        bool isOutputLayer = (i == layerSizes.size() - 1);
        
        if (isOutputLayer) {
            layers.push_back(createOutputLayer(layerSizes[i - 1], layerSizes[i], 
                                             outputActivation, layerSeed));
        } else {
            layers.push_back(createHiddenLayer(layerSizes[i - 1], layerSizes[i], 
                                             hiddenActivation, layerSeed));
        }
    }
}

std::unique_ptr<Layer> NeuralNetwork::createHiddenLayer(int inputSize, int outputSize,
                                                       const std::string& activation,
                                                       unsigned int seed) {
    if (activation == "relu") {
        return std::make_unique<Layer>(inputSize, outputSize,
            [](double x) { return ActivationFunctions::relu(x); },
            [](double x) { return ActivationFunctions::reluDerivative(x); },
            "relu", seed);
    } else if (activation == "sigmoid") {
        return std::make_unique<Layer>(inputSize, outputSize,
            [](double x) { return ActivationFunctions::sigmoid(x); },
            [](double y) { return ActivationFunctions::sigmoidDerivative(y); },
            "sigmoid", seed);
    } else {
        throw std::invalid_argument("Unsupported hidden activation: " + activation);
    }
}

std::unique_ptr<Layer> NeuralNetwork::createOutputLayer(int inputSize, int outputSize,
                                                       const std::string& activation,
                                                       unsigned int seed) {
    if (activation == "softmax") {
        return std::make_unique<Layer>(inputSize, outputSize, true, seed);
    } else if (activation == "relu") {
        return std::make_unique<Layer>(inputSize, outputSize,
            [](double x) { return ActivationFunctions::relu(x); },
            [](double x) { return ActivationFunctions::reluDerivative(x); },
            "relu", seed);
    } else if (activation == "sigmoid") {
        return std::make_unique<Layer>(inputSize, outputSize,
            [](double x) { return ActivationFunctions::sigmoid(x); },
            [](double y) { return ActivationFunctions::sigmoidDerivative(y); },
            "sigmoid", seed);
    } else if (activation == "linear") {
        return std::make_unique<Layer>(inputSize, outputSize,
            [](double x) { return x; },
            [](double /*y*/) { return 1.0; },
            "linear", seed);
    } else {
        throw std::invalid_argument("Unsupported output activation: " + activation);
    }
}

void NeuralNetwork::setupLossFunction(const std::string& lossFunction) {
    if (lossFunction == "crossEntropy") {
        this->lossFunction = LossFunction::crossEntropy;
        this->lossDerivative = LossFunction::crossEntropyDerivative;
    } else if (lossFunction == "meanSquaredError") {
        this->lossFunction = LossFunction::meanSquaredError;
        this->lossDerivative = LossFunction::meanSquaredErrorDerivative;
    } else {
        throw std::invalid_argument("Unsupported loss function: " + lossFunction);
    }
}

void NeuralNetwork::setupOptimizer(const std::string& optimizer) {
    if (optimizer == "SGD") {
        this->optimizer = std::make_unique<SGD>();
    } else if (optimizer == "Momentum") {
        this->optimizer = std::make_unique<Momentum>(0.9);
    } else if (optimizer == "Adam") {
        this->optimizer = std::make_unique<Adam>(0.9, 0.999, 1e-8);
    } else {
        throw std::invalid_argument("Unsupported optimizer: " + optimizer);
    }
}
