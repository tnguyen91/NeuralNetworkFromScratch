#include "../include/NeuralNetwork.h"
#include "../include/ActivationFunctions.h"
#include "SGD.h"
#include "Momentum.h"
#include "Adam.h"
#include <iostream>
#include <algorithm>
#include <cmath>

NeuralNetwork::NeuralNetwork() {
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes,
                             const std::string& activationFunction,
                             const std::string& lossFunction,
                             const std::string& optimizer,
                             unsigned int seed) {
    if (activationFunction == "sigmoid") {
        for (size_t i = 1; i < layerSizes.size(); ++i) {
            unsigned int layerSeed = (seed == 0) ? 0 : seed + i;
            layers.push_back(std::make_unique<Layer>(
                layerSizes[i - 1], layerSizes[i],
                [](double x) {
                    return ActivationFunctions::sigmoid(x);
                },
                [](double sigmoid_output) {
                    return ActivationFunctions::sigmoidDerivative(sigmoid_output);
                },
                layerSeed
            ));
        }
    } else if (activationFunction == "relu") {
        for (size_t i = 1; i < layerSizes.size(); ++i) {
            unsigned int layerSeed = (seed == 0) ? 0 : seed + i;
            layers.push_back(std::make_unique<Layer>(
                layerSizes[i - 1], layerSizes[i],
                [](double x) {
                    return ActivationFunctions::relu(x);
                },
                [](double x) {
                    return ActivationFunctions::reluDerivative(x);
                },
                layerSeed
            ));
        }
    } else {
        throw std::invalid_argument("Unsupported activation function: " + activationFunction);
    }

    if (lossFunction == "crossEntropy") {
        this->lossFunction = LossFunction::crossEntropy;
        this->lossDerivative = LossFunction::crossEntropyDerivative;
    } else if (lossFunction == "meanSquaredError") {
        this->lossFunction = LossFunction::meanSquaredError;
        this->lossDerivative = LossFunction::meanSquaredErrorDerivative;
    }

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

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs,
                          const std::vector<std::vector<double>>& targets,
                          int epochs, double learningRate) {
    
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
                auto weightGradients = (*it)->computeWeightGradients(gradients);
                optimizer->updateWeights((*it)->getWeights(), weightGradients, learningRate);

                auto biasGradients = (*it)->computeBiasGradients(gradients);
                optimizer->updateBiases((*it)->getBiases(), biasGradients, learningRate);

                gradients = (*it)->backward(gradients);
            }
        }

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << totalLoss / inputs.size() << std::endl;
        }
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
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
