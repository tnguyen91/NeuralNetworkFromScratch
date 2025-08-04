#include "../include/NeuralNetwork.h"
#include "../include/ActivationFunctions.h"
#include "SGD.h"
#include "Momentum.h"
#include "Adam.h"
#include <iostream>

NeuralNetwork::NeuralNetwork() {
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes,
                             const std::string& activationFunction = "sigmoid",
                             const std::string& lossFunction = "crossEntropy",
                             const std::string& optimizer = "SGD") {
    if (activationFunction == "sigmoid") {
        for (size_t i = 1; i < layerSizes.size(); ++i) {
            layers.push_back(std::make_unique<Layer>(
                layerSizes[i - 1], layerSizes[i],
                [](double x) {
                    return ActivationFunctions::sigmoid(x);
                },
                [](double sigmoid_output) {
                    return ActivationFunctions::sigmoidDerivative(sigmoid_output);
                }
            ));
        }
    } else if (activationFunction == "relu") {
        for (size_t i = 1; i < layerSizes.size(); ++i) {
            layers.push_back(std::make_unique<Layer>(
                layerSizes[i - 1], layerSizes[i],
                [](double x) {
                    return ActivationFunctions::relu(x);
                },
                [](double x) {
                    return ActivationFunctions::reluDerivative(x);
                }
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
                                const std::vector<std::vector<double>>& targets) {
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<double> prediction = predict(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "] -> Prediction: [" << prediction[0]
                  << "], Target: [" << targets[i][0] << "]" << std::endl;
        if ((prediction[0] >= 0.5 && targets[i][0] == 1.0) ||
            (prediction[0] < 0.5 && targets[i][0] == 0.0)) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / inputs.size();
}
