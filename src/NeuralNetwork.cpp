#include "../include/NeuralNetwork.h"
#include "../include/ActivationFunctions.h"
#include <iostream>

NeuralNetwork::NeuralNetwork() {
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes) {
    for (size_t i = 1; i < layerSizes.size(); ++i) {
        auto layer = std::make_unique<Layer>(layerSizes[i - 1], layerSizes[i]);
        layers.push_back(std::move(layer));
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

            totalLoss += LossFunction::meanSquaredError(output, targets[i]);

            std::vector<double> gradients = LossFunction::meanSquaredErrorDerivative(output, targets[i]);

            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                gradients = (*it)->backward(gradients, learningRate);
            }
        }

        std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss / inputs.size() << std::endl;
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
        auto output = predict(inputs[i]);
        size_t predicted = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        size_t actual = std::distance(targets[i].begin(), std::max_element(targets[i].begin(), targets[i].end()));
        if (predicted == actual) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / inputs.size();
}
