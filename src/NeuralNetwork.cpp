#include "../include/NeuralNetwork.h"
#include "../include/ActivationFunctions.h"
#include "../include/LossFunction.h"

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes) {
    for (size_t i = 1; i < layerSizes.size(); ++i) {
        layers.emplace_back(layerSizes[i - 1], layerSizes[i]);
    }
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs,
                          const std::vector<std::vector<double>>& targets,
                          int epochs, double learningRate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            // Forward propagation
            std::vector<double> output = inputs[i];
            for (auto& layer : layers) {
                output = layer.forward(output);
            }

            // Compute loss
            totalLoss += LossFunction::meanSquaredError(output, targets[i]);

            // Compute gradient of loss w.r.t. output
            std::vector<double> lossGradient(output.size());
            for (size_t j = 0; j < output.size(); ++j) {
                lossGradient[j] = 2 * (output[j] - targets[i][j]);
            }

            // Backward propagation
            std::vector<double> gradients = lossGradient;
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                gradients = it->backward(gradients);
            }

            // Update weights and biases
            for (auto& layer : layers) {
                for (int j = 0; j < layer.getOutputSize(); ++j) {
                    for (int k = 0; k < layer.getInputSize(); ++k) {
                        layer.getWeights()[j][k] -= learningRate * layer.getWeightGradients()[j][k];
                    }
                    layer.getBiases()[j] -= learningRate * layer.getBiasGradients()[j];
                }
            }
        }

        // Print loss for the epoch
        std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss / inputs.size() << std::endl;
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    std::vector<double> output = input;
    for (auto& layer : layers) {
        output = layer.forward(output);
    }
    return output;
}
