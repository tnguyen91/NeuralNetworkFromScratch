#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Layer.h"
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layerSizes);

    // Train the network
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               int epochs, double learningRate);

    // Predict outputs
    std::vector<double> predict(const std::vector<double>& input);

private:
    std::vector<Layer> layers;
};

#endif // NEURAL_NETWORK_H
