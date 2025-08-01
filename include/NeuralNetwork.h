#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Layer.h"
#include "LossFunction.h"
#include <vector>
#include <memory>

class NeuralNetwork {
public:
    NeuralNetwork();
    
    NeuralNetwork(const std::vector<int>& layerSizes);

    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               int epochs, double learningRate);

    std::vector<double> predict(const std::vector<double>& input);

    void addLayer(std::unique_ptr<Layer> layer);

    double evaluate(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& targets);

private:
    std::vector<std::unique_ptr<Layer>> layers;
};

#endif
