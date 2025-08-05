#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Layer.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include <vector>
#include <memory>

class NeuralNetwork {
public:
    NeuralNetwork();

    NeuralNetwork(const std::vector<int>& layerSizes,
                  const std::string& activationFunction,
                  const std::string& lossFunction,
                  const std::string& optimizer,
                  unsigned int seed);

    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               int epochs, double learningRate);

    std::vector<double> predict(const std::vector<double>& input);

    void addLayer(std::unique_ptr<Layer> layer);

    double evaluate(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& targets,
                    double tolerance);

private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::function<double(const std::vector<double>&, const std::vector<double>&)> lossFunction;
    std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)> lossDerivative;
    std::unique_ptr<Optimizer> optimizer;
};

#endif
