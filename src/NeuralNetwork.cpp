#include "NeuralNetwork.h"
#include "EarlyStopping.h"
#include "ActivationFunctions.h"
#include "DenseLayer.h"
#include "SGD.h"
#include "Momentum.h"
#include "Adam.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <cstdint>

NeuralNetwork::NeuralNetwork() : l2_lambda_(0.0) {
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes,
                             const std::string& activationFunction,
                             const std::string& lossFunction,
                             const std::string& optimizer,
                             double l2_lambda,
                             unsigned int seed)
    : NeuralNetwork(
        layerSizes,
        activationFunction == "softmax" ? "relu" : activationFunction,
        activationFunction == "softmax" ? "softmax" : activationFunction,
        lossFunction,
        optimizer,
        l2_lambda,
        seed) {}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes,
                             const std::string& hiddenAct,
                             const std::string& outputAct,
                             const std::string& lossFunction,
                             const std::string& optimizer,
                             double l2_lambda,
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
    setupOptimizer(optimizer, l2_lambda);
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
                output = layer->forward(output, true);
            }

            totalLoss += lossFunction(output, targets[i]);
            std::vector<double> gradients = lossDerivative(output, targets[i]);

            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                gradients = (*it)->backward(gradients);
                
                if (auto dense = dynamic_cast<DenseLayer*>(it->get())) {
                    optimizer->updateWeights(dense->getWeights(), dense->getWeightGradients(), learningRate);
                    optimizer->updateBiases(dense->getBiases(), dense->getBiasGradients(), learningRate);
                }
            }
        }

        if (epoch % 100 == 0 || epoch < 5) {
            std::cout << "Epoch " << epoch << ", Loss: " << totalLoss / inputs.size() << std::endl;
        }
    }
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& train_inputs,
                          const std::vector<std::vector<double>>& train_targets,
                          const std::vector<std::vector<double>>& val_inputs,
                          const std::vector<std::vector<double>>& val_targets,
                          int epochs, double learningRate,
                          EarlyStopping& early_stopping) {
    if (train_inputs.empty() || train_targets.empty()) {
        throw std::invalid_argument("Training data cannot be empty");
    }
    if (train_inputs.size() != train_targets.size()) {
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
        for (size_t i = 0; i < train_inputs.size(); ++i) {
            std::vector<double> output = train_inputs[i];
            for (auto& layer : layers) {
                output = layer->forward(output, true);
            }
            totalLoss += lossFunction(output, train_targets[i]);
            std::vector<double> gradients = lossDerivative(output, train_targets[i]);
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                gradients = (*it)->backward(gradients);
                if (auto dense = dynamic_cast<DenseLayer*>(it->get())) {
                    optimizer->updateWeights(dense->getWeights(), dense->getWeightGradients(), learningRate);
                    optimizer->updateBiases(dense->getBiases(), dense->getBiasGradients(), learningRate);
                }
            }
        }
        double val_loss = 0.0;
        for (size_t i = 0; i < val_inputs.size(); ++i) {
            std::vector<double> output = val_inputs[i];
            for (auto& layer : layers) {
                output = layer->forward(output, false);
            }
            val_loss += lossFunction(output, val_targets[i]);
        }
        val_loss /= val_inputs.size();
        std::cout << "Epoch " << epoch << ", Train Loss: " << totalLoss / train_inputs.size() << ", Val Loss: " << val_loss << std::endl;

        if (early_stopping.should_stop(val_loss)) {
            std::cout << "Early stopping at epoch " << epoch << std::endl;
            break;
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
        output = layer->forward(output, false);
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

void NeuralNetwork::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for saving: " + filename);
    }

    int32_t num_layers = static_cast<int32_t>(layers.size());
    out.write(reinterpret_cast<const char*>(&num_layers), sizeof(int32_t));

    std::vector<int32_t> layer_sizes;
    if (num_layers == 0) {
        throw std::runtime_error("Cannot save: network has no layers.");
    }
    const DenseLayer* first = dynamic_cast<const DenseLayer*>(layers[0].get());
    if (!first) throw std::runtime_error("Save only supports DenseLayer layers.");
    layer_sizes.push_back(static_cast<int32_t>(first->getInputSize()));
    for (const auto& layer_ptr : layers) {
        const DenseLayer* dense = dynamic_cast<const DenseLayer*>(layer_ptr.get());
        if (!dense) throw std::runtime_error("Save only supports DenseLayer layers.");
        layer_sizes.push_back(static_cast<int32_t>(dense->getOutputSize()));
    }
    out.write(reinterpret_cast<const char*>(layer_sizes.data()), sizeof(int32_t) * layer_sizes.size());

    // Write weights, biases, and activation name for each layer
    for (const auto& layer_ptr : layers) {
        DenseLayer* dense = dynamic_cast<DenseLayer*>(layer_ptr.get());
        if (!dense) throw std::runtime_error("Save only supports DenseLayer layers.");
        int32_t in_size = static_cast<int32_t>(dense->getInputSize());
        int32_t out_size = static_cast<int32_t>(dense->getOutputSize());
        out.write(reinterpret_cast<const char*>(&in_size), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(&out_size), sizeof(int32_t));

        const std::string& act = dense->getActivationName();
        int32_t act_len = static_cast<int32_t>(act.size());
        out.write(reinterpret_cast<const char*>(&act_len), sizeof(int32_t));
        out.write(act.data(), act_len);

        const auto& weights = dense->getWeights();
        for (int i = 0; i < in_size; ++i) {
            out.write(reinterpret_cast<const char*>(weights[i].data()), sizeof(double) * out_size);
        }
        const auto& biases = dense->getBiases();
        out.write(reinterpret_cast<const char*>(biases.data()), sizeof(double) * out_size);
    }

    if (!out) {
        throw std::runtime_error("Error occurred while writing to file: " + filename);
    }
    out.close();
}

void NeuralNetwork::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for loading: " + filename);
    }

    int32_t num_layers = 0;
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(int32_t));
    if (!in || num_layers <= 0) {
        throw std::runtime_error("Invalid or corrupt model file (num_layers).");
    }

    std::vector<int32_t> layer_sizes(num_layers + 1);
    in.read(reinterpret_cast<char*>(layer_sizes.data()), sizeof(int32_t) * (num_layers + 1));
    if (!in) {
        throw std::runtime_error("Invalid or corrupt model file (layer_sizes).");
    }

    layers.clear();

    for (int l = 0; l < num_layers; ++l) {
        int32_t in_size = 0, out_size = 0;
        in.read(reinterpret_cast<char*>(&in_size), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&out_size), sizeof(int32_t));
        if (!in || in_size != layer_sizes[l] || out_size != layer_sizes[l+1]) {
            throw std::runtime_error("Model file mismatch or corrupt (layer size).");
        }

        int32_t act_len = 0;
        in.read(reinterpret_cast<char*>(&act_len), sizeof(int32_t));
        if (!in || act_len <= 0 || act_len > 100) {
            throw std::runtime_error("Model file corrupt (activation name length).");
        }
        std::string act(act_len, '\0');
        in.read(&act[0], act_len);
        if (!in) {
            throw std::runtime_error("Model file corrupt (activation name read).");
        }

        std::unique_ptr<DenseLayer> layer;
        if (act == "relu") {
            layer = std::make_unique<DenseLayer>(in_size, out_size, "relu");
        } else if (act == "sigmoid") {
            layer = std::make_unique<DenseLayer>(in_size, out_size, "sigmoid");
        } else if (act == "softmax") {
            layer = std::make_unique<DenseLayer>(in_size, out_size, true);
        } else if (act == "linear") {
            layer = std::make_unique<DenseLayer>(in_size, out_size, "linear");
        } else {
            throw std::runtime_error("Unsupported activation in model file: " + act);
        }

        std::vector<std::vector<double>>& weights = layer->getWeights();
        for (int i = 0; i < in_size; ++i) {
            in.read(reinterpret_cast<char*>(weights[i].data()), sizeof(double) * out_size);
        }
        std::vector<double>& biases = layer->getBiases();
        in.read(reinterpret_cast<char*>(biases.data()), sizeof(double) * out_size);

        if (!in) {
            throw std::runtime_error("Model file corrupt (weights/biases).");
        }

        layers.push_back(std::move(layer));
    }

    in.close();
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
        return std::make_unique<DenseLayer>(inputSize, outputSize, "relu", seed);
    } else if (activation == "sigmoid") {
        return std::make_unique<DenseLayer>(inputSize, outputSize, "sigmoid", seed);
    } else {
        throw std::invalid_argument("Unsupported hidden activation: " + activation);
    }
}

std::unique_ptr<Layer> NeuralNetwork::createOutputLayer(int inputSize, int outputSize,
                                                       const std::string& activation,
                                                       unsigned int seed) {
    if (activation == "softmax") {
        return std::make_unique<DenseLayer>(inputSize, outputSize, true, seed);
    } else if (activation == "relu") {
        return std::make_unique<DenseLayer>(inputSize, outputSize, "relu", seed);
    } else if (activation == "sigmoid") {
        return std::make_unique<DenseLayer>(inputSize, outputSize, "sigmoid", seed);
    } else if (activation == "linear") {
        return std::make_unique<DenseLayer>(inputSize, outputSize, "linear", seed);
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

void NeuralNetwork::setupOptimizer(const std::string& optimizer, double l2_lambda) {
    if (optimizer == "SGD") {
        this->optimizer = std::make_unique<SGD>(l2_lambda);
    } else if (optimizer == "Momentum") {
        this->optimizer = std::make_unique<Momentum>(0.9, l2_lambda);
    } else if (optimizer == "Adam") {
        this->optimizer = std::make_unique<Adam>(0.9, 0.999, 1e-8, l2_lambda);
    } else {
        throw std::invalid_argument("Unsupported optimizer: " + optimizer);
    }
}
