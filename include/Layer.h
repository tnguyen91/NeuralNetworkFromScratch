#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <functional>
#include <string>

class Layer {
public:
    Layer(int inputSize, int outputSize, unsigned int seed = 0);
    Layer(int inputSize, int outputSize, std::function<double(double)> activation,
          std::function<double(double)> activationDerivative, unsigned int seed = 0);
    Layer(int inputSize, int outputSize, std::function<double(double)> activation,
          std::function<double(double)> activationDerivative, const std::string& activationName, 
          unsigned int seed = 0);
    Layer(int inputSize, int outputSize, bool useSoftmax, unsigned int seed = 0);

    std::vector<double> forward(const std::vector<double>& inputs);

    std::vector<double> backward(const std::vector<double>& gradients);

    std::vector<std::vector<double>> computeWeightGradients(const std::vector<double>& gradients);
    std::vector<double> computeBiasGradients(const std::vector<double>& gradients);

    int getInputSize() const;
    int getOutputSize() const;
    std::vector<std::vector<double>>& getWeights() ;
    std::vector<double>& getBiases() ;
    
    const std::vector<std::vector<double>>& getWeightGradients() const;
    const std::vector<double>& getBiasGradients() const;

private:
    int inputSize;
    int outputSize;
    
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> weightsGradients;
    std::vector<double> biases;
    std::vector<double> biasGradients;

    std::vector<double> inputs; 
    std::vector<double> outputs;

    std::function<double(double)> activation;
    std::function<double(double)> activationDerivative;
    bool isSoftmax = false;

    void initializeWeights(unsigned int seed, const std::string& activationName);
};

#endif
