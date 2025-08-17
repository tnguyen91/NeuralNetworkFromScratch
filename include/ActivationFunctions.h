#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <vector>
#include <cmath>

namespace ActivationFunctions {
    double sigmoid(double x);
    double sigmoidDerivative(double sigmoid_output);
    
    double relu(double x);
    double reluDerivative(double x);

    std::vector<double> softmax(const std::vector<double>& x);
}

#endif 
