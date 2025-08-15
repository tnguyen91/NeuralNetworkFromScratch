#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <vector>
#include <cmath>

namespace ActivationFunctions {
    double sigmoid(double x);
    std::vector<double> sigmoid(const std::vector<double>& x);
    
    double relu(double x);
    std::vector<double> relu(const std::vector<double>& x);
    
    std::vector<double> softmax(const std::vector<double>& x);
    
    double sigmoidDerivativeFromInput(double x);
    double sigmoidDerivative(double sigmoid_output);
    
    double reluDerivative(double x);

}

#endif 
