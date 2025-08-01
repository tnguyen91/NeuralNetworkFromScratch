#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <vector>
#include <cmath>

namespace ActivationFunctions {

    double sigmoid(double x);
    std::vector<double> sigmoid(const std::vector<double>& x);

    double sigmoidDerivative(double x);

    double relu(double x);
    std::vector<double> relu(const std::vector<double>& x);

    double reluDerivative(double x);

}

#endif 
