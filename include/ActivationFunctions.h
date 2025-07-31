#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <vector>
#include <cmath>

namespace ActivationFunctions {

    // Sigmoid activation function
    double sigmoid(double x);
    std::vector<double> sigmoid(const std::vector<double>& x);

    // ReLU activation function
    double relu(double x);
    std::vector<double> relu(const std::vector<double>& x);

} // namespace ActivationFunctions

#endif // ACTIVATION_FUNCTIONS_H
