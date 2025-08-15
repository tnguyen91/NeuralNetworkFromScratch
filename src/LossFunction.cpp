#include "../include/LossFunction.h"
#include <cmath>

namespace LossFunction {

    double meanSquaredError(const std::vector<double>& predicted, const std::vector<double>& actual) {
        double sum = 0.0;
        for (size_t i = 0; i < predicted.size(); ++i) {
            double diff = predicted[i] - actual[i];
            sum += diff * diff;
        }
        return sum / predicted.size();
    }

    std::vector<double> meanSquaredErrorDerivative(const std::vector<double>& predicted, const std::vector<double>& actual) {
        std::vector<double> derivative(predicted.size());
        for (size_t i = 0; i < predicted.size(); ++i) {
            derivative[i] = 2 * (predicted[i] - actual[i]) / predicted.size();
        }
        return derivative;
    }

    double crossEntropy(const std::vector<double>& predicted, const std::vector<double>& actual) {
        double sum = 0.0;
        for (size_t i = 0; i < predicted.size(); ++i) {
            sum += actual[i] * std::log(predicted[i] + 1e-15); // avoid log(0)
        }
        return -sum;
    }

    std::vector<double> crossEntropyDerivative(const std::vector<double>& predicted, const std::vector<double>& actual) {
        std::vector<double> derivative(predicted.size());
        for (size_t i = 0; i < predicted.size(); ++i) {
            derivative[i] = predicted[i] - actual[i];
        }
        return derivative;
    }

}
