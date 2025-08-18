#include "LossFunction.h"
#include <cmath>
#include <algorithm>

namespace LossFunction {

    double meanSquaredError(const std::vector<double>& predicted, const std::vector<double>& actual) {
        if (predicted.empty()) return 0.0;
        
        double sum = 0.0;
        for (size_t i = 0; i < predicted.size(); ++i) {
            double diff = predicted[i] - actual[i];
            sum += diff * diff;
        }
        return sum / predicted.size();
    }

    std::vector<double> meanSquaredErrorDerivative(const std::vector<double>& predicted, const std::vector<double>& actual) {
        std::vector<double> derivative(predicted.size());
        if (predicted.empty()) return derivative;
        
        for (size_t i = 0; i < predicted.size(); ++i) {
            derivative[i] = 2.0 * (predicted[i] - actual[i]) / predicted.size();
        }
        return derivative;
    }

    double crossEntropy(const std::vector<double>& predicted, const std::vector<double>& actual) {
        if (predicted.empty()) return 0.0;
        
        double sum = 0.0;
        constexpr double epsilon = 1e-15;
        for (size_t i = 0; i < predicted.size(); ++i) {
            double p = std::max(epsilon, std::min(1.0 - epsilon, predicted[i]));
            sum += actual[i] * std::log(p);
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
