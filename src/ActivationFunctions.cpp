#include "../include/ActivationFunctions.h"

namespace ActivationFunctions {

    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    std::vector<double> sigmoid(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = sigmoid(x[i]);
        }
        return result;
    }

    double relu(double x) {
        return std::max(0.0, x);
    }

    std::vector<double> relu(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = relu(x[i]);
        }
        return result;
    }

} // namespace ActivationFunctions
