#include "ActivationFunctions.h"
#include <limits>
#include <algorithm>

namespace ActivationFunctions {

    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    double sigmoidDerivative(double sigmoid_output) {
        return sigmoid_output * (1.0 - sigmoid_output);
    }

    double relu(double x) {
        return std::max(0.0, x);
    }
    double reluDerivative(double x) {
        return x > 0.0 ? 1.0 : 0.0;
    }

    std::vector<double> softmax(const std::vector<double>& x) {
        // subtract max for numerical stability
        double maxVal = -std::numeric_limits<double>::infinity();
        for (double v : x) maxVal = std::max(maxVal, v);
        std::vector<double> expVals(x.size());
        double sumExp = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            double e = std::exp(x[i] - maxVal);
            expVals[i] = e;
            sumExp += e;
        }
        if (sumExp == 0.0) {
            double uniform = 1.0 / std::max<size_t>(1, x.size());
            return std::vector<double>(x.size(), uniform);
        }
        for (size_t i = 0; i < x.size(); ++i) {
            expVals[i] /= sumExp;
        }
        return expVals;
    }
}
