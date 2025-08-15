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

    double sigmoidDerivativeFromInput(double x) {
        double sig = sigmoid(x);
        return sig * (1 - sig);
    }

    double sigmoidDerivative(double sigmoid_output) {
        return sigmoid_output * (1.0 - sigmoid_output);
    }

    double reluDerivative(double x) {
        return x > 0.0 ? 1.0 : 0.0;
    }

}
