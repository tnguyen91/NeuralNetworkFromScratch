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

} // namespace LossFunction
