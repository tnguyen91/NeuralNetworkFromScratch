#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <vector>

namespace LossFunction {

    double meanSquaredError(const std::vector<double>& predicted, const std::vector<double>& actual);
    std::vector<double> meanSquaredErrorDerivative(const std::vector<double>& predicted, const std::vector<double>& actual);
    double crossEntropy(const std::vector<double>& predicted, const std::vector<double>& actual);
    std::vector<double> crossEntropyDerivative(const std::vector<double>& predicted, const std::vector<double>& actual);

}

#endif
