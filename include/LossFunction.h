#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <vector>

namespace LossFunction {

    // Mean Squared Error (MSE)
    double meanSquaredError(const std::vector<double>& predicted, const std::vector<double>& actual);

} // namespace LossFunction

#endif // LOSS_FUNCTION_H
