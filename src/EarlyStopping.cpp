#include "EarlyStopping.h"
#include <limits>

EarlyStopping::EarlyStopping(int patience)
    : best_loss(std::numeric_limits<double>::max()), patience(patience), counter(0), first(true) {}

bool EarlyStopping::should_stop(double val_loss) {
    if (first || val_loss < best_loss) {
        best_loss = val_loss;
        counter = 0;
        first = false;
    } else {
        counter++;
    }
    return counter >= patience;
}
