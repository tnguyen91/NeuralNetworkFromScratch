#pragma once

class EarlyStopping {
public:
    EarlyStopping(int patience);
    bool should_stop(double val_loss);

private:
    double best_loss;
    int patience;
    int counter;
    bool first;
};
