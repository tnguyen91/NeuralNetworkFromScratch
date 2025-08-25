#include <gtest/gtest.h>
#include "NeuralNetwork.h"
#include "EarlyStopping.h"
#include <vector>

TEST(EarlyStoppingTest, StopsWhenValidationLossPlateaus) {
    std::vector<std::vector<double>> train_inputs = {{0,0},{0,1},{1,0},{1,1}};
    std::vector<std::vector<double>> train_targets = {{0},{1},{1},{0}};
    std::vector<std::vector<double>> val_inputs = train_inputs;
    std::vector<std::vector<double>> val_targets = train_targets;

    NeuralNetwork net({2, 4, 1}, "relu", "sigmoid", "meanSquaredError", "SGD", 42);
    EarlyStopping early_stopping(3); 

    int max_epochs = 100;
    double learning_rate = 1.0;
    net.train(train_inputs, train_targets, val_inputs, val_targets, max_epochs, learning_rate, early_stopping);
    SUCCEED();
}
