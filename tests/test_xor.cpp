#include <gtest/gtest.h>
#include "../include/NeuralNetwork.h"

TEST(XORNeuralNetwork, LearnsXOR) {
    std::vector<int> layers = {2, 8, 1};
    NeuralNetwork net(layers, "relu", "sigmoid", "meanSquaredError", "Adam", 42);
    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<std::vector<double>> targets = {
        {0}, {1}, {1}, {0}
    };
    net.train(inputs, targets, 1000, 0.02);
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto pred = net.predict(inputs[i]);
        double expected = targets[i][0];
        EXPECT_NEAR(pred[0], expected, 0.2) << "Failed on input: [" << inputs[i][0] << ", " << inputs[i][1] << "]";
    }
}
