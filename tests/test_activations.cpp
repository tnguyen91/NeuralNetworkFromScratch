#include <gtest/gtest.h>
#include "../include/ActivationFunctions.h"
#include <vector>
#include <cmath>

TEST(ActivationFunctions, Sigmoid) {
    EXPECT_NEAR(ActivationFunctions::sigmoid(0.0), 0.5, 1e-10);
    EXPECT_NEAR(ActivationFunctions::sigmoid(1.0), 0.7311, 1e-3);
    EXPECT_NEAR(ActivationFunctions::sigmoid(-1.0), 0.2689, 1e-3);
}

TEST(ActivationFunctions, SigmoidDerivative) {
    double sigmoid_output = ActivationFunctions::sigmoid(0.0);
    EXPECT_NEAR(ActivationFunctions::sigmoidDerivative(sigmoid_output), 0.25, 1e-10);
}

TEST(ActivationFunctions, ReLU) {
    EXPECT_EQ(ActivationFunctions::relu(2.0), 2.0);
    EXPECT_EQ(ActivationFunctions::relu(-2.0), 0.0);
    EXPECT_EQ(ActivationFunctions::relu(0.0), 0.0);
}

TEST(ActivationFunctions, ReLUDerivative) {
    EXPECT_EQ(ActivationFunctions::reluDerivative(2.0), 1.0);
    EXPECT_EQ(ActivationFunctions::reluDerivative(-2.0), 0.0);
}

TEST(ActivationFunctions, Softmax) {
    std::vector<double> input = {1.0, 2.0, 3.0};
    std::vector<double> result = ActivationFunctions::softmax(input);
    
    double sum = 0.0;
    for (double val : result) {
        sum += val;
        EXPECT_GE(val, 0.0);
    }
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST(ActivationFunctions, SoftmaxNumericalStability) {
    std::vector<double> input = {1000.0, 1001.0, 1002.0};
    std::vector<double> result = ActivationFunctions::softmax(input);
    
    double sum = 0.0;
    for (double val : result) {
        sum += val;
        EXPECT_FALSE(std::isinf(val));
        EXPECT_FALSE(std::isnan(val));
    }
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

