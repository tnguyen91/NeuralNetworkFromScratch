#include <gtest/gtest.h>
#include "../include/ActivationFunctions.h"
#include <vector>  // For std::vector
#include <cmath>   // For std::exp and std::isfinite

TEST(ActivationFunctions, SigmoidAtZero) {
    double input = 0.0;
    double expected = 0.5;
    
    double result = ActivationFunctions::sigmoid(input);
    
    EXPECT_NEAR(result, expected, 1e-10);
}

TEST(ActivationFunctions, SigmoidPositiveInput) {
    double input = 1.0;
    double result = ActivationFunctions::sigmoid(input);
    
    EXPECT_GT(result, 0.5);

    EXPECT_GE(result, 0.0);
    EXPECT_LE(result, 1.0);
}

TEST(ActivationFunctions, SigmoidNegativeInput) {
    double input = -1.0;
    double result = ActivationFunctions::sigmoid(input);
    
    EXPECT_LT(result, 0.5);

    EXPECT_GE(result, 0.0);
    EXPECT_LE(result, 1.0);
}

TEST(ActivationFunctions, SigmoidDerivativeAtZero) {
    double input = 0.0;
    double expected = 0.25;
    
    double result = ActivationFunctions::sigmoidDerivativeFromInput(input);
    
    EXPECT_NEAR(result, expected, 1e-10);
}

TEST(ActivationFunctions, ReLUPositiveInput) {
    double input = 5.0;
    double result = ActivationFunctions::relu(input);
    
    EXPECT_EQ(result, input);
}

TEST(ActivationFunctions, ReLUNegativeInput) {
    double input = -3.0;
    double result = ActivationFunctions::relu(input);
    
    EXPECT_EQ(result, 0.0);
}

TEST(ActivationFunctions, ReLUAtZero) {
    double input = 0.0;
    double result = ActivationFunctions::relu(input);
    
    EXPECT_EQ(result, 0.0);
}

TEST(ActivationFunctions, ReLUDerivativePositive) {
    double input = 2.0;
    double result = ActivationFunctions::reluDerivative(input);
    
    EXPECT_EQ(result, 1.0);
}

TEST(ActivationFunctions, ReLUDerivativeNegative) {
    double input = -2.0;
    double result = ActivationFunctions::reluDerivative(input);
    
    EXPECT_EQ(result, 0.0);
}

TEST(ActivationFunctions, SigmoidProperties) {
    double x = 2.0;
    double pos = ActivationFunctions::sigmoid(x);
    double neg = ActivationFunctions::sigmoid(-x);
    
    EXPECT_NEAR(pos + neg, 1.0, 1e-10);
    
    std::vector<double> test_values = {-10.0, -1.0, 0.0, 1.0, 10.0};
    for (double val : test_values) {
        double result = ActivationFunctions::sigmoid(val);
        EXPECT_GT(result, 0.0) << "sigmoid(" << val << ") should be > 0";
        EXPECT_LT(result, 1.0) << "sigmoid(" << val << ") should be < 1";
    }
}

TEST(ActivationFunctions, SoftmaxBasic) {
    std::vector<double> input = {1.0, 2.0, 3.0};
    std::vector<double> result = ActivationFunctions::softmax(input);
    
    EXPECT_EQ(result.size(), input.size());
    
    double sum = 0.0;
    for (double val : result) {
        EXPECT_GT(val, 0.0);
        EXPECT_LT(val, 1.0);
        sum += val;
    }
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST(ActivationFunctions, SoftmaxNumericalStability) {
    std::vector<double> input = {1000.0, 1001.0, 999.0};
    std::vector<double> result = ActivationFunctions::softmax(input);
    
    for (double val : result) {
        EXPECT_TRUE(std::isfinite(val));
    }
    
    double sum = 0.0;
    for (double val : result) {
        sum += val;
    }
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

