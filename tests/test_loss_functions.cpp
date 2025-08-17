#include <gtest/gtest.h>
#include "../include/LossFunction.h"
#include <vector>
#include <cmath>

TEST(LossFunction, MeanSquaredError) {
    std::vector<double> predicted = {0.8, 0.3, 0.9};
    std::vector<double> actual = {1.0, 0.0, 0.5};
    
    double result = LossFunction::meanSquaredError(predicted, actual);
    EXPECT_NEAR(result, 0.0967, 1e-3);
}

TEST(LossFunction, MSEPerfectPrediction) {
    std::vector<double> predicted = {1.0, 0.5, 0.0};
    std::vector<double> actual = {1.0, 0.5, 0.0};
    
    double result = LossFunction::meanSquaredError(predicted, actual);
    EXPECT_NEAR(result, 0.0, 1e-10);
}

TEST(LossFunction, MSEDerivative) {
    std::vector<double> predicted = {0.8, 0.3, 0.9};
    std::vector<double> actual = {1.0, 0.0, 0.5};
    
    auto result = LossFunction::meanSquaredErrorDerivative(predicted, actual);
    EXPECT_NEAR(result[0], -0.1333, 1e-3);
    EXPECT_NEAR(result[1], 0.2, 1e-3);
    EXPECT_NEAR(result[2], 0.2667, 1e-3);
}

TEST(LossFunction, CrossEntropy) {
    std::vector<double> predicted = {0.8, 0.3, 0.9};
    std::vector<double> actual = {1.0, 0.0, 0.5};
    
    double result = LossFunction::crossEntropy(predicted, actual);
    EXPECT_NEAR(result, 0.2758, 1e-3);
}

TEST(LossFunction, CrossEntropyDerivative) {
    std::vector<double> predicted = {0.8, 0.3, 0.9};
    std::vector<double> actual = {1.0, 0.0, 0.5};
    
    auto result = LossFunction::crossEntropyDerivative(predicted, actual);
    std::vector<double> expected = {-0.2, 0.3, 0.4};
    
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-10);
    }
}

TEST(LossFunction, EmptyVectors) {
    std::vector<double> empty;
    
    EXPECT_NO_THROW({
        double result = LossFunction::meanSquaredError(empty, empty);
        EXPECT_EQ(result, 0.0);
    });
    
    EXPECT_NO_THROW({
        double result = LossFunction::crossEntropy(empty, empty);
        EXPECT_EQ(result, 0.0);
    });
}
