#include <gtest/gtest.h>
#include "../include/SGD.h"
#include "../include/Momentum.h"
#include "../include/Adam.h"
#include <vector>

class OptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        weights = {{0.1, 0.2}, {0.3, 0.4}};
        biases = {0.1, 0.2};
        weight_gradients = {{0.01, 0.02}, {0.03, 0.04}};
        bias_gradients = {0.01, 0.02};
        learning_rate = 0.1;
    }
    
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<std::vector<double>> weight_gradients;
    std::vector<double> bias_gradients;
    double learning_rate;
};

TEST_F(OptimizerTest, SGD) {
    SGD optimizer;
    
    auto original_weights = weights;
    auto original_biases = biases;
    
    optimizer.updateWeights(weights, weight_gradients, learning_rate);
    optimizer.updateBiases(biases, bias_gradients, learning_rate);
    
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            double expected = original_weights[i][j] - learning_rate * weight_gradients[i][j];
            EXPECT_NEAR(weights[i][j], expected, 1e-10);
        }
    }
    
    for (size_t i = 0; i < biases.size(); ++i) {
        double expected = original_biases[i] - learning_rate * bias_gradients[i];
        EXPECT_NEAR(biases[i], expected, 1e-10);
    }
}

TEST_F(OptimizerTest, Momentum) {
    Momentum optimizer(0.9);
    
    auto original_weights = weights;
    auto original_biases = biases;
    
    optimizer.updateWeights(weights, weight_gradients, learning_rate);
    optimizer.updateBiases(biases, bias_gradients, learning_rate);
    
    EXPECT_NE(weights[0][0], original_weights[0][0]);
    EXPECT_NE(biases[0], original_biases[0]);
    
    auto weights_after_first = weights;
    optimizer.updateWeights(weights, weight_gradients, learning_rate);
    
    EXPECT_NE(weights[0][0], weights_after_first[0][0]);
}

TEST_F(OptimizerTest, Adam) {
    Adam optimizer(0.9, 0.999, 1e-8);
    
    auto original_weights = weights;
    auto original_biases = biases;
    
    optimizer.updateWeights(weights, weight_gradients, learning_rate);
    optimizer.updateBiases(biases, bias_gradients, learning_rate);
    
    EXPECT_NE(weights[0][0], original_weights[0][0]);
    EXPECT_NE(biases[0], original_biases[0]);
}

TEST(OptimizerEdgeCases, EmptyWeights) {
    SGD sgd;
    Momentum momentum(0.9);
    Adam adam(0.9, 0.999, 1e-8);
    
    std::vector<std::vector<double>> empty_weights;
    std::vector<double> empty_biases;
    std::vector<std::vector<double>> empty_weight_grads;
    std::vector<double> empty_bias_grads;
    
    EXPECT_NO_THROW({
        sgd.updateWeights(empty_weights, empty_weight_grads, 0.1);
        sgd.updateBiases(empty_biases, empty_bias_grads, 0.1);
        
        momentum.updateWeights(empty_weights, empty_weight_grads, 0.1);
        momentum.updateBiases(empty_biases, empty_bias_grads, 0.1);
        
        adam.updateWeights(empty_weights, empty_weight_grads, 0.1);
        adam.updateBiases(empty_biases, empty_bias_grads, 0.1);
    });
}
