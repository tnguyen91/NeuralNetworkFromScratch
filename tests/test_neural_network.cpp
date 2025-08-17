#include <gtest/gtest.h>
#include "../include/NeuralNetwork.h"
#include <vector>

class NeuralNetworkTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::vector<int> layer_sizes = {2, 3, 2};
        network = std::make_unique<NeuralNetwork>(
            layer_sizes, "relu", "meanSquaredError", "SGD", 42
        );
    }
    
    std::unique_ptr<NeuralNetwork> network;
};

TEST_F(NeuralNetworkTest, Constructor) {
    EXPECT_NO_THROW({
        NeuralNetwork nn({2, 3, 1}, "sigmoid", "meanSquaredError", "SGD");
    });
}

TEST_F(NeuralNetworkTest, ConstructorValidation) {
    EXPECT_THROW({
        NeuralNetwork nn({1}, "relu", "meanSquaredError", "SGD"); 
    }, std::invalid_argument);
    
    EXPECT_THROW({
        NeuralNetwork nn({2, 0, 1}, "relu", "meanSquaredError", "SGD"); 
    }, std::invalid_argument);
    
    EXPECT_THROW({
        NeuralNetwork nn({2, 3, 1}, "invalid", "meanSquaredError", "SGD");
    }, std::invalid_argument);
}

TEST_F(NeuralNetworkTest, Prediction) {
    std::vector<double> input = {0.5, -0.3};
    std::vector<double> output = network->predict(input);
    
    EXPECT_EQ(output.size(), 2);
    for (double val : output) {
        EXPECT_GE(val, 0.0);
    }
}

TEST_F(NeuralNetworkTest, PredictionValidation) {
    std::vector<double> empty_input;
    
    EXPECT_THROW({
        network->predict(empty_input);
    }, std::invalid_argument);
}

TEST_F(NeuralNetworkTest, TrainingXOR) {
    std::vector<int> layer_sizes = {2, 3, 1};
    NeuralNetwork xor_network(layer_sizes, "sigmoid", "meanSquaredError", "SGD", 42);
    
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
    };
    std::vector<std::vector<double>> targets = {
        {0.0}, {1.0}, {1.0}, {0.0}
    };
    
    EXPECT_NO_THROW({
        xor_network.train(inputs, targets, 10, 0.1);
    });
    
    std::vector<double> prediction = xor_network.predict({0.0, 0.0});
    EXPECT_EQ(prediction.size(), 1);
}

TEST_F(NeuralNetworkTest, TrainingValidation) {
    std::vector<std::vector<double>> inputs = {{0.5, -0.3}};
    std::vector<std::vector<double>> targets = {{1.0, 0.0}};
    
    EXPECT_THROW({
        network->train({}, targets, 10, 0.1);
    }, std::invalid_argument);
    
    EXPECT_THROW({
        network->train(inputs, {}, 10, 0.1);
    }, std::invalid_argument);
    
    EXPECT_THROW({
        network->train(inputs, targets, -1, 0.1); 
    }, std::invalid_argument);
    
    EXPECT_THROW({
        network->train(inputs, targets, 10, -0.1);
    }, std::invalid_argument);
}

TEST_F(NeuralNetworkTest, Evaluation) {
    std::vector<std::vector<double>> inputs = {{0.5, -0.3}, {1.0, 2.0}};
    std::vector<std::vector<double>> targets = {{1.0, 0.0}, {0.0, 1.0}};
    
    double accuracy = network->evaluate(inputs, targets);
    EXPECT_GE(accuracy, 0.0);
    EXPECT_LE(accuracy, 1.0);
}

TEST(NeuralNetworkTypes, DifferentOptimizers) {
    std::vector<int> layer_sizes = {2, 3, 1};
    
    EXPECT_NO_THROW({
        NeuralNetwork sgd_net(layer_sizes, "relu", "meanSquaredError", "SGD");
        NeuralNetwork momentum_net(layer_sizes, "relu", "meanSquaredError", "Momentum");
        NeuralNetwork adam_net(layer_sizes, "relu", "meanSquaredError", "Adam");
    });
}

TEST(NeuralNetworkTypes, ActivationCombinations) {
    std::vector<int> layer_sizes = {2, 3, 1};
    
    EXPECT_NO_THROW({
        NeuralNetwork relu_net(layer_sizes, "relu", "sigmoid", "meanSquaredError", "SGD");
        NeuralNetwork sigmoid_net(layer_sizes, "sigmoid", "sigmoid", "meanSquaredError", "SGD");
        NeuralNetwork linear_net(layer_sizes, "relu", "linear", "meanSquaredError", "SGD");
    });
}

TEST(NeuralNetworkTypes, SoftmaxCrossEntropy) {
    std::vector<int> layer_sizes = {3, 5, 3};
    
    EXPECT_NO_THROW({
        NeuralNetwork softmax_net(layer_sizes, "relu", "softmax", "crossEntropy", "Adam");
    });
}
