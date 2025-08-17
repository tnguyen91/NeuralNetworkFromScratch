#include <gtest/gtest.h>
#include "../include/DenseLayer.h"
#include "../include/ActivationFunctions.h"
#include <vector>

class LayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<DenseLayer>(
            2, 3,
            [](double x) { return ActivationFunctions::relu(x); },
            [](double x) { return ActivationFunctions::reluDerivative(x); },
            "relu",
            42
        );
    }
    std::unique_ptr<DenseLayer> layer;
};

TEST_F(LayerTest, Constructor) {
    EXPECT_EQ(layer->getWeights().size(), 2);
    EXPECT_EQ(layer->getWeights()[0].size(), 3);
    EXPECT_EQ(layer->getBiases().size(), 3);
}

TEST_F(LayerTest, ForwardPass) {
    std::vector<double> input = {0.5, -0.3};
    std::vector<double> output = layer->forward(input, true);
    EXPECT_EQ(output.size(), 3);
    for (double val : output) {
        EXPECT_GE(val, 0.0);
    }
}

TEST_F(LayerTest, BackwardPass) {
    std::vector<double> input = {0.5, -0.3};
    layer->forward(input, true);
    std::vector<double> output_gradients = {0.1, -0.05, 0.2};
    std::vector<double> input_gradients = layer->backward(output_gradients);
    EXPECT_EQ(input_gradients.size(), 2);
}

TEST_F(LayerTest, GradientStorage) {
    std::vector<double> input = {0.5, -0.3};
    layer->forward(input, true);
    std::vector<double> output_gradients = {0.1, -0.05, 0.2};
    layer->backward(output_gradients);
    const auto& weight_grads = layer->getWeightGradients();
    const auto& bias_grads = layer->getBiasGradients();
    EXPECT_EQ(weight_grads.size(), 2);
    EXPECT_EQ(weight_grads[0].size(), 3);
    EXPECT_EQ(bias_grads.size(), 3);
}

TEST_F(LayerTest, SoftmaxLayer) {
    DenseLayer softmax_layer(2, 3, true, 42);
    std::vector<double> input = {1.0, 2.0};
    std::vector<double> output = softmax_layer.forward(input, true);
    EXPECT_EQ(output.size(), 3);
    double sum = 0.0;
    for (double val : output) {
        sum += val;
        EXPECT_GE(val, 0.0);
    }
    EXPECT_NEAR(sum, 1.0, 1e-10);
}