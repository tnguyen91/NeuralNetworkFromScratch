#include <gtest/gtest.h>
#include "../include/NeuralNetwork.h"
#include "../include/DataLoader.h"
#include <cstdio>

TEST(NeuralNetworkSaveLoad, SaveAndLoadModelPreservesWeights) {
    std::vector<int> layers = {2, 4, 1};
    NeuralNetwork net(layers, "relu", "sigmoid", "meanSquaredError", "Adam", 42);
    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<std::vector<double>> targets = {
        {0}, {1}, {1}, {0}
    };
    net.train(inputs, targets, 10, 0.01); 

    const std::string filename = "test_model.bin";
    net.save(filename);

    NeuralNetwork loaded;
    loaded.load(filename);

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto pred1 = net.predict(inputs[i]);
        auto pred2 = loaded.predict(inputs[i]);
        ASSERT_EQ(pred1.size(), pred2.size());
        for (size_t j = 0; j < pred1.size(); ++j) {
            EXPECT_NEAR(pred1[j], pred2[j], 1e-9) << "Mismatch at input " << i << ", output " << j;
        }
    }

    std::remove(filename.c_str());
}
