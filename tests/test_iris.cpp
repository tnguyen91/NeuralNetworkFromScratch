#include <gtest/gtest.h>
#include "../include/NeuralNetwork.h"
#include "../include/DataLoader.h"

TEST(IrisNeuralNetwork, LearnsIrisClassification) {
    auto dataset = DataLoader::loadIrisDataset();
    DataLoader::Dataset trainSet, testSet;
    DataLoader::trainTestSplit(dataset, trainSet, testSet, 0.2, 42);
    DataLoader::normalizeFeatures(trainSet.inputs);
    DataLoader::normalizeFeatures(testSet.inputs);
    std::vector<int> layers = {4, 8, 3};
    NeuralNetwork net(layers, "relu", "softmax", "crossEntropy", "Adam", 42);
    net.train(trainSet.inputs, trainSet.targets, 200, 0.001);
    int correct = 0;
    for (size_t i = 0; i < testSet.inputs.size(); ++i) {
        auto pred = net.predict(testSet.inputs[i]);
        int pred_class = std::distance(pred.begin(), std::max_element(pred.begin(), pred.end()));
        int true_class = std::distance(testSet.targets[i].begin(), std::max_element(testSet.targets[i].begin(), testSet.targets[i].end()));
        if (pred_class == true_class) correct++;
    }
    double accuracy = static_cast<double>(correct) / testSet.inputs.size();
    EXPECT_GT(accuracy, 0.9) << "Test accuracy too low: " << accuracy;
}
