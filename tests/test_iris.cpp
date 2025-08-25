#include <gtest/gtest.h>
#include <algorithm>
#include "../include/NeuralNetwork.h"
#include "../include/DataLoader.h"
#include "../include/EarlyStopping.h"

TEST(IrisNeuralNetwork, LearnsIrisClassification) {
    auto dataset = DataLoader::loadIrisDataset();
    DataLoader::Dataset trainSet, valSet, testSet;
    DataLoader::trainValidationTestSplit(dataset, trainSet, valSet, testSet, 0.6, 0.2, 0.2, 42);
    DataLoader::normalizeFeatures(trainSet.inputs);
    DataLoader::normalizeFeatures(valSet.inputs);
    DataLoader::normalizeFeatures(testSet.inputs);
    std::vector<int> layers = {4, 15, 3};
    NeuralNetwork net(layers, "relu", "softmax", "crossEntropy", "SGD", 42);
    EarlyStopping early_stopping(5);
    net.train(trainSet.inputs, trainSet.targets, valSet.inputs, valSet.targets, 500, 0.01, early_stopping);
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
