#include <iostream>
#include <cassert>
#include <memory>
#include "../include/ActivationFunctions.h"
#include "../include/LossFunction.h"
#include "../include/Layer.h"
#include "../include/NeuralNetwork.h"
#include "../include/DataLoader.h"

#define SEED 1234

void testActivationFunctions() {
    std::cout << "Testing sigmoid function..." << std::endl;
    double input = 0.0;
    double sigmoidOutput = ActivationFunctions::sigmoid(input);
    double sigmoidDerivativeOutput = ActivationFunctions::sigmoidDerivativeFromInput(input);

    std::cout << "sigmoid(" << input << ") = " << sigmoidOutput << std::endl;
    std::cout << "sigmoidDerivative(" << input << ") = " << sigmoidDerivativeOutput << std::endl;
    assert(std::abs(sigmoidOutput - 0.5) < 1e-10);
    assert(std::abs(sigmoidDerivativeOutput - 0.25) < 1e-10);

    std::cout << "Testing ReLU function..." << std::endl;
    int positiveInput = 2;
    int negativeInput = -2;
    double reluOutputPositive = ActivationFunctions::relu(positiveInput);
    double reluOutputNegative = ActivationFunctions::relu(negativeInput);
    double reluDerivativePositive = ActivationFunctions::reluDerivative(positiveInput);
    double reluDerivativeNegative = ActivationFunctions::reluDerivative(negativeInput);

    std::cout << "ReLU positive input: " << positiveInput << " -> Output: " << reluOutputPositive << std::endl;
    std::cout << "ReLU negative input: " << negativeInput << " -> Output: " << reluOutputNegative << std::endl;
    std::cout << "ReLU positive derivative: " << reluDerivativePositive << std::endl;
    std::cout << "ReLU negative derivative: " << reluDerivativeNegative << std::endl;
    assert(reluOutputPositive == 2.0); 
    assert(reluOutputNegative == 0.0); 
    assert(reluDerivativePositive == 1.0);
    assert(reluDerivativeNegative == 0.0);

    std::cout << "All activation function tests passed!\n" << std::endl;
}

void testLossFunctions() {
    std::cout << "Testing MSE loss function..." << std::endl;
    std::vector<double> predicted = {0.8, 0.3, 0.9};
    std::vector<double> actual = {1.0, 0.0, 0.5};

    double mse = LossFunction::meanSquaredError(predicted, actual);
    std::cout << "Mean Squared Error: " << mse << std::endl;
    assert(std::abs(mse - 0.0967) < 1e-4);

    auto mseDerivative = LossFunction::meanSquaredErrorDerivative(predicted, actual);
    std::cout << "Mean Squared Error Derivative: ";
    for (const auto& val : mseDerivative) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    assert(std::abs(mseDerivative[0] - (-0.1333)) < 1e-4);
    assert(std::abs(mseDerivative[1] - 0.2) < 1e-4);
    assert(std::abs(mseDerivative[2] - 0.2667) < 1e-4);

    std::cout << "Testing Cross Entropy loss function..." << std::endl;
    double crossEntropy = LossFunction::crossEntropy(predicted, actual);
    std::cout << "Cross Entropy: " << crossEntropy << std::endl;
    assert(std::abs(crossEntropy - 0.0919) < 1e-4);

    auto crossEntropyDerivative = LossFunction::crossEntropyDerivative(predicted, actual);
    std::cout << "Cross Entropy Derivative: ";
    for (const auto& val : crossEntropyDerivative) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    assert(std::abs(crossEntropyDerivative[0] - (-0.2)) < 1e-5);
    assert(std::abs(crossEntropyDerivative[1] - 0.3) < 1e-5);
    assert(std::abs(crossEntropyDerivative[2] - 0.4) < 1e-5);

    std::cout << "All loss function tests passed!\n" << std::endl;
}

void testLayer() {
    std::cout << "Testing forward..." << std::endl;
    Layer layer(2, 3);
    std::cout<< "Layer dimensions: " << layer.getInputSize() << " -> " << layer.getOutputSize() << std::endl;
    assert(layer.getInputSize() == 2);
    assert(layer.getOutputSize() == 3);


    std::vector<double> input = {0.5, -0.3};
    std::vector<double> output = layer.forward(input);
    
    std::cout << "Layer input: ";
    for (const auto& val : input) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "Layer output: ";
    for (const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    for (const auto& val: output) {
        assert(val >= 0.0 && val <= 1.0);
    }

    std::cout << "Testing backward..." << std::endl;
    std::vector<double> gradients = {0.1, -0.05, 0.2};
    std::vector<double> inputGradients = layer.backward(gradients);
    std::cout << "Gradients: ";
    for (const auto& val : gradients) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Input gradients: ";
    for (const auto& val : inputGradients) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    for (const auto& val: inputGradients) {
        assert(std::isfinite(val));
    }

    std::cout << "Layer test passed!\n" << std::endl;
}

void testXOR() {
    std::cout << "Testing XOR Neural Network..." << std::endl;

    std::vector<int> layerSizes = {2, 8, 1};
    NeuralNetwork nn(layerSizes, "sigmoid", "crossEntropy", "SGD", SEED);
    
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
    };
    
    std::vector<std::vector<double>> targets = {
        {0.0}, {1.0}, {1.0}, {0.0}
    };

    std::cout << "Evaluating before training..." << std::endl;
    double initial_accuracy = nn.evaluate(inputs, targets);
    std::cout << "Initial accuracy: " << initial_accuracy * 100 << "%" << std::endl;

    std::cout << "Training the Neural Network..." << std::endl;
    nn.train(inputs, targets, 1500, 0.5);
    
    std::cout << "Evaluating after training..." << std::endl;
    double final_accuracy = nn.evaluate(inputs, targets);
    std::cout << "Final accuracy: " << final_accuracy * 100 << "%" << std::endl;

    assert(final_accuracy >= 0.9);
    assert(nn.predict({0.0, 0.0})[0] < 0.5);
    assert(nn.predict({0.0, 1.0})[0] >= 0.5);
    assert(nn.predict({1.0, 0.0})[0] >= 0.5);
    assert(nn.predict({1.0, 1.0})[0] < 0.5);
    std::cout << "XOR Neural Network test passed!\n" << std::endl;
}

void testIrisDataset() {
    std::cout << "Testing Iris Dataset Classification..." << std::endl;
    
    auto dataset = DataLoader::loadIrisDataset();
    DataLoader::normalizeFeatures(dataset.inputs);
    
    DataLoader::Dataset trainSet, validateSet, testSet;

    DataLoader::trainValidationTestSplit(dataset, trainSet, validateSet, testSet, 0.6, 0.2, 0.2, SEED);

    std::cout << "Data split: " << trainSet.inputs.size() << " train, "
              << validateSet.inputs.size() << " validation, "
              << testSet.inputs.size() << " test samples" << std::endl;
    
    std::cout << "\nClass distribution across sets:" << std::endl;
    
    auto analyzeDistribution = [&](const DataLoader::Dataset& set, const std::string& setName) {
        std::vector<int> classCount(dataset.classNames.size(), 0);
        for (const auto& target : set.targets) {
            for (size_t i = 0; i < target.size(); ++i) {
                if (target[i] == 1.0) {
                    classCount[i]++;
                    break;
                }
            }
        }
        
        std::cout << setName << " set:" << std::endl;
        for (size_t i = 0; i < dataset.classNames.size(); ++i) {
            std::cout << "  " << dataset.classNames[i] << ": " << classCount[i] << " samples" << std::endl;
        }
    };
    
    analyzeDistribution(trainSet, "Train");
    analyzeDistribution(validateSet, "Validation"); 
    analyzeDistribution(testSet, "Test");
    
    std::vector<int> layerSizes = {4, 16, 8, 3};
    NeuralNetwork nn(layerSizes, "sigmoid", "crossEntropy", "Adam", SEED);
    
    std::cout << "Training on Iris dataset..." << std::endl;
    
    nn.train(trainSet.inputs, trainSet.targets, 100, 0.01);
    
    std::cout << "Evaluating on validation set..." << std::endl;
    double validation_accuracy = nn.evaluate(validateSet.inputs, validateSet.targets);
    std::cout << "Validation accuracy: " << validation_accuracy * 100 << "%" << std::endl;
    assert(validation_accuracy > 0.8);

    std::cout << "Predicting on test set..." << std::endl;
    int correctPredictions = 0;
    for (size_t i = 0; i < testSet.inputs.size(); ++i) {
        std::vector<double> prediction = nn.predict(testSet.inputs[i]);
        int predictedClass = std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
        std::cout << "Input: ";
        for (const auto& val : testSet.inputs[i]) {
            std::cout << val << " ";
        }
        std::cout << " -> Predicted class: " << dataset.classNames[predictedClass] << std::endl;
        int actualClass = std::distance(testSet.targets[i].begin(), std::max_element(testSet.targets[i].begin(), testSet.targets[i].end()));
        std::cout << "Actual class: " << dataset.classNames[actualClass] << std::endl;
        if (predictedClass == actualClass) {
            correctPredictions++;
        }
        assert(predictedClass == actualClass || validation_accuracy > 0.8);
    }
    double test_accuracy = static_cast<double>(correctPredictions) / testSet.inputs.size();
    std::cout << "Test accuracy: " << test_accuracy * 100 << "%" << std::endl;
    assert(test_accuracy > 0.8);
    std::cout << "Iris dataset classification test passed!\n" << std::endl;
}

int main() {
    std::cout << "Running tests..." << std::endl;

    testActivationFunctions();
    testLossFunctions();
    testLayer();
    testXOR();
    testIrisDataset();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}