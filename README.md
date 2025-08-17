
# Neural Network From Scratch

A modern C++17 implementation of a feedforward neural network, built from first principles. This project demonstrates core machine learning concepts, modular C++ design, and practical ML workflows—without relying on external ML libraries.

## Features

- **Modular Architecture**: Clean separation of layers, activation functions, loss functions, and optimizers
- **Optimizers**: SGD, Momentum, Adam
- **Activation Functions**: ReLU, Sigmoid, Softmax, Linear
- **Loss Functions**: Mean Squared Error, Cross-Entropy
- **Data Utilities**: Built-in Iris dataset loader, CSV parsing, normalization, one-hot encoding, and train/test/validation split
- **Comprehensive Unit Tests**: GTest-based tests for all major components 
- **Example Problems**: Solves XOR and Iris classification with high accuracy

## Quick Start

```bash
# Build (from project root)
mkdir -p build && cd build
cmake .. && make

# Run all tests
ctest --output-on-failure
```

## Usage Example

```cpp
#include "NeuralNetwork.h"
#include "DataLoader.h"

// Load and preprocess Iris data
auto dataset = DataLoader::loadIrisDataset();
DataLoader::Dataset trainSet, testSet;
DataLoader::trainTestSplit(dataset, trainSet, testSet, 0.2, 42);
DataLoader::normalizeFeatures(trainSet.inputs);
DataLoader::normalizeFeatures(testSet.inputs);

// Define network: 4 inputs → 10 hidden → 3 outputs
std::vector<int> layers = {4, 10, 3};
NeuralNetwork net(layers, "relu", "softmax", "crossEntropy", "Adam", 42);
net.train(trainSet.inputs, trainSet.targets, 200, 0.001);

// Evaluate accuracy
int correct = 0;
for (size_t i = 0; i < testSet.inputs.size(); ++i) {
	auto pred = net.predict(testSet.inputs[i]);
	int pred_class = std::distance(pred.begin(), std::max_element(pred.begin(), pred.end()));
	int true_class = std::distance(testSet.targets[i].begin(), std::max_element(testSet.targets[i].begin(), testSet.targets[i].end()));
	if (pred_class == true_class) correct++;
}
double accuracy = static_cast<double>(correct) / testSet.inputs.size();
std::cout << "Test accuracy: " << accuracy << std::endl;
```

## Results

- **XOR Problem**: Achieves 100% accuracy (see `tests/test_xor.cpp`)
- **Iris Classification**: Typically 90%+ test accuracy (see `tests/test_iris.cpp`)
- **All Unit Tests**: 100% pass rate 

## Directory Structure

```
include/    # C++ header files
src/        # C++ implementation files
tests/      # GTest unit and integration tests
build/      # Build artifacts 
```

## Requirements

- C++17 compiler
- CMake >= 3.10
- GTest (for running tests)

## License

MIT License
