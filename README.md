# Neural Network From Scratch

A C++ implementation of a feedforward neural network built from first principles, demonstrating core machine learning concepts without external ML libraries.

## Features

- **Modular Design**: Separate components for layers, optimizers, activation functions, and loss functions
- **Multiple Optimizers**: SGD, Momentum, and Adam
- **Activation Functions**: ReLU, Sigmoid, Softmax, Linear  
- **Loss Functions**: MSE and Cross-Entropy with numerical stability
- **Comprehensive Tests**: Using GTests covering all components
- **Real Data**: Iris dataset with preprocessing utilities

## Quick Start

```bash
# Build
mkdir build && cd build
cmake .. && make

# Run tests
ctest
```

## Usage

```cpp
#include "NeuralNetwork.h"

// Create network: 4 inputs -> 8 hidden -> 3 outputs
std::vector<int> layers = {4, 8, 3};
NeuralNetwork network(layers, "relu", "softmax", "crossEntropy", "Adam");

// Train on your data
network.train(inputs, targets, 1000, 0.01);

// Make predictions
auto result = network.predict({5.1, 3.5, 1.4, 0.2});
```

## Results

- **XOR Problem**: Converges to 100% accuracy
- **Iris Classification**: 80-95% test accuracy
- **All Tests**: 100% pass rate

## Project Structure

```
include/          # Header files
src/              # Implementation files  
tests/            # GTest test suites
build/            # Build artifacts
```
