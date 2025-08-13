#include "../include/DataLoader.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <set>

bool DataLoader::downloadIrisDataset(const std::string& filename) {
    std::cout << "Downloading Iris dataset from UCI Machine Learning Repository..." << std::endl;
    
    std::string url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";
    
    std::string mkdirCommand = "mkdir -p data";
    
    int mkdirResult = std::system(mkdirCommand.c_str());
    if (mkdirResult != 0) {
        std::cerr << "Failed to create data directory" << std::endl;
        return false;
    }
    
    std::string command = "curl -s -L -o " + filename + " " + url;
    int downloadResult = std::system(command.c_str());
    if (downloadResult != 0) {
        std::cerr << "Failed to download dataset. Error code: " << downloadResult << std::endl;
        return false;
    }
    
    std::ifstream testFile(filename);
    if (!testFile.is_open() || testFile.peek() == std::ifstream::traits_type::eof()) {
        std::cerr << "Downloaded file is empty or corrupted" << std::endl;
        return false;
    }
    
    std::cout << "Successfully downloaded dataset to " << filename << std::endl;
    return true;
}

DataLoader::Dataset DataLoader::loadIrisDataset() {
    return loadIrisFromCSV("data/iris.csv");
}

DataLoader::Dataset DataLoader::loadIrisFromCSV(const std::string& filename) {
    Dataset dataset;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cout << "CSV file not found. Attempting to download from UCI ML Repository..." << std::endl;
        if (!downloadIrisDataset(filename)) {
            throw std::runtime_error("Could not download or find Iris dataset file: " + filename);
        }
        
        file.open(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open downloaded Iris dataset file: " + filename);
        }
    }
    
    std::set<std::string> uniqueSpecies;
    std::vector<std::string> allSpecies; 
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::vector<double> features;
        std::string species;
        
        std::stringstream ss(line);
        std::string cell;
        
        // first 4 features
        for (int i = 0; i < 4; ++i) {
            if (std::getline(ss, cell, ',')) {
                try {
                    features.push_back(std::stod(cell));
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing feature " << i << " in line: " << line << std::endl;
                    continue;
                }
            }
        }
        
        // species
        if (std::getline(ss, species, ',')) {
            species.erase(species.find_last_not_of(" \t\r\n\"'") + 1);
            species.erase(0, species.find_first_not_of(" \t\r\n\"'"));
            
            if (features.size() == 4) {
                dataset.inputs.push_back(features);
                allSpecies.push_back(species);
                uniqueSpecies.insert(species);
            }
        }
    }
    
    file.close();
    
    if (dataset.inputs.empty()) {
        throw std::runtime_error("No data loaded from file: " + filename);
    }
    
    std::vector<std::string> speciesVector(uniqueSpecies.begin(), uniqueSpecies.end());
    std::sort(speciesVector.begin(), speciesVector.end());
    
    dataset.featureNames = {"sepal_length", "sepal_width", "petal_length", "petal_width"};
    dataset.classNames = speciesVector; 
    
    dataset.targets.reserve(allSpecies.size());
    for (const auto& species : allSpecies) {
        std::vector<double> target(speciesVector.size(), 0.0);
        auto it = std::find(speciesVector.begin(), speciesVector.end(), species);
        if (it != speciesVector.end()) {
            size_t classIndex = std::distance(speciesVector.begin(), it);
            target[classIndex] = 1.0;
            dataset.targets.push_back(target);
        } else {
            std::cerr << "Unknown species during encoding: " << species << std::endl;
        }
    }
    
    std::cout << "Loaded " << dataset.inputs.size() << " samples from " << filename << std::endl;
    std::cout << "Discovered " << dataset.classNames.size() << " classes: ";
    for (size_t i = 0; i < dataset.classNames.size(); ++i) {
        std::cout << dataset.classNames[i] << (i < dataset.classNames.size() - 1 ? ", " : "");
    }
    std::cout << std::endl;
    return dataset;
}

void DataLoader::normalizeFeatures(std::vector<std::vector<double>>& data) {
    if (data.empty()) return;
    
    std::vector<double> mean = computeMean(data);
    std::vector<double> std = computeStd(data, mean);
    
    for (auto& sample : data) {
        for (size_t i = 0; i < sample.size(); ++i) {
            if (std[i] > 1e-8) { // Avoid dividing by zero
                sample[i] = (sample[i] - mean[i]) / std[i];
            }
        }
    }
}

void DataLoader::trainTestSplit(const Dataset& dataset, Dataset& trainSet, Dataset& testSet, double testRatio, unsigned int seed) {
    if (dataset.inputs.empty()) {
        throw std::invalid_argument("Dataset cannot be empty");
    }
    if (dataset.inputs.size() != dataset.targets.size()) {
        throw std::invalid_argument("Inputs and targets must have the same number of samples");
    }
    size_t totalSamples = dataset.inputs.size();
    size_t testSize = static_cast<size_t>(totalSamples * testRatio);
    size_t trainSize = totalSamples - testSize;
    
    std::vector<size_t> indices(totalSamples);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::mt19937 g;
    if (seed == 0) {
        std::random_device rd;
        g.seed(rd());
    } else {
        g.seed(seed);
    }
    std::shuffle(indices.begin(), indices.end(), g);
    
    trainSet.inputs.clear();
    trainSet.targets.clear();
    testSet.inputs.clear();
    testSet.targets.clear();
    
    trainSet.featureNames = dataset.featureNames;
    trainSet.classNames = dataset.classNames;
    testSet.featureNames = dataset.featureNames;
    testSet.classNames = dataset.classNames;

    trainSet.inputs.reserve(trainSize);
    trainSet.targets.reserve(trainSize);
    testSet.inputs.reserve(testSize);
    testSet.targets.reserve(testSize);
    
    for (size_t i = 0; i < trainSize; ++i) {
        trainSet.inputs.push_back(dataset.inputs[indices[i]]);
        trainSet.targets.push_back(dataset.targets[indices[i]]);
    }
    
    for (size_t i = trainSize; i < totalSamples; ++i) {
        testSet.inputs.push_back(dataset.inputs[indices[i]]);
        testSet.targets.push_back(dataset.targets[indices[i]]);
    }
    
    std::cout << "Train/Test split: " << trainSet.inputs.size() << " train samples, " 
              << testSet.inputs.size() << " test samples" << std::endl;
}

void DataLoader::trainValidationTestSplit(const Dataset& dataset,
                                         Dataset& trainSet,
                                         Dataset& validationSet,
                                         Dataset& testSet,
                                         double trainRatio,
                                         double validationRatio,
                                         double testRatio,
                                         unsigned int seed) {
    if (dataset.inputs.empty()) {
        throw std::invalid_argument("Dataset cannot be empty");
    }
    if (dataset.inputs.size() != dataset.targets.size()) {
        throw std::invalid_argument("Number of inputs must match number of targets");
    }
    if (trainRatio < 0 || validationRatio < 0 || testRatio < 0) {
        throw std::invalid_argument("All ratios must be non-negative");
    }
    if (std::abs(trainRatio + validationRatio + testRatio - 1.0) > 1e-6) {
        throw std::invalid_argument("Train, validation, and test ratios must sum to 1.0");
    }
    
    size_t totalSamples = dataset.inputs.size();
    size_t trainSize = static_cast<size_t>(totalSamples * trainRatio);
    size_t validationSize = static_cast<size_t>(totalSamples * validationRatio);
    size_t testSize = totalSamples - trainSize - validationSize;
    
    std::vector<size_t> indices(totalSamples);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::mt19937 g;
    if (seed == 0) {
        std::random_device rd;
        g.seed(rd());
    } else {
        g.seed(seed);
    }
    std::shuffle(indices.begin(), indices.end(), g);
    
    trainSet.inputs.clear();
    trainSet.targets.clear();
    validationSet.inputs.clear();
    validationSet.targets.clear();
    testSet.inputs.clear();
    testSet.targets.clear();
    
    trainSet.featureNames = dataset.featureNames;
    trainSet.classNames = dataset.classNames;
    validationSet.featureNames = dataset.featureNames;
    validationSet.classNames = dataset.classNames;
    testSet.featureNames = dataset.featureNames;
    testSet.classNames = dataset.classNames;
    
    trainSet.inputs.reserve(trainSize);
    trainSet.targets.reserve(trainSize);
    validationSet.inputs.reserve(validationSize);
    validationSet.targets.reserve(validationSize);
    testSet.inputs.reserve(testSize);
    testSet.targets.reserve(testSize);

    size_t currentIndex = 0;
    
    for (size_t i = 0; i < trainSize; ++i) {
        trainSet.inputs.push_back(dataset.inputs[indices[currentIndex]]);
        trainSet.targets.push_back(dataset.targets[indices[currentIndex]]);
        currentIndex++;
    }
    
    for (size_t i = 0; i < validationSize; ++i) {
        validationSet.inputs.push_back(dataset.inputs[indices[currentIndex]]);
        validationSet.targets.push_back(dataset.targets[indices[currentIndex]]);
        currentIndex++;
    }
    
    for (size_t i = 0; i < testSize; ++i) {
        testSet.inputs.push_back(dataset.inputs[indices[currentIndex]]);
        testSet.targets.push_back(dataset.targets[indices[currentIndex]]);
        currentIndex++;
    }
    
    std::cout << "Train/Validation/Test split: " << trainSet.inputs.size() << " train, " 
              << validationSet.inputs.size() << " validation, " 
              << testSet.inputs.size() << " test samples" << std::endl;
}

// for int labels
std::vector<std::vector<double>> DataLoader::oneHotEncode(const std::vector<int>& labels, int numClasses) {
    std::vector<std::vector<double>> encoded(labels.size(), std::vector<double>(numClasses, 0.0));
    
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] >= 0 && labels[i] < numClasses) {
            encoded[i][labels[i]] = 1.0;
        }
    }
    
    return encoded;
}

std::vector<double> DataLoader::computeMean(const std::vector<std::vector<double>>& data) {
    if (data.empty()) return {};
    
    std::vector<double> mean(data[0].size(), 0.0);
    
    for (const auto& sample : data) {
        for (size_t i = 0; i < sample.size(); ++i) {
            mean[i] += sample[i];
        }
    }
    
    for (double& m : mean) {
        m /= data.size();
    }
    
    return mean;
}

std::vector<double> DataLoader::computeStd(const std::vector<std::vector<double>>& data, 
                                           const std::vector<double>& mean) {
    if (data.empty()) return {};
    
    std::vector<double> variance(mean.size(), 0.0);
    
    for (const auto& sample : data) {
        for (size_t i = 0; i < sample.size(); ++i) {
            double diff = sample[i] - mean[i];
            variance[i] += diff * diff;
        }
    }
    
    std::vector<double> std(variance.size());
    for (size_t i = 0; i < variance.size(); ++i) {
        variance[i] /= data.size();
        std[i] = std::sqrt(variance[i]);
    }
    
    return std;
}
