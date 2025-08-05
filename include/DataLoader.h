#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>

class DataLoader {
public:
    struct Dataset {
        std::vector<std::vector<double>> inputs;
        std::vector<std::vector<double>> targets;
        std::vector<std::string> featureNames;
        std::vector<std::string> classNames;
    };

    static Dataset loadIrisDataset();
    
    static bool downloadIrisDataset(const std::string& filename);
    
    static Dataset loadIrisFromCSV(const std::string& filename);
    
    static void normalizeFeatures(std::vector<std::vector<double>>& data);
    
    static void trainTestSplit(const Dataset& dataset, 
                              Dataset& trainSet, 
                              Dataset& testSet, 
                              double testRatio);
    
    static void trainValidationTestSplit(const Dataset& dataset,
                                       Dataset& trainSet,
                                       Dataset& validationSet,
                                       Dataset& testSet,
                                       double trainRatio,
                                       double validationRatio,
                                       double testRatio);
    
    static std::vector<std::vector<double>> oneHotEncode(const std::vector<int>& labels, int numClasses);

private:
    static std::vector<double> computeMean(const std::vector<std::vector<double>>& data);
    static std::vector<double> computeStd(const std::vector<std::vector<double>>& data, 
                                         const std::vector<double>& mean);
};

#endif
