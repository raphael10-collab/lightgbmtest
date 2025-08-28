#include <LightGBM/config.h>
#include <LightGBM/dataset_loader.h>
#include <LightGBM/boosting.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>
#include <LightGBM/utils/common.h>

#include <iostream>
#include <random>
#include <algorithm>

int main(int argc, char **argv)
{
    /* create example dataset */
    std::random_device rd;
    std::mt19937 gen(rd());

    // one random generator for every class
    std::vector<std::normal_distribution<>> dists = {
        std::normal_distribution<>(0, 1),
        std::normal_distribution<>(10, 1)};

    /* create raw data */
    const int numSamples = 5000;
    const int numFeats = 2;
    const int numClasses = static_cast<int>(dists.size());

    std::cout << "Num classes: " << numClasses << std::endl;

    // labels
    std::vector<float>   labels(numSamples);
    for (int i=0; i < numSamples; i++)
        labels[i] = i % numClasses;

    std::vector< std::vector<double> > features(numSamples);
    for (int i=0; i < numSamples; i++)
    {
        features[i].resize(numFeats);
        for (int j=0; j < numFeats; j++)
        {
            const auto lbl = static_cast<int>(labels[i]);
            features[i][j] = dists[lbl](gen);
        }
    }

    // prepare sample data
    std::vector< std::vector<double> > sampleData(numFeats);
    for (int i=0; i < numSamples; i++)
    {
        for (int j=0; j < numFeats; j++)
            sampleData[j].push_back(features[i][j]);
    }

    return EXIT_SUCCESS;
}
