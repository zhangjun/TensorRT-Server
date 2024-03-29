#include "engine.h"
#include <opencv2/opencv.hpp>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

int main() {
    // Specify our GPU inference configuration options
    Options options;
    // TODO: If your model only supports a static batch size
    options.useDynamicShape = false;
    options.precision = Precision::FP16; // Use fp16 precision for faster inference.

    if (options.doesSupportDynamicBatchSize) {
        options.optBatchSize = 4;
        options.maxBatchSize = 16;
    } else {
        options.optBatchSize = 1;
        options.maxBatchSize = 1;
    }

    Engine engine(options);

    // TODO: Specify your model here.
    // Must specify a dynamic batch size when exporting the model from onnx.
    // If model only specifies a static batch size, must set the above variable doesSupportDynamicBatchSize to false.
    const std::string onnxModelpath = "../models/arcfaceresnet100-8.onnx";

    bool succ = engine.build(onnxModelpath);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    succ = engine.load();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    // Let's use a batch size which matches that which we set the Options.optBatchSize option
    size_t batchSize = options.optBatchSize;

    std::array<float, 3> subVals {0.5f, 0.5f, 0.5f};
    std::array<float, 3> divVals {0.5f, 0.5f, 0.5f};
    bool normalize = true;

    // Discard the first inference time as it takes longer
    std::vector<std::vector<std::vector<float>>> featureVectors;
    succ = engine.runInference(inputs, featureVectors, subVals, divVals, normalize);
    if (!succ) {
        throw std::runtime_error("Unable to run inference.");
    }

    size_t numIterations = 100;

    // Benchmark the inference time
    auto t1 = Clock::now();
    for (size_t i = 0; i < numIterations; ++i) {
        featureVectors.clear();
        engine.runInference(inputs, featureVectors, subVals, divVals);
    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Success! Average time per inference: " << totalTime / numIterations / static_cast<float>(inputs[0].size()) <<
    " ms, for batch size of: " << inputs[0].size() << std::endl;

    // Print the feature vectors
    for (size_t batch = 0; batch < featureVectors.size(); ++batch) {
        for (size_t outputNum = 0; outputNum < featureVectors[batch].size(); ++outputNum) {
            std::cout << "Batch " << batch << ", " << "output " << outputNum << std::endl;
            int i = 0;
            for (const auto &e:  featureVectors[batch][outputNum]) {
                std::cout << e << " ";
                if (++i == 10) {
                    std::cout << "...";
                    break;
                }
            }
            std::cout << "\n" << std::endl;
        }
    }

    // TODO: If your model requires post processing (ex. convert feature vector into bounding boxes) then you would do so here.

    return 0;
}
