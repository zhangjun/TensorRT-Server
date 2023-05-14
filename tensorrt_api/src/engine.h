#pragma once
#include <vector>
#include <memory>

#include "NvInfer.h"
#include "torch/script.h"

// Precision used for GPU inference
enum class Precision {
    FP32,
    FP16,
    INT8
};

// Options for the network
struct Options {
    bool useDynamicShape = true;
    // Precision to use for GPU inference. 16 bit is faster but may reduce accuracy.
    Precision precision = Precision::FP16;
    // The batch size which should be optimized for.
    int32_t optBatchSize = 1;
    // Maximum allowable batch size
    int32_t maxBatchSize = 16;
    // Max allowable GPU memory to be used for model conversion, in bytes.
    // Applications should allow the engine builder as much workspace as they can afford;
    // at runtime, the SDK allocates no more than this and typically less.
    size_t maxWorkspaceSize = 4000000000;
    // GPU device index
    int deviceIndex = 0;
};

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};

class Engine {
public:
    Engine(const Options& options);
    ~Engine();
    // Build the network
    bool build(std::string onnxModelPath);
    // Load and prepare the network for inference
    bool load();
    // Run inference.
    // Input format [input][batch][image]
    // Output format [batch][output][feature_vector]
    bool predict(const std::vector<torch::Tensor> &inputs, std::vector<torch::Tensor>& outputs);

private:
    // Converts the engine options into a string
    std::string serializeEngineOptions(const Options& options);

    void getDeviceNames(std::vector<std::string>& deviceNames);

    bool doesFileExist(const std::string& filepath);

    // Holds pointers to the input and output GPU buffers
    std::vector<void*> m_buffers;
    std::vector<uint32_t> m_outputLengthsFloat{};
    std::vector<nvinfer1::Dims3> m_inputDims;

    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    Options m_options;
    Logger m_logger;
    std::string m_engineName;

    inline void checkCudaErrorCode(cudaError_t code);
};
