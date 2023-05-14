#include <iostream>
#include <fstream>

#include "engine.h"
#include "NvOnnxParser.h"

using namespace nvinfer1;

static size_t getDataSize(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
// #if IS_TRT_VERSION_GE(7000)
//     case nvinfer1::DataType::kBOOL:
//       return 1;
// #endif
    default:
      throw std::runtime_error("unsupported data type.");
  }
}


void Logger::log(Severity severity, const char *msg) noexcept {
    // Would advise using a proper logging utility such as https://github.com/gabime/spdlog
    // For the sake of this tutorial, will just log to the console.

    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

bool Engine::doesFileExist(const std::string &filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

Engine::Engine(const Options &options)
    : m_options(options) {
    if (!m_options.useDynamicShape) {
        std::cout << "Model does not support dynamic batch size, using optBatchSize and maxBatchSize of 1" << std::endl;
        m_options.optBatchSize = 1;
        m_options.maxBatchSize = 1;
    }
}

bool Engine::build(std::string onnxModelPath) {
    // Only regenerate the engine file if it has not already been generated for the specified options
    m_engineName = serializeEngineOptions(m_options);
    std::cout << "Searching for engine file with name: " << m_engineName << std::endl;

    if (doesFileExist(m_engineName)) {
        std::cout << "Engine found, not regenerating..." << std::endl;
        return true;
    }

    if (!doesFileExist(onnxModelPath)) {
        throw std::runtime_error("Could not find model at path: " + onnxModelPath);
    }

    // Was not able to find the engine file, generate...
    std::cout << "Engine not found, generating..." << std::endl;

    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Define an explicit batch size and then create the network.
    // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    // We are going to first read the onnx file into memory, then pass that buffer to the parser.
    // Had our onnx model file been encrypted, this approach would allow us to first decrypt the buffer.

    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // Register a single optimization profile
    IOptimizationProfile *optProfile = builder->createOptimizationProfile();
    const int32_t numInputs = network->getNbInputs();
    for (int32_t i = 0; i < numInputs; ++i) {
        const auto input = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        int32_t inputC = inputDims.d[1];
        int32_t inputH = inputDims.d[2];
        int32_t inputW = inputDims.d[3];

        // Specify the optimization profile
        optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(m_options.optBatchSize, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
    }
    config->addOptimizationProfile(optProfile);

    // config->setMaxWorkspaceSize(m_options.maxWorkspaceSize);
    config->setMemoryPoolLimit(
      nvinfer1::MemoryPoolType::kWORKSPACE, m_options.maxWorkspaceSize);

    if (m_options.precision == Precision::FP16) {
        bool support_fp16 = builder->platformHasFastFp16();
        if (support_fp16) {
            config->setFlag(BuilderFlag::kFP16);
        }
    } else if (m_options.precision == Precision::INT8) {
        bool support_int8 = builder->platformHasFastInt8();
        if (support_int8) {
            config->setFlag(BuilderFlag::kINT8);
        }
    }

    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;
    checkCudaErrorCode(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);

    // Build the engine
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    // Write the engine to disk
    std::ofstream outfile(m_engineName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    std::cout << "Success, saved engine to " << m_engineName << std::endl;

    checkCudaErrorCode(cudaStreamDestroy(profileStream));
    return true;
}

Engine::~Engine() {
    // Free the GPU memory
    for (auto & buffer : m_buffers) {
        checkCudaErrorCode(cudaFree(buffer));
    }

    m_buffers.clear();
}

bool Engine::load() {
    // Read the serialized model from disk
    std::ifstream file(m_engineName, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    std::unique_ptr<IRuntime> runtime{createInferRuntime(m_logger)};
    if (!runtime) {
        return false;
    }

    // Set the device index
    auto ret = cudaSetDevice(m_options.deviceIndex);
    if (ret != 0) {
        int numGPUs{0};
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) +
                ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    // Storage for holding the input and output buffers
    // This will be passed to TensorRT for inference
    m_buffers.resize(m_engine->getNbBindings());

    return true;
}

void Engine::checkCudaErrorCode(cudaError_t code) {
    if (code != 0) {
        std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) + "), with message: " + cudaGetErrorString(code);
        std::cout << errMsg << std::endl;
        throw std::runtime_error(errMsg);
    }
}

bool Engine::predict(const std::vector<torch::Tensor> &inputs, std::vector<torch::Tensor>& outputs
                          ) {
    // First we do some error checking
    if (inputs.empty() || inputs[0].numel() == 0) {
        std::cout << "Provided input vector is empty!" << std::endl;
        return false;
    }

    // Create the cuda stream that will be used for inference
    cudaStream_t stream;
    checkCudaErrorCode(cudaStreamCreate(&stream));

    // Allocate GPU memory for input and output buffers
    int num_input = 0;
    m_outputLengthsFloat.clear();
    for (int i = 0; i < m_engine->getNbBindings(); ++i) {
        if (m_engine->bindingIsInput(i)) {
            auto inputBindingDims = m_engine->getBindingDimensions(i);
            auto data_type = m_engine->getBindingDataType(i);

            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], inputs[i].nbytes(), stream));
            ++ num_input;

            // m_engine->setBindingDimensions()
            if (m_engine->isShapeBinding(i)) {
                // m_context->->setInputShapeBinding(i, shape_v.data());
            }
        } else {
            // The binding is an output
            uint32_t outputLenFloat = 1;
            auto outputDims = m_engine->getBindingDimensions(i);
            auto data_type = m_engine->getBindingDataType(i);

            for (int j = 0; j < outputDims.nbDims; ++j) {
                // We ignore j = 0 because that is the batch size, and we will take that into account when sizing the buffer
                outputLenFloat *= outputDims.d[j];
            }

            m_outputLengthsFloat.push_back(outputLenFloat);
            // Now size the output buffer appropriately, taking into account the max possible batch size (although we could actually end up using less memory)
            // checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputLenFloat * m_options.maxBatchSize * sizeof(float), stream));
            checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputLenFloat * getDataSize(data_type), stream));
        }
    }

    if (inputs.size() != num_input) {
        std::cout << "Incorrect number of inputs provided!" << std::endl;
        return false;
    }

    // Ensure the batch size param was set correctly
    if (!m_options.useDynamicShape) {
        if (inputs[0].numel() > 1) {
            std::cout << "Model does not support running batch inference!" << std::endl;
            std::cout << "Please only provide a single input" << std::endl;
            return false;
        }
    }

    // Preprocess all the inputs
    for (size_t i = 0; i < num_input; ++i) {
        const auto& input = inputs[i];

        nvinfer1::Dims4 inputDims;
        inputDims.nbDims = input.dim();
        for (int j = 0; j < input.dim(); ++ j) {
            inputDims.d[j] = input.size(j);
        }
        m_context->setBindingDimensions(i, inputDims); // Define the batch size

        checkCudaErrorCode(cudaMemcpyAsync(m_buffers[i], input.data_ptr(),
                                           input.nbytes(),
                                           cudaMemcpyDeviceToDevice, stream));
    }

    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified()) {
        throw std::runtime_error("Error, not all required dimensions specified.");
    }

    // Run inference.
    bool status = m_context->enqueueV2(m_buffers.data(), stream, nullptr);
    if (!status) {
        return false;
    }

    // Copy the outputs back to CPU
    outputs.clear();

    for (int32_t outputBinding = num_input; outputBinding < m_engine->getNbBindings(); ++outputBinding) {
        auto data_type = m_engine->getBindingDataType(outputBinding);
        checkCudaErrorCode(cudaMemcpyAsync(outputs[outputBinding - num_input].data_ptr(), static_cast<char*>(m_buffers[outputBinding]), m_outputLengthsFloat[outputBinding - num_input] * getDataSize(data_type), cudaMemcpyDeviceToHost, stream));
    }

    // Synchronize the cuda stream
    checkCudaErrorCode(cudaStreamSynchronize(stream));
    checkCudaErrorCode(cudaStreamDestroy(stream));
    return true;
}

std::string Engine::serializeEngineOptions(const Options &options) {
    std::string engineName = "trt.engine";

    // Add the GPU device name to the file to ensure that the model is only used on devices with the exact same GPU
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);

    if (static_cast<size_t>(options.deviceIndex) >= deviceNames.size()) {
        throw std::runtime_error("Error, provided device index is out of range!");
    }

    auto deviceName = deviceNames[options.deviceIndex];
    // Remove spaces from the device name
    deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());

    engineName+= "." + deviceName;

    // Serialize the specified options into the filename
    if (options.precision == Precision::FP16) {
        engineName += ".fp16";
    } else if (options.precision == Precision::INT8) {
        engineName += ".int8";
    } else {
        engineName += ".fp32";
    }

    engineName += "." + std::to_string(options.maxBatchSize);
    engineName += "." + std::to_string(options.optBatchSize);
    engineName += "." + std::to_string(options.maxWorkspaceSize);

    return engineName;
}

void Engine::getDeviceNames(std::vector<std::string>& deviceNames) {
    int numGPUs{0};
    cudaGetDeviceCount(&numGPUs);

    for (int device=0; device<numGPUs; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        deviceNames.push_back(std::string(prop.name));
    }
}