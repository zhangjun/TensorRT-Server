#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "trt_engine.h"

using namespace nvonnxparser;
using namespace nvinfer1;

static Profiler profiler;

bool TRTEngine::Init() {
  initLibNvInferPlugins(&gLogger, "");
  // profile_ = runtime_->GetBuilder()->createOptimizationProfile();
  builder_.reset(createInferBuilder(gLogger.getTRTLogger()));
  builder_config_.reset(builder_->createBuilderConfig());

  //   uint32_t flags = 1U << static_cast<int>(
  //             nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  nvinfer1::NetworkDefinitionCreationFlags flags =
      (1U << static_cast<int>(
           nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  // flags |= (1U << static_cast<int>(
  //               nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_PRECISION));
  network_.reset(builder_->createNetworkV2(flags));

  cudaStreamCreate(&stream_);
}

bool TRTEngine::Save(const std::string &model_file,
                     const std::string &engine_file) {

  builder_config_->setMaxWorkspaceSize(max_workspace_size_);
  builder_->setMaxBatchSize(max_batch_size_);
  // builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
  // builder_config_->setInt8Calibrator(calibrator_);
  // builder_config_->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
  // builder_config_->setDLACore(dla_core_);
  // builder_config_->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
  bool fp16 = builder_->platformHasFastFp16();
  bool int8 = builder_->platformHasFastInt8();

  if (fp16) {
    builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  if (int8) {
    builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    builder_config_->setFlag(nvinfer1::BuilderFlag::kINT8);
  }
  auto parser = TrtUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network(), gLogger.getTRTLogger()));
  if (!parser->parseFromFile(
          model_file.c_str(),
          static_cast<int>(gLogger.getReportableSeverity()))) {
    std::cout << "Failed to parse onnx file." << std::endl;
    parser.reset();
    return false;
  }
  const std::string trt_network_name = "trt_engine";
  network()->setName(trt_network_name.c_str());
  serialized_engine_.reset(
      builder_->buildSerializedNetwork(*network(), *builder_config_));

  // engine_.reset(builder_->buildEngineWithConfig(*network(),
  // *builder_config_)); nvinfer1::IHostMemory* model = engine->serialize();
  // std::string engine_filename;
  // std::ofstream engine_file(engine_filename.c_str());
  // if (!engine_file) {
  //   return false;
  // }
  // engine_file.write((char*)serialized_engine_->data(),
  // serialized_engine_->size()); engine_file.close();

  auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(createInferRuntime(gLogger));
  if (runtime == nullptr) {
    std::cout << "Error creating TRT runtime. " << std::endl;
    return false;
  }
  engine_.reset(runtime->deserializeCudaEngine(serialized_engine_->data(),
                                               serialized_engine_->size()));
  if (engine_.get() == nullptr) {
    std::cout << "Failed to build TensorRT engine." << std::endl;
    return false;
  }
}

bool TRTEngine::Load(const std::string &engine_file) {
  std::ifstream input(engine_file, std::ios::binary);
  if (!input) {
    std::cout << "Error opening engine file: " << engine_file << std::endl;
    return false;
  }
  input.seekg(0, input.end);
  const size_t fsize = input.tellg();
  input.seekg(0, input.beg);

  std::vector<char> bytes(fsize);
  input.read(bytes.data(), fsize);

  auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(createInferRuntime(gLogger));
  if (runtime == nullptr) {
    std::cout << "Error creating TRT runtime. " << std::endl;
    return false;
  }
  int dla_core_id = 0;
  if (dla_core_id != -1) {
    auto dla_core_count = runtime->getNbDLACores();
    if (dla_core_id < dla_core_id) {
      runtime->setDLACore(dla_core_id);
    }
  }
  engine_.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
  return true;
}

void TRTEngine::Run() {
  // bindings_.resize(engine_->getNbBindings());
  std::vector<void *> buffers(engine_->getNbBindings());
  // const ICudaEngine &engine = context.getEngine();
  cudaStreamSynchronize(stream_);
  context()->enqueueV2(buffers.data(), stream_, nullptr);
  cudaStreamSynchronize(stream_);
  
  if (use_profiler_) {
    profiler.printLayerTimes();
  }
}