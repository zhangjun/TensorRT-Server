#include "trt_engine.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "NvInferPlugin.h"
#include "NvOnnxParser.h"

using namespace nvonnxparser;
using namespace nvinfer1;

static Profiler profiler;
Logger gLogger(Severity::kINFO);

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
  // std::cout << "init fisnish." << std::endl;
  return true;
}

bool TRTEngine::Save(const std::string& model_file,
                     const std::string& engine_file) {
  // builder_config_->setMaxWorkspaceSize(max_workspace_size_); TRT_DEPRECATED
  builder_config_->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,
                                      max_workspace_size_);

  builder_->setMaxBatchSize(max_batch_size_);
  // builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
  // builder_config_->setInt8Calibrator(calibrator_);
  // builder_config_->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
  // builder_config_->setDLACore(dla_core_);
  // builder_config_->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
  // config.setFlag(BuilderFlag::kDIRECT_IO);
  bool fp16 = builder_->platformHasFastFp16();
  bool int8 = builder_->platformHasFastInt8();

  if (fp16) {
    builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  if (false && int8) {
    builder_config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    builder_config_->setFlag(nvinfer1::BuilderFlag::kINT8);
  }
  builder_config_->clearFlag(BuilderFlag::kTF32);
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

  bool hasDynamicShapes{false};
  IOptimizationProfile* profile{nullptr};
  profile = builder_->createOptimizationProfile();

  // Set formats and data types of inputs
  for (int32_t i = 0; i < network()->getNbInputs(); ++i) {
    auto* input = network()->getInput(i);
    switch (input->getType()) {
      case nvinfer1::DataType::kINT32:
      case nvinfer1::DataType::kBOOL:
      case nvinfer1::DataType::kHALF:
        // Leave these as is.
        break;
      case nvinfer1::DataType::kFLOAT:
      case nvinfer1::DataType::kINT8:
        // User did not specify a floating-point format.  Default to kFLOAT.
        input->setType(nvinfer1::DataType::kFLOAT);
        break;
    }
    input->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));

    if (profile) {
      auto const dims = input->getDimensions();
      auto const isScalar = dims.nbDims == 0;
      auto const isDynamicInput =
          std::any_of(dims.d,
                      dims.d + dims.nbDims,
                      [](int32_t dim) { return dim == -1; }) ||
          input->isShapeTensor();
      if (isDynamicInput) {
        hasDynamicShapes = true;
        ShapeRange shapes{};

        // If no shape is provided, set dynamic dimensions to 1.
        constexpr int DEFAULT_DIMENSION = 1;
        std::vector<int> staticDims;
        if (input->isShapeTensor()) {
          if (isScalar) {
            staticDims.push_back(1);
          } else {
            staticDims.resize(dims.d[0]);
            std::fill(staticDims.begin(), staticDims.end(), DEFAULT_DIMENSION);
          }
        } else {
          staticDims.resize(dims.nbDims);
          std::transform(dims.d,
                         dims.d + dims.nbDims,
                         staticDims.begin(),
                         [&](int dimension) {
                           return dimension > 0 ? dimension : DEFAULT_DIMENSION;
                         });
        }
        auto vec2str = [](const std::vector<int> data) {
          std::string res;
          for (auto& item : data) {
            res += std::to_string(item) + " ";
          }
          return res;
        };
        std::cout << "Dynamic dimensions required for input: "
                  << input->getName()
                  << ", but no shapes were provided. Automatically overriding "
                     "shape to: "
                  << vec2str(staticDims) << std::endl;
        std::fill(shapes.begin(), shapes.end(), staticDims);

        std::vector<int> profileDims{};
        if (input->isShapeTensor()) {
          profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMIN)];
          profile->setShapeValues(input->getName(),
                                  OptProfileSelector::kMIN,
                                  profileDims.data(),
                                  static_cast<int>(profileDims.size()));
          profileDims = shapes[static_cast<size_t>(OptProfileSelector::kOPT)];
          profile->setShapeValues(input->getName(),
                                  OptProfileSelector::kOPT,
                                  profileDims.data(),
                                  static_cast<int>(profileDims.size()));
          profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMAX)];
          profile->setShapeValues(input->getName(),
                                  OptProfileSelector::kMAX,
                                  profileDims.data(),
                                  static_cast<int>(profileDims.size()));
        } else {
          profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMIN)];
          std::cout << "min: " << vec2str(profileDims) << std::endl;
          profile->setDimensions(
              input->getName(), OptProfileSelector::kMIN, toDims(profileDims));
          profileDims = shapes[static_cast<size_t>(OptProfileSelector::kOPT)];
          std::cout << "opt: " << vec2str(profileDims) << std::endl;
          profile->setDimensions(
              input->getName(), OptProfileSelector::kOPT, toDims(profileDims));
          profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMAX)];
          std::cout << "max: " << vec2str(profileDims) << std::endl;
          profile->setDimensions(
              input->getName(), OptProfileSelector::kMAX, toDims(profileDims));
        }
      }
    }
  }
  if (profile && hasDynamicShapes) {
    if (!profile->isValid() ||
        !(builder_config_->addOptimizationProfile(profile) != -1)) {
      std::cerr << "optimization profile is invalid, or have Error in add "
                   "optimization profile."
                << std::endl;
    }
  }

  for (uint32_t i = 0, n = network()->getNbOutputs(); i < n; i++) {
    // Set formats and data types of outputs
    auto* output = network()->getOutput(i);
    output->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
  }
  // disable time cache
  // builder_config_->setFlag(BuilderFlag::kDISABLE_TIMING_CACHE);

  bool refittable = false;
  if (refittable) {
    builder_config_->setFlag(BuilderFlag::kREFIT);
  }

  SparsityFlag sparsity{SparsityFlag::kDISABLE};
  // builder_config_->setFlag(BuilderFlag::kSPARSE_WEIGHTS);

  bool use_int8 = false;
  if (use_int8) {
    IOptimizationProfile* profileCalib{nullptr};
    std::unordered_map<std::string, ShapeRange> shapesCalib;
    if (!shapesCalib.empty()) {
      profileCalib = builder_->createOptimizationProfile();
      for (uint32_t i = 0, n = network()->getNbInputs(); i < n; i++) {
        auto* input = network()->getInput(i);
        Dims profileDims{};
        auto shape = shapesCalib.find(input->getName());
        ShapeRange shapesCalib{};
        shapesCalib = shape->second;

        profileDims =
            toDims(shapesCalib[static_cast<size_t>(OptProfileSelector::kOPT)]);
        // Here we check only kMIN as all profileDims are the same.
        profileCalib->setDimensions(
            input->getName(), OptProfileSelector::kMIN, profileDims);
        profileCalib->setDimensions(
            input->getName(), OptProfileSelector::kOPT, profileDims);
        profileCalib->setDimensions(
            input->getName(), OptProfileSelector::kMAX, profileDims);
      }
      if (profileCalib->isValid() &&
          builder_config_->setCalibrationProfile(profileCalib) != -1) {
        std::cerr << "Calibration profile is invalid, or have Error in add "
                     "calibration profile."
                  << std::endl;
      }
    }

    std::vector<int64_t> elemCount{};
    for (int i = 0; i < network()->getNbInputs(); i++) {
      auto* input = network()->getInput(i);
      auto const dims = input->getDimensions();
      auto const isDynamicInput = std::any_of(
          dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim == -1; });
      if (profileCalib) {
        elemCount.push_back(volume(profileCalib->getDimensions(
            input->getName(), OptProfileSelector::kOPT)));
      } else if (profile && isDynamicInput) {
        elemCount.push_back(volume(profile->getDimensions(
            input->getName(), OptProfileSelector::kOPT)));
      } else {
        elemCount.push_back(volume(input->getDimensions()));
      }
    }

    // builder_config_->setInt8Calibrator(new RndInt8Calibrator(1, elemCount,
    // build.calibration, network, err));
  }

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
  return true;
}

bool TRTEngine::Load(const std::string& engine_file) {
  std::ifstream input(engine_file, std::ios::binary);
  if (!input) {
    std::cout << "Error opening engine file: " << engine_file << std::endl;
    std::cout << "fff" << std::endl;
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
  builder_config_->setProfileStream(stream_);
  engine_.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
  return true;
}

void TRTEngine::PrepareForRun() {
  auto num_binds = engine_->getNbBindings();
  for (auto i = 0; i < num_binds; ++i) {
    auto dims = engine_->getBindingDimensions(i);
    std::string bind_name = std::string(engine()->getBindingName(i));
    auto dtype = engine_->getBindingDataType(i);

    bool is_input = engine()->bindingIsInput(i);
    binding_buffers_->AddBinding(i, bind_name, is_input, 0, dtype);
  }
}

void TRTEngine::Run(const std::vector<Tensor>& input,
                    std::vector<Tensor>& output) {
  // bindings_.resize(engine_->getNbBindings());
  std::vector<void*> buffers(engine_->getNbBindings(), nullptr);
  // const ICudaEngine &engine = context.getEngine();

  auto input_binds = binding_buffers_->GetInputBindings();
  size_t idx = 0;
  for (auto bind_item : input_binds) {
    const int bind_idx = bind_item.second;

    context()->setBindingDimensions(bind_idx, vec2dims(input[idx].shape()));
    buffers[bind_idx] = binding_buffers_->get_binds()[bind_idx].buffer->data();
    int64_t size = volume(input[idx].shape());
    binding_buffers_->get_binds()[bind_idx].buffer->resize(size);
    cudaMemcpyAsync(binding_buffers_->get_binds()[bind_idx].buffer->data(),
                    input[idx].data(),
                    size * getElementSize(input[idx].type()),
                    cudaMemcpyHostToDevice,
                    stream_);
    context()->setBindingDimensions(bind_idx, vec2dims(input[idx].shape()));
    idx++;
  }

  auto output_binds = binding_buffers_->GetOutputBindings();
  idx = 0;
  for (auto bind_item : output_binds) {
    const int bind_idx = bind_item.second;
    buffers[bind_idx] = binding_buffers_->get_binds()[bind_idx].buffer->data();
    auto dims = engine()->getBindingDimensions(bind_idx);
    int64_t size = volume(dims);
    binding_buffers_->get_binds()[bind_idx].buffer->resize(size);
    context()->setBindingDimensions(bind_idx, vec2dims(input[idx].shape()));
    idx++;
  }

  context()->enqueueV2(buffers.data(), stream_, nullptr);
  cudaStreamSynchronize(stream_);

  idx = 0;
  for (auto bind_item : output_binds) {
    const int bind_idx = bind_item.second;
    auto dims = engine()->getBindingDimensions(bind_idx);
    output[idx].Reshape(dims2vec(dims));
    int64_t size = volume(dims);
    cudaMemcpyAsync(output[idx].data(),
                    binding_buffers_->get_binds()[bind_idx].buffer->data(),
                    size * getElementSize(output[idx].type()),
                    cudaMemcpyHostToDevice,
                    stream_);
    idx++;
  }

  if (use_profiler_) {
    profiler.printLayerTimes();
  }
}