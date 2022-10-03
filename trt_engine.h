#pragma once
#include <cuda_runtime.h>

#include <unordered_map>
#include <vector>

#include "NvInfer.h"
#include "common.h"
#include "tensor.h"
#include "trt_plugin.h"
#include "trt_utils.h"

class TRTEngine {
 public:
  TRTEngine(size_t max_batch_size, size_t max_workspace_size)
      : max_batch_size_(max_batch_size),
        max_workspace_size_(max_workspace_size),
        use_profiler_(false) {}
  ~TRTEngine() { cudaStreamDestroy(stream_); }
  bool Init();
  bool Load(const std::string &engine_file);
  bool Save(const std::string &model_file, const std::string &engine_file);
  void PrepareForRun();
  void Run(const std::vector<Tensor> &input, std::vector<Tensor> &output);
  nvinfer1::INetworkDefinition *network() { return network_.get(); }
  nvinfer1::ICudaEngine *engine() { return engine_.get(); }
  nvinfer1::IExecutionContext *context() {
    if (trt_context_ == nullptr) {
      trt_context_.reset(engine_->createExecutionContext());
    }
    return trt_context_.get();
  }

  nvinfer1::IPluginV2Layer *AddDynamicPlugin(nvinfer1::ITensor *const *inputs,
                                             int num_inputs,
                                             nvinfer1::IPluginV2 *plugin) {
    return network()->addPluginV2(inputs, num_inputs, *plugin);
  }

 private:
  size_t max_workspace_size_;
  size_t max_batch_size_;
  cudaStream_t stream_{NULL};
  TrtUniquePtr<nvinfer1::IExecutionContext> trt_context_{nullptr};
  TrtUniquePtr<nvinfer1::ICudaEngine> engine_{nullptr};
  TrtUniquePtr<nvinfer1::IBuilder> builder_;
  TrtUniquePtr<nvinfer1::IBuilderConfig> builder_config_;
  TrtUniquePtr<nvinfer1::INetworkDefinition> network_;
  TrtUniquePtr<nvinfer1::IHostMemory> serialized_engine_{nullptr};

  std::unordered_map<std::string, Tensor> output_;

  std::unique_ptr<Bindings> binding_buffers_;
  // std::vector<void*> buffers_;

  bool use_profiler_;
};