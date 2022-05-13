#pragma once
#include "NvInfer.h"

class TRTPluginV2 : public nvinfer1::IPluginV2 {
public:
  TRTPluginV2() {}
};

class TRTPluginV2DynamicExt : public nvinfer1::IPluginV2DynamicExt {
public:
  TRTPluginV2DynamicExt() {}
};

class TRTPluginCreator : public nvinfer1::IPluginCreator {
public:
  TRTPluginCreator() = default;
};