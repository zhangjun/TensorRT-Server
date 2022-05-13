
# TensorRT-Server
## Build
```
mkdir build && cd build
cmake ..
```

# TensorRT Engine
## engine
- API
    ```
    getNBBindings()
    getBindingDimensions(index)
    getBindingName(index)
    bindingIsInput(index)
    getBindingIndex(name)
    ```
## network
- API
    ```
    markOutput()

    ```
## ILayer and ITensor
### ILayer
### ITensor

## custom plugin
    ```
    auto creator = getPluginRegistry()->getPluginCreator("mish_trt", "1");
    const PluginFieldCollection *pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin(("mish" + std::to_string(linx)).c_str(), pluginData);
    ITensor *inputTensors[] = {bn1->getOutput(0)};
    auto mish = network->addPluginV2(&inputTensors[0], 1, *pluginObj);
    return mish;
    ```
- IPluginV2IOExt
    ```
    int initialize()
    void terminate()
    void destroy()
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs)
    IPluginV2IOExt* clone()
    size_t getWorkspaceSize(int maxBatchSize)
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    int enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
    void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    void setPluginNamespace(const char* pluginNamespace)
    const char* getPluginNamespace() const
    size_t getSerializationSize()
    void serialize(void* buffer)
    const char* getPluginType()
    const char* getPluginVersion()

    ```
- IPluginCreator
    ```
    IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc)
    IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    const char* getPluginName() const
    const char* getPluginVersion()
    const PluginFieldCollection* getFieldNames()
    void setPluginNamespace(const char* libNamespace)
    const char* getPluginNamespace() const

    ```

# TensorRT Optimization

# code
https://github.com/jkjung-avt/tensorrt_demos/blob/master/yolo/onnx_to_tensorrt.py
https://github.com/jkjung-avt/tensorrt_demos/blob/master/yolo/yolo_to_onnx.py
https://github.com/jkjung-avt/tensorrt_demos/blob/master/plugins/yolo_layer.h
https://github.com/tjuskyzhang/Scaled-YOLOv4-TensorRT/blob/master/yolov4-csp-tensorrt/yololayer.h
https://github.com/tjuskyzhang/Scaled-YOLOv4-TensorRT/blob/master/yolov4-csp-tensorrt/yolov4-csp.cpp
