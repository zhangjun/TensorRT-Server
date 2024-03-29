#pragma once

// class Tensor {};
// class Tensor {
// public:
//   void Resize(const DDimLite &ddim) { dims_ = ddim; }
//   size_t data_size() const { return this->dims().production(); }
//   size_t memory_size() const { return memory_size_; }
//   size_t offset() const { return offset_; }
//   void *raw_data() {
//     return static_cast<char *>(
//         (static_cast<char *>(buffer_->data()) + offset_));
//   }
// private:
//   DDimLite dims_;
//   std::shared_ptr<Buffer> buffer_;
//   LoD lod_;
//   size_t memory_size_{};

//   /// @brief Buffer may be shared with other tensors
//   size_t offset_{0};
// };

enum class DataType {
  FLOAT,
  HALF,
  INT8,
  INT16,
  INT32,
  INT64,
  DOUBLE,
  BOOL,
  INVALID,
};

enum class DeviceType {
  CPU,
  CUDA,
};

inline int64_t getElementSize(DataType t) noexcept {
  switch (t) {
    case DataType::INT32:
      return 4;
    case DataType::FLOAT:
      return 4;
    case DataType::INT16:
      return 2;
    case DataType::HALF:
      return 2;
    case DataType::INT64:
      return 8;
    case DataType::DOUBLE:
      return 8;
    case DataType::BOOL:
      return 1;
    case DataType::INT8:
      return 1;
    default:
      return 0;
  }
}

namespace nvinfer1
{
  enum class DataType;
}

inline DataType GetDataType(const nvinfer1::DataType& dtype) {
  if (dtype == nvinfer1::DataType::kFLOAT) {
    return DataType::FLOAT;
  } else if (dtype == nvinfer1::DataType::kHALF) {
    return DataType::HALF;
  } else if (dtype == nvinfer1::DataType::kINT8) {
    return DataType::INT8;
  } else if (dtype == nvinfer1::DataType::kINT32) {
    return DataType::INT32;
  }
  // kBOOL
  return DataType::BOOL;
}
class Tensor {
 public:
  void Reshape(const std::vector<int>& data_shape, DataType data_type) {
    dims_.assign(data_shape.begin(), data_shape.end());
    data_type_ = data_type;
    size_t size = 0;
    for (auto dim : data_shape) {
      size *= dim;
    }
    data_ = reinterpret_cast<uint8_t*>(malloc(size));
  }

  template <typename T>
  T* mutable_data() {
    return reinterpret_cast<T*>(data());
  }

  template <typename T>
  T* data() const {
    return static_cast<const T*>(data());
  }

  template <typename T>
  void CopyFromCpu(const T* data) {
    size_t ele_size = numel() * sizeof(T);
  }

  template <typename T>
  void CopyToCpu(T* data) const {}

  std::vector<int> shape() const { return dims_; }

  int64_t numel() const {
    int64_t res = 0L;
    for (auto dim : dims_) {
      res *= dim;
    }
    return res;
  }
  const std::string& name() const;
  DataType type() const { return data_type_; }
  DeviceType device() const { return device_type_; }

  const void* data() const { return static_cast<const void*>(data_); }

  void* data() { return data_; }

  ~Tensor() {
    if (data_ != nullptr) {
      free(data_);
      data_ = nullptr;
    }
  }

//  private:
//   Tensor(const Tensor& tensor) = delete;
//   Tensor(const Tensor&& tensor) = delete;
//   Tensor& operator=(const Tensor&) = delete;
//   Tensor& operator=(const Tensor&&) = delete;

 protected:
  void* data_{nullptr};
  std::vector<int> dims_;
  DataType data_type_{DataType::FLOAT};
  DeviceType device_type_{DeviceType::CPU};
};

struct NamedTensor {
  std::string name;
  Tensor tensor;
};