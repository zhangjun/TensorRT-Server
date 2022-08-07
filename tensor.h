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
  INVALID,
};

enum class DeviceType {
  CPU,
  CUDA,
};

class Tensor {
public:
  void Reshape(const std::vector<int>& data_shape) {
    dims_.assign(data_shape.begin(), data_shape.end());
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

  std::vector<int> shape() const {
    return dims_;
  }

  int64_t numel() const {
    int64_t res = 0L;
    for(auto dim: dims_) {
      res *= dim;
    }
    return res;
  }
  const std::string& name() const;
  DataType type() const {
    return data_type_;
  }
  DeviceType device() const {
    return device_type_;
  }

protected:
  const void* data() const {
    return static_cast<const void*>(data_);
  }

  void* data() {
    return data_;
  }
  void* data_{nullptr};
  std::vector<int> dims_;
  DataType data_type_{DataType::FLOAT};
  DeviceType device_type_{DeviceType::CPU};
};

struct NamedTensor {
  std::string name;
  Tensor tensor;
};