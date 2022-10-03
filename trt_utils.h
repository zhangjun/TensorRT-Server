#pragma once
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "NvInferRuntimeCommon.h"

inline uint32_t getElementSize(nvinfer1::DataType t) noexcept {
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return 1;
  }
  return 0;
}

inline int64_t volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline int64_t volume(const std::vector<int>& shape) {
  return std::accumulate(
      shape.begin(), shape.end() + shape.size(), 1, std::multiplies<int64_t>());
}

inline nvinfer1::Dims vec2dims(const std::vector<int>& shape) {
  nvinfer1::Dims dims{-1};
  dims.nbDims = static_cast<int>(shape.size());

  for (size_t i = 0; i < shape.size(); ++i) {
    dims.d[i] = static_cast<int>(shape[i]);
  }
  return dims;
}

inline std::vector<int> dims2vec(const nvinfer1::Dims& d) {
  std::vector<int> shape;
  for (size_t i = 0; i < d.nbDims; ++i) {
    shape.push_back(d.d[i]);
  }
  return shape;
}

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class
//! handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte
//!          buffers. The template parameters AllocFunc and FreeFunc are used
//!          for the allocation and deallocation of the buffer. AllocFunc must
//!          be a functor that takes in (void** ptr, size_t size) and returns
//!          bool. ptr is a pointer to where the allocated buffer address should
//!          be stored. size is the amount of memory in bytes to allocate. The
//!          boolean indicates whether or not the memory allocation was
//!          successful. FreeFunc must be a functor that takes in (void* ptr)
//!          and returns void. ptr is the allocated buffer address. It must work
//!          with nullptr input.
//!
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer {
 public:
  //!
  //! \brief Construct an empty buffer.
  //!
  GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
      : mSize(0), mCapacity(0), mType(type), mBuffer(nullptr) {}

  //!
  //! \brief Construct a buffer with the specified allocation size in bytes.
  //!
  GenericBuffer(size_t size, nvinfer1::DataType type)
      : mSize(size), mCapacity(size), mType(type) {
    if (!allocFn(&mBuffer, this->nbBytes())) {
      throw std::bad_alloc();
    }
  }

  GenericBuffer(GenericBuffer&& buf)
      : mSize(buf.mSize),
        mCapacity(buf.mCapacity),
        mType(buf.mType),
        mBuffer(buf.mBuffer) {
    buf.mSize = 0;
    buf.mCapacity = 0;
    buf.mType = nvinfer1::DataType::kFLOAT;
    buf.mBuffer = nullptr;
  }

  GenericBuffer& operator=(GenericBuffer&& buf) {
    if (this != &buf) {
      freeFn(mBuffer);
      mSize = buf.mSize;
      mCapacity = buf.mCapacity;
      mType = buf.mType;
      mBuffer = buf.mBuffer;
      // Reset buf.
      buf.mSize = 0;
      buf.mCapacity = 0;
      buf.mBuffer = nullptr;
    }
    return *this;
  }

  //!
  //! \brief Returns pointer to underlying array.
  //!
  void* data() { return mBuffer; }

  //!
  //! \brief Returns pointer to underlying array.
  //!
  const void* data() const { return mBuffer; }

  //!
  //! \brief Returns the size (in number of elements) of the buffer.
  //!
  size_t size() const { return mSize; }

  //!
  //! \brief Returns the size (in bytes) of the buffer.
  //!
  size_t nbBytes() const { return this->size() * getElementSize(mType); }

  //!
  //! \brief Resizes the buffer. This is a no-op if the new size is smaller than
  //! or equal to the current capacity.
  //!
  void resize(size_t newSize) {
    mSize = newSize;
    if (mCapacity < newSize) {
      freeFn(mBuffer);
      if (!allocFn(&mBuffer, this->nbBytes())) {
        throw std::bad_alloc{};
      }
      mCapacity = newSize;
    }
  }

  //!
  //! \brief Overload of resize that accepts Dims
  //!
  void resize(const nvinfer1::Dims& dims) { return this->resize(volume(dims)); }

  ~GenericBuffer() { freeFn(mBuffer); }

 private:
  size_t mSize{0}, mCapacity{0};
  nvinfer1::DataType mType;
  void* mBuffer;
  AllocFunc allocFn;
  FreeFunc freeFn;
};

class DeviceAllocator {
 public:
  bool operator()(void** ptr, size_t size) const {
    return cudaMalloc(ptr, size) == cudaSuccess;
  }
};

class DeviceFree {
 public:
  void operator()(void* ptr) const { cudaFree(ptr); }
};

class PinnedAllocator {
 public:
  bool operator()(void** ptr, size_t size) const {
    return cudaMallocHost(ptr, size) == cudaSuccess;
  }
};

class PinnedFree {
 public:
  void operator()(void* ptr) const { cudaFreeHost(ptr); }
};
class HostAllocator {
 public:
  bool operator()(void** ptr, size_t size) const {
    *ptr = malloc(size);
    return *ptr != nullptr;
  }
};

class HostFree {
 public:
  void operator()(void* ptr) const { free(ptr); }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using PinnedBuffer = GenericBuffer<PinnedAllocator, PinnedFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

struct Binding {
  bool is_input{false};
  int64_t volume{0};
  nvinfer1::DataType data_type{nvinfer1::DataType::kFLOAT};
  std::unique_ptr<DeviceBuffer> buffer;
  std::unique_ptr<HostBuffer> host_buffer;
  std::string name;
};

class Bindings {
 public:
  Bindings() = delete;

  void AddBinding(int32_t idx,
                  const std::string& name,
                  bool is_input,
                  int64_t volume,
                  nvinfer1::DataType data_type) {
    while (bindings_.size() <= static_cast<size_t>(idx)) {
      bindings_.emplace_back();
    }
    names_[name] = idx;
    if (bindings_[idx].buffer == nullptr) {
      bindings_[idx].buffer.reset(new DeviceBuffer);
    }
    // Some memory allocators return nullptr when allocating zero bytes, but
    // TensorRT requires a non-null ptr even for empty tensors, so allocate a
    // dummy byte.
    if (volume == 0) {
      bindings_[idx].buffer->resize(1);
    } else {
      bindings_[idx].buffer->resize(static_cast<size_t>(volume));
    }
    bindings_[idx].is_input = is_input;
    bindings_[idx].data_type = data_type;
    bindings_[idx].name = name;
  }

  std::unordered_map<std::string, int> GetInputBindings() {
    return GetBindings([](const Binding& b) -> bool { return b.is_input; });
  }

  std::unordered_map<std::string, int> GetOutputBindings() {
    return GetBindings([](const Binding& b) -> bool { return !b.is_input; });
  }

  std::unordered_map<std::string, int> GetBindings() {
    return GetBindings([](const Binding& b) -> bool { return true; });
  }

  std::unordered_map<std::string, int> GetBindings(
      std::function<bool(const Binding& b)> predicate) {
    std::unordered_map<std::string, int> bindings;
    for (const auto& n : names_) {
      const auto binding = n.second;
      if (predicate(bindings_[binding])) {
        bindings.insert(n);
      }
    }
    return bindings;
    // std::vector<Binding> bindings;
    // for (const auto& b : bindings_) {
    //   if (predicate(b)) {
    //     bindings.emplace_back(std::move(b));
    //   }
    // }
    // return bindings;
  }

  const std::vector<Binding>& get_binds() { return bindings_; }

 private:
  std::unordered_map<std::string, int32_t> names_;
  std::vector<Binding> bindings_;
};

inline nvinfer1::Dims toDims(const std::vector<int>& vec) {
  int limit = static_cast<int>(nvinfer1::Dims::MAX_DIMS);
  if (static_cast<int>(vec.size()) > limit) {
    // sample::gLogWarning << "Vector too long, only first 8 elements are used
    // in dimension." << std::endl;
  }
  // Pick first nvinfer1::Dims::MAX_DIMS elements
  nvinfer1::Dims dims{std::min(static_cast<int>(vec.size()), limit), {}};
  std::copy_n(vec.begin(), dims.nbDims, std::begin(dims.d));
  return dims;
}

enum class SparsityFlag { kDISABLE, kENABLE, kFORCE };

using ShapeRange =
    std::array<std::vector<int32_t>,
               nvinfer1::EnumMax<nvinfer1::OptProfileSelector>()>;