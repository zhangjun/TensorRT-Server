#pragma once

class Tensor {};
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

// struct Tensor {
//   void* data{nullptr};
//   std::vector<int> dims;
//   DateType date_type{DataType::FLOAT};
//   DeviceType device_type{DeviceType::CPU};
// };