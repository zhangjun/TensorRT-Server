#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

#include "NvInferRuntimeCommon.h"

template <typename T>
struct TrtDestroyer {
  void operator()(T *t) { t->destroy(); }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T>>;

// define log
using Severity = nvinfer1::ILogger::Severity;
class Logger : public nvinfer1::ILogger {
 public:
  explicit Logger(Severity severity = Severity::kWARNING)
      : severity_(severity) {}
  nvinfer1::ILogger &getTRTLogger() { return *this; }
  Severity getReportableSeverity() const { return severity_; }

 private:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity > severity_) return;
    printf("%s %s.\n", severityPrefix(severity), msg);
  }
  static const char *severityPrefix(Severity severity) {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        return "[F] ";
      case Severity::kERROR:
        return "[E] ";
      case Severity::kWARNING:
        return "[W] ";
      case Severity::kINFO:
        return "[I] ";
      case Severity::kVERBOSE:
        return "[V] ";
      default:
        assert(0);
        return "";
    }
  }
  Severity severity_;
};

extern Logger gLogger;
// #define gLogFatal gLogger.report(Severity::kINTERNAL_ERROR, "")
// #define gLogError gLogger.report(Severity::kERROR, "")
// #define gLogWarning gLogger.report(Severity::kWARNING, "")
// #define gLogInfo gLogger.report(Severity::kINFO, "")
// #define gLogVerbose gLogger.report(Severity::kVERBOSE, "")

// static Logger gLogger(ILogger::Severity::kWARNING);

class TimerBase {
 public:
  virtual void start() {}
  virtual void stop() {}
  float microseconds() const noexcept { return mMs * 1000.f; }
  float milliseconds() const noexcept { return mMs; }
  float seconds() const noexcept { return mMs / 1000.f; }
  void reset() noexcept { mMs = 0.f; }

 protected:
  float mMs{0.0f};
};

class GpuTimer : public TimerBase {
 public:
  explicit GpuTimer(cudaStream_t stream) : mStream(stream) {
    cudaEventCreate(&mStart);
    cudaEventCreate(&mStop);
  }
  ~GpuTimer() {
    cudaEventDestroy(mStart);
    cudaEventDestroy(mStop);
  }
  void start() { cudaEventRecord(mStart, mStream); }
  void stop() {
    cudaEventRecord(mStop, mStream);
    float ms{0.0f};
    cudaEventSynchronize(mStop);
    cudaEventElapsedTime(&ms, mStart, mStop);
    mMs += ms;
  }

 private:
  cudaEvent_t mStart, mStop;
  cudaStream_t mStream;
};  // class GpuTimer

class Timer {
 public:
  void tic() {
    start_ticking_ = true;
    start_ = std::chrono::high_resolution_clock::now();
  }
  void toc() {
    if (!start_ticking_) return;
    end_ = std::chrono::high_resolution_clock::now();
    start_ticking_ = false;
    t = std::chrono::duration<double, std::milli>(end_ - start_).count();
    // std::cout << "Time: " << t << " ms" << std::endl;
  }
  double t;

 private:
  bool start_ticking_ = false;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};

class Profiler : public nvinfer1::IProfiler {
 public:
  void printLayerTimes(int itrationsTimes = 1) {
    float totalTime = 0;
    for (size_t i = 0; i < mProfile.size(); i++) {
      printf("%-40.40s %4.3fms\n",
             mProfile[i].first.c_str(),
             mProfile[i].second / itrationsTimes);
      totalTime += mProfile[i].second;
    }
    printf("Time over all layers: %4.3f\n", totalTime / itrationsTimes);
  }

 private:
  using Record = std::pair<std::string, float>;
  std::vector<Record> mProfile;

  virtual void reportLayerTime(const char *layerName, float ms) noexcept {
    auto record =
        std::find_if(mProfile.begin(), mProfile.end(), [&](const Record &r) {
          return r.first == layerName;
        });
    if (record == mProfile.end())
      mProfile.push_back(std::make_pair(layerName, ms));
    else
      record->second += ms;
  }
};

// #include <nvtx3/nvToolsExt.h>
// // Updates a server stat with duration measured by a C++ scope.
// class NvtxRange {
//  public:
//   explicit NvtxRange(const char* label) { nvtxRangePushA(label); }

//   explicit NvtxRange(const std::string& label) : NvtxRange(label.c_str()) {}

//   ~NvtxRange() { nvtxRangePop(); }
// };
// #define NVTX_INITIALIZE nvtxInitialize(nullptr)
// #define NVTX_RANGE(V, L) triton::common::NvtxRange V(L)
// #define NVTX_MARKER(L) nvtxMarkA(L)