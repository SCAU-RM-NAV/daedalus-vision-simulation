#ifndef TOOLS__ONNXRUNTIME_CUDA_HPP
#define TOOLS__ONNXRUNTIME_CUDA_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace inference
{
struct OnnxTensorOutput
{
  std::vector<std::int64_t> shape;
  std::vector<float> values;
};

class OnnxRuntimeCudaSession
{
public:
  explicit OnnxRuntimeCudaSession(const std::string & model_path, int device_id = 0);
  ~OnnxRuntimeCudaSession();

  OnnxRuntimeCudaSession(const OnnxRuntimeCudaSession &) = delete;
  OnnxRuntimeCudaSession & operator=(const OnnxRuntimeCudaSession &) = delete;

  bool uses_cuda() const;
  OnnxTensorOutput run_bgr_nchw(const cv::Mat & bgr_image) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace inference

#endif  // TOOLS__ONNXRUNTIME_CUDA_HPP
