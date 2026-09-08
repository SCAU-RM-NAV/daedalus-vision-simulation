#include "tools/onnxruntime_cuda.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

namespace inference
{
namespace
{
constexpr std::array<std::string_view, 5> kCudaLibraries = {
  "libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12", "libcurand.so.10", "libcudnn.so.9"};

Ort::Env & ort_environment()
{
  // ONNX Runtime owns process-wide exit handlers that depend on this environment.
  static auto * environment = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "sp_vision");
  return *environment;
}

void load_cuda_runtime_libraries()
{
  const char * environment_root = std::getenv("ONNXRUNTIME_CUDA_RUNTIME_ROOT");
#ifdef SP_VISION_ONNXRUNTIME_CUDA_RUNTIME_ROOT
  const std::filesystem::path runtime_root =
    environment_root != nullptr ? environment_root : SP_VISION_ONNXRUNTIME_CUDA_RUNTIME_ROOT;
#else
  if (environment_root == nullptr) {
    throw std::runtime_error(
      "ONNX Runtime CUDA needs ONNXRUNTIME_CUDA_RUNTIME_ROOT to locate CUDA and cuDNN libraries");
  }
  const std::filesystem::path runtime_root = environment_root;
#endif

  const std::array<std::filesystem::path, 5> library_dirs = {
    runtime_root / "nvidia/cuda_runtime/lib", runtime_root / "nvidia/cublas/lib",
    runtime_root / "nvidia/cublas/lib", runtime_root / "nvidia/curand/lib", runtime_root / "nvidia/cudnn/lib"};

  for (std::size_t index = 0; index < kCudaLibraries.size(); ++index) {
    const auto path = library_dirs[index] / kCudaLibraries[index];
    if (!std::filesystem::is_regular_file(path)) {
      throw std::runtime_error("missing ONNX Runtime CUDA dependency: " + path.string());
    }
    if (dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL) == nullptr) {
      throw std::runtime_error("cannot load ONNX Runtime CUDA dependency " + path.string() + ": " + dlerror());
    }
  }
}

bool has_cuda_provider()
{
  const auto providers = Ort::GetAvailableProviders();
  return std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();
}

std::vector<std::int64_t> required_input_shape(const Ort::Session & session)
{
  if (session.GetInputCount() != 1) {
    throw std::runtime_error("ONNX Runtime CUDA backend requires exactly one model input");
  }
  const auto shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] != 640 || shape[3] != 640) {
    throw std::runtime_error("ONNX Runtime CUDA backend requires input shape [1,3,640,640]");
  }
  return shape;
}

std::string allocated_name(Ort::AllocatedStringPtr name)
{
  if (!name) throw std::runtime_error("ONNX Runtime model has an unnamed input or output");
  return name.get();
}
}  // namespace

class OnnxRuntimeCudaSession::Impl
{
public:
  Impl(const std::string & model_path, int device_id)
  {
    (void)ort_environment();
    load_cuda_runtime_libraries();
    if (!has_cuda_provider()) {
      throw std::runtime_error("ONNX Runtime was built without CUDAExecutionProvider");
    }

    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options_.SetIntraOpNumThreads(1);
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = device_id;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
    cuda_options.do_copy_in_default_stream = 1;
    options_.AppendExecutionProvider_CUDA(cuda_options);
    session_ = std::make_unique<Ort::Session>(ort_environment(), model_path.c_str(), options_);

    input_shape_ = required_input_shape(*session_);
    input_type_ = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
    if (input_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        input_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      throw std::runtime_error("ONNX Runtime CUDA backend supports only float32 and float16 model inputs");
    }
    if (session_->GetOutputCount() != 1) {
      throw std::runtime_error("ONNX Runtime CUDA backend requires exactly one model output");
    }
    if (session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType() !=
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("ONNX Runtime CUDA backend requires float32 model output");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = allocated_name(session_->GetInputNameAllocated(0, allocator));
    output_name_ = allocated_name(session_->GetOutputNameAllocated(0, allocator));
  }

  OnnxTensorOutput run_bgr_nchw(const cv::Mat & bgr_image) const
  {
    if (bgr_image.empty() || bgr_image.type() != CV_8UC3 || bgr_image.rows != input_shape_[2] ||
        bgr_image.cols != input_shape_[3]) {
      throw std::invalid_argument("ONNX Runtime CUDA input must be a 640x640 CV_8UC3 image");
    }

    cv::Mat rgb;
    cv::cvtColor(bgr_image, rgb, cv::COLOR_BGR2RGB);
    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> channels;
    cv::split(normalized, channels);

    constexpr std::size_t kImageArea = 640U * 640U;
    std::array<const char *, 1> input_names = {input_name_.c_str()};
    std::array<const char *, 1> output_names = {output_name_.c_str()};
    Ort::RunOptions run_options;
    const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> outputs;
    if (input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      std::vector<Ort::Float16_t> input(kImageArea * 3U);
      for (std::size_t channel = 0; channel < 3; ++channel) {
        const auto * source = channels[channel].ptr<float>();
        auto * destination = input.data() + channel * kImageArea;
        for (std::size_t index = 0; index < kImageArea; ++index) destination[index] = Ort::Float16_t(source[index]);
      }
      auto tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
        memory_info, input.data(), input.size(), input_shape_.data(), input_shape_.size());
      outputs = session_->Run(
        run_options, input_names.data(), &tensor, 1, output_names.data(), output_names.size());
    } else {
      std::vector<float> input(kImageArea * 3U);
      for (std::size_t channel = 0; channel < 3; ++channel) {
        std::memcpy(
          input.data() + channel * kImageArea, channels[channel].ptr<float>(), kImageArea * sizeof(float));
      }
      auto tensor = Ort::Value::CreateTensor<float>(
        memory_info, input.data(), input.size(), input_shape_.data(), input_shape_.size());
      outputs = session_->Run(
        run_options, input_names.data(), &tensor, 1, output_names.data(), output_names.size());
    }

    const auto & output = outputs.at(0);
    const auto output_info = output.GetTensorTypeAndShapeInfo();
    OnnxTensorOutput result;
    result.shape = output_info.GetShape();
    const auto count = output_info.GetElementCount();
    const auto * data = output.GetTensorData<float>();
    result.values.assign(data, data + count);
    return result;
  }

private:
  Ort::SessionOptions options_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::int64_t> input_shape_;
  ONNXTensorElementDataType input_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  std::string input_name_;
  std::string output_name_;
};

OnnxRuntimeCudaSession::OnnxRuntimeCudaSession(const std::string & model_path, int device_id)
: impl_(std::make_unique<Impl>(model_path, device_id))
{
}

OnnxRuntimeCudaSession::~OnnxRuntimeCudaSession() = default;

bool OnnxRuntimeCudaSession::uses_cuda() const { return impl_ != nullptr; }

OnnxTensorOutput OnnxRuntimeCudaSession::run_bgr_nchw(const cv::Mat & bgr_image) const
{
  return impl_->run_bgr_nchw(bgr_image);
}
}  // namespace inference
