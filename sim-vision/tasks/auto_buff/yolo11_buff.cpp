#include "yolo11_buff.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace auto_buff
{
namespace
{
std::string shape_to_string(const ov::Shape & shape)
{
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) oss << ",";
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

// 把坐标限制在图像范围内，防止越界
float clamp_float(float value, float low, float high)
{
  return std::max(low, std::min(value, high));
}

struct BuffYoloConfig
{
  std::string backend = "openvino";
  std::string model_path;
  std::string device;
  double confidence_threshold = 0.0;
  double nms_threshold = 0.0;
  double keypoint_confidence_threshold = 0.0;
  bool center_roi_enabled = false;
  int center_roi_size = 0;
  int num_requests = 0;
  int result_queue_size = 0;
};

YAML::Node require_mapping(
  const YAML::Node & parent, const std::string & key, const std::string & path)
{
  const YAML::Node node = parent[key];
  if (!node || !node.IsMap()) {
    throw std::invalid_argument("[YOLO11_BUFF config] '" + path + "' must be a mapping.");
  }
  return node;
}

template <typename T>
T require_scalar(const YAML::Node & parent, const std::string & key, const std::string & path)
{
  const YAML::Node node = parent[key];
  if (!node || !node.IsScalar()) {
    throw std::invalid_argument("[YOLO11_BUFF config] '" + path + "' must be a scalar.");
  }

  try {
    return node.as<T>();
  } catch (const YAML::Exception &) {
    throw std::invalid_argument("[YOLO11_BUFF config] '" + path + "' has an invalid value.");
  }
}

void require_probability(double value, const std::string & path)
{
  if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
    throw std::invalid_argument("[YOLO11_BUFF config] '" + path + "' must be in [0, 1].");
  }
}

void require_positive(int value, const std::string & path)
{
  if (value <= 0) {
    throw std::invalid_argument("[YOLO11_BUFF config] '" + path + "' must be positive.");
  }
}

BuffYoloConfig load_buff_yolo_config(const YAML::Node & yaml)
{
  if (!yaml || !yaml.IsMap()) {
    throw std::invalid_argument("[YOLO11_BUFF config] root must be a mapping.");
  }

  const auto detector = require_mapping(yaml, "buff_detector", "buff_detector");
  const auto openvino = require_mapping(detector, "openvino", "buff_detector.openvino");

  BuffYoloConfig config;
  if (detector["backend"]) {
    config.backend = require_scalar<std::string>(detector, "backend", "buff_detector.backend");
  }
  config.model_path = require_scalar<std::string>(detector, "model", "buff_detector.model");
  config.device =
    require_scalar<std::string>(openvino, "device", "buff_detector.openvino.device");
  config.confidence_threshold =
    require_scalar<double>(detector, "conf_threshold", "buff_detector.conf_threshold");
  config.nms_threshold =
    require_scalar<double>(detector, "nms_threshold", "buff_detector.nms_threshold");
  config.keypoint_confidence_threshold =
    require_scalar<double>(detector, "kpt_conf_threshold", "buff_detector.kpt_conf_threshold");
  config.center_roi_enabled =
    require_scalar<bool>(detector, "buff_center_roi_enable", "buff_detector.buff_center_roi_enable");
  config.center_roi_size =
    require_scalar<int>(detector, "buff_center_roi_size", "buff_detector.buff_center_roi_size");
  config.num_requests =
    require_scalar<int>(openvino, "num_requests", "buff_detector.openvino.num_requests");
  config.result_queue_size =
    require_scalar<int>(openvino, "result_queue_size", "buff_detector.openvino.result_queue_size");

  if (config.model_path.empty() || config.device.empty()) {
    throw std::invalid_argument("[YOLO11_BUFF config] model path and device must not be empty.");
  }
  require_probability(config.confidence_threshold, "buff_detector.conf_threshold");
  require_probability(config.nms_threshold, "buff_detector.nms_threshold");
  require_probability(config.keypoint_confidence_threshold, "buff_detector.kpt_conf_threshold");
  require_positive(config.center_roi_size, "buff_detector.buff_center_roi_size");
  require_positive(config.num_requests, "buff_detector.openvino.num_requests");
  require_positive(config.result_queue_size, "buff_detector.openvino.result_queue_size");
  return config;
}
}  // namespace

YOLO11_BUFF::YOLO11_BUFF(const std::string & config)
{
  const auto settings = load_buff_yolo_config(YAML::LoadFile(config));
  backend_ = settings.backend;
  const std::string & model_path = settings.model_path;
  const std::string & device = settings.device;
  confidence_threshold_ = settings.confidence_threshold;
  nms_threshold_ = settings.nms_threshold;
  kpt_conf_threshold_ = settings.keypoint_confidence_threshold;
  buff_center_roi_enable_ = settings.center_roi_enabled;
  buff_center_roi_size_ = settings.center_roi_size;
  num_requests_ = static_cast<std::size_t>(settings.num_requests);
  result_queue_limit_ = static_cast<std::size_t>(settings.result_queue_size);

  if (backend_ == "onnxruntime_cuda") {
#ifdef SP_VISION_HAS_ONNXRUNTIME_CUDA
    onnxruntime_cuda_ = std::make_unique<inference::OnnxRuntimeCudaSession>(model_path);
    tools::logger()->info(
      "[YOLO11_BUFF] model: {}, backend: ONNX Runtime CUDA, requests: {}, result_queue: {}, conf: {:.2f}, "
      "nms: {:.2f}, kpt_conf: {:.2f}, center_roi_enable: {}, center_roi_size: {}",
      model_path, num_requests_, result_queue_limit_, confidence_threshold_, nms_threshold_,
      kpt_conf_threshold_, buff_center_roi_enable_, buff_center_roi_size_);
    return;
#else
    throw std::runtime_error(
      "buff_detector.backend=onnxruntime_cuda requires a build with ONNX Runtime CUDA support");
#endif
  }
  if (backend_ != "openvino") {
    throw std::runtime_error("Unknown Buff detector backend: " + backend_);
  }

  // 读取
  model_ = core_.read_model(model_path);

  // 配置Prepostprocessor
  ov::preprocess::PrePostProcessor ppp(model_);
  auto & input = ppp.input();
  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, INPUT_SIZE, INPUT_SIZE, 3})
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);
  input.model().set_layout("NCHW");
  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale(255.0);

  // build
  model_ = ppp.build();

  // 尝试编译
  try {
    compiled_model_ = core_.compile_model(
      model_, device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
      ov::hint::num_requests(static_cast<uint32_t>(num_requests_)));
  } catch (const std::exception & e) {
    if (device == "CPU") throw;
    tools::logger()->warn(
      "[YOLO11_BUFF] Failed to compile on {}: {}. Fallback to CPU.", device, e.what());
    compiled_model_ = core_.compile_model(
      model_, "CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
      ov::hint::num_requests(static_cast<uint32_t>(num_requests_)));
  }

  // 检查输出shape是否正确
  validate_output_shape(compiled_model_.output().get_shape());

  // 创建异步请求池
  // 每个 RequestContext 里创建一个 InferRequest，绑定 callback，最后把请求 id 放进 free_request_ids_
  request_contexts_.reserve(num_requests_);
  for (std::size_t i = 0; i < num_requests_; ++i) {
    auto ctx = std::make_unique<RequestContext>();
    ctx->id = static_cast<int>(i);
    ctx->request = compiled_model_.create_infer_request();
    auto * raw_ctx = ctx.get();
    ctx->request.set_callback([this, raw_ctx](std::exception_ptr ex_ptr) {
      handle_completion(raw_ctx, ex_ptr);
    });

    request_contexts_.push_back(std::move(ctx));
    free_request_ids_.push(static_cast<int>(i));
  }

  tools::logger()->info(
    "[YOLO11_BUFF] model: {}, device: {}, output_shape: {}, requests: {}, result_queue: {}, conf: {:.2f}, "
    "nms: {:.2f}, kpt_conf: {:.2f}, center_roi_enable: {}, center_roi_size: {}",
    model_path, device, shape_to_string(compiled_model_.output().get_shape()), num_requests_,
    result_queue_limit_, confidence_threshold_, nms_threshold_, kpt_conf_threshold_,
    buff_center_roi_enable_, buff_center_roi_size_);
}

YOLO11_BUFF::~YOLO11_BUFF()
{
  for (auto & ctx : request_contexts_) {
    if (!ctx) continue;
    try {
      ctx->request.wait();
    } catch (...) {
    }
  }
  stopping_.store(true);
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_multicandidateboxes(cv::Mat & image)
{
  if (image.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return {};
  }

  cv::Mat input;
  LetterboxInfo info;
  if (!prepare_input(image, input, info)) {
    tools::logger()->warn("[YOLO11_BUFF] Failed to prepare input image.");
    return {};
  }

#ifdef SP_VISION_HAS_ONNXRUNTIME_CUDA
  if (backend_ == "onnxruntime_cuda") {
    const auto cuda_output = onnxruntime_cuda_->run_bgr_nchw(input);
    const ov::Shape shape(cuda_output.shape.begin(), cuda_output.shape.end());
    validate_output_shape(shape);
    ov::Tensor output(ov::element::f32, shape, const_cast<float *>(cuda_output.values.data()));
    return parse_output(output, info, image.size());
  }
#endif

  ov::Tensor input_tensor(ov::element::u8, {1, INPUT_SIZE, INPUT_SIZE, 3}, input.data);
  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  const ov::Tensor output = infer_request.get_output_tensor();
  auto objects = parse_output(output, info, image.size());

  return objects;
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_onecandidatebox(cv::Mat & image)
{
  auto objects = get_multicandidateboxes(image);
  if (objects.empty()) return {};

  auto best = std::max_element(objects.begin(), objects.end(), [](const Object & a, const Object & b) {
    return a.prob < b.prob;
  });
  return {*best};
}

// 异步提交
bool YOLO11_BUFF::submit(
  const cv::Mat & image, int frame_count, const std::chrono::steady_clock::time_point & timestamp)
{
  if (stopping_.load()) return false;
  if (image.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return false;
  }

#ifdef SP_VISION_HAS_ONNXRUNTIME_CUDA
  if (backend_ == "onnxruntime_cuda") {
    const auto submit_time = std::chrono::steady_clock::now();
    cv::Mat input;
    LetterboxInfo letterbox;
    if (!prepare_input(image, input, letterbox)) return false;
    try {
      const auto infer_start = std::chrono::steady_clock::now();
      const auto cuda_output = onnxruntime_cuda_->run_bgr_nchw(input);
      const auto infer_end = std::chrono::steady_clock::now();
      const ov::Shape shape(cuda_output.shape.begin(), cuda_output.shape.end());
      validate_output_shape(shape);
      ov::Tensor output(ov::element::f32, shape, const_cast<float *>(cuda_output.values.data()));

      AsyncResult result;
      result.image = image.clone();
      result.frame_count = frame_count;
      result.timestamp = timestamp;
      result.infer_dt_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
      const auto postprocess_start = std::chrono::steady_clock::now();
      result.objects = parse_output(output, letterbox, result.image.size());
      result.postprocess_dt_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - postprocess_start)
                                .count();
      result.detect_dt_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - submit_time)
                              .count();
      enqueue_async_result(std::move(result));
      return true;
    } catch (const std::exception & error) {
      tools::logger()->error("[YOLO11_BUFF CUDA] Inference failed: {}", error.what());
      return false;
    }
  }
#endif

  int id = -1;
  {
    std::lock_guard<std::mutex> lock(request_mutex_);
    if (free_request_ids_.empty()) return false;
    id = free_request_ids_.front();
    free_request_ids_.pop();
  }

  auto * ctx = request_contexts_[id].get();
  ctx->original = image.clone();
  ctx->input.release();
  ctx->letterbox = {};
  ctx->frame_count = frame_count;
  ctx->timestamp = timestamp;
  ctx->submit_time = std::chrono::steady_clock::now();
  ctx->infer_start_time = ctx->submit_time;
  ctx->preprocess_dt_ms = 0.0;

  try {
    const auto preprocess_start = std::chrono::steady_clock::now();
    if (!prepare_input(ctx->original, ctx->input, ctx->letterbox)) {
      release_request(id);
      return false;
    }
    ctx->preprocess_dt_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - preprocess_start)
                              .count();

    ov::Tensor input_tensor(ov::element::u8, {1, INPUT_SIZE, INPUT_SIZE, 3}, ctx->input.data);
    ctx->request.set_input_tensor(input_tensor);
    ctx->infer_start_time = std::chrono::steady_clock::now();
    ctx->request.start_async();
  } catch (...) {
    handle_completion(ctx, std::current_exception());
    return false;
  }

  return true;
}

bool YOLO11_BUFF::submit(const cv::Mat & image, int frame_count)
{
  return submit(image, frame_count, std::chrono::steady_clock::now());
}

bool YOLO11_BUFF::fetch(AsyncResult & result)
{
  std::lock_guard<std::mutex> lock(result_mutex_);
  while (!result_queue_.empty()) {
    auto candidate = std::move(result_queue_.front());
    result_queue_.pop_front();

    // 保证获取的结果严格按时间递增
    if (has_delivered_timestamp_ && candidate.timestamp <= last_delivered_timestamp_) {
      async_stale_drop_count_++;
      continue;
    }

    last_delivered_timestamp_ = candidate.timestamp;
    has_delivered_timestamp_ = true;
    result = std::move(candidate);
    return true;
  }

  return false;
}

// 居中letterbox
bool YOLO11_BUFF::prepare_input(
  const cv::Mat & input_image, cv::Mat & input, LetterboxInfo & info) const
{
  if (input_image.empty()) return false;

  info = {};

  cv::Rect crop_rect(0, 0, input_image.cols, input_image.rows);
  if (buff_center_roi_enable_) {
    const int requested_side = std::max(1, buff_center_roi_size_);
    const int half_side = requested_side / 2;
    const int center_x = input_image.cols / 2;
    const int center_y = input_image.rows / 2;
    crop_rect = cv::Rect(
      center_x - half_side, center_y - half_side, requested_side, requested_side) &
      cv::Rect(0, 0, input_image.cols, input_image.rows);
    if (crop_rect.width <= 0 || crop_rect.height <= 0) return false;
    info.crop_x = crop_rect.x;
    info.crop_y = crop_rect.y;
    info.crop_side = std::min(crop_rect.width, crop_rect.height);
  }

  const cv::Mat model_input_view = input_image(crop_rect);
  const double scale = std::min(
    static_cast<double>(INPUT_SIZE) / model_input_view.rows,
    static_cast<double>(INPUT_SIZE) / model_input_view.cols);
  const int resized_w = static_cast<int>(std::round(model_input_view.cols * scale));
  const int resized_h = static_cast<int>(std::round(model_input_view.rows * scale));
  const int pad_x = static_cast<int>(std::round((INPUT_SIZE - resized_w) / 2.0 - 0.1));
  const int pad_y = static_cast<int>(std::round((INPUT_SIZE - resized_h) / 2.0 - 0.1));

  input = cv::Mat(INPUT_SIZE, INPUT_SIZE, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::Mat resized;
  cv::resize(model_input_view, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);
  resized.copyTo(input(cv::Rect(pad_x, pad_y, resized_w, resized_h)));

  info.scale = scale;
  info.pad_x = pad_x;
  info.pad_y = pad_y;
  return true;
}

// 解析模型输出
std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::parse_output(
  const ov::Tensor & output, const LetterboxInfo & info, const cv::Size & image_size) const
{
  const ov::Shape shape = output.get_shape();
  validate_output_shape(shape);

  const float * data = output.data<const float>();
  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<int> labels;
  std::vector<std::vector<cv::Point2f>> keypoints_list;

  boxes.reserve(64);
  confidences.reserve(64);
  labels.reserve(64);
  keypoints_list.reserve(64);

  const auto restore_x = [&info](float x) {
    return static_cast<float>((x - info.pad_x) / info.scale + info.crop_x);
  };
  const auto restore_y = [&info](float y) {
    return static_cast<float>((y - info.pad_y) / info.scale + info.crop_y);
  };

  for (int i = 0; i < OUTPUT_PROPOSALS; ++i) {
    const float cls0 = data[i + 4 * OUTPUT_PROPOSALS];
    const float cls1 = data[i + 5 * OUTPUT_PROPOSALS];
    const int label = cls1 > cls0 ? 1 : 0;
    const float score = std::max(cls0, cls1);
    if (score < confidence_threshold_) continue;

    float kpt_conf_sum = 0.0f;
    for (int k = 0; k < KEYPOINT_NUM; ++k) {
      const int conf_offset = 4 + CLASS_NUM + k * KEYPOINT_STRIDE + 2;
      kpt_conf_sum += data[i + conf_offset * OUTPUT_PROPOSALS];
    }
    if (kpt_conf_sum / KEYPOINT_NUM < kpt_conf_threshold_) continue;

    const float cx = restore_x(data[i + 0 * OUTPUT_PROPOSALS]);
    const float cy = restore_y(data[i + 1 * OUTPUT_PROPOSALS]);
    const float w = static_cast<float>(data[i + 2 * OUTPUT_PROPOSALS] / info.scale);
    const float h = static_cast<float>(data[i + 3 * OUTPUT_PROPOSALS] / info.scale);

    const float left = clamp_float(cx - w * 0.5f, 0.0f, static_cast<float>(image_size.width - 1));
    const float top = clamp_float(cy - h * 0.5f, 0.0f, static_cast<float>(image_size.height - 1));
    const float right = clamp_float(cx + w * 0.5f, 0.0f, static_cast<float>(image_size.width - 1));
    const float bottom = clamp_float(cy + h * 0.5f, 0.0f, static_cast<float>(image_size.height - 1));
    if (right <= left || bottom <= top) continue;

    std::vector<cv::Point2f> keypoints;
    keypoints.reserve(KEYPOINT_NUM);
    for (int k = 0; k < KEYPOINT_NUM; ++k) {
      const int base = 4 + CLASS_NUM + k * KEYPOINT_STRIDE;
      keypoints.emplace_back(
        restore_x(data[i + base * OUTPUT_PROPOSALS]),
        restore_y(data[i + (base + 1) * OUTPUT_PROPOSALS]));
    }

    boxes.emplace_back(
      static_cast<int>(std::round(left)), static_cast<int>(std::round(top)),
      static_cast<int>(std::round(right - left)), static_cast<int>(std::round(bottom - top)));
    confidences.emplace_back(score);
    labels.emplace_back(label);
    keypoints_list.emplace_back(std::move(keypoints));
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, nms_threshold_, indices);

  std::vector<Object> objects;
  objects.reserve(indices.size());
  for (const int index : indices) {
    Object obj;
    obj.rect = boxes[index];
    obj.label = labels[index];
    obj.prob = confidences[index];
    obj.kpt = keypoints_list[index];
    objects.emplace_back(std::move(obj));
  }

  return objects;
}

void YOLO11_BUFF::validate_output_shape(const ov::Shape & shape) const
{
  if (shape.size() != 3 || shape[0] != 1 || shape[1] != OUTPUT_CHANNELS ||
      shape[2] != OUTPUT_PROPOSALS) {
    throw std::runtime_error(
      "[YOLO11_BUFF] invalid output shape: " + shape_to_string(shape) +
      ", expected [1,18,8400]");
  }
}

std::size_t YOLO11_BUFF::async_stale_drop_count() const
{
  std::lock_guard<std::mutex> lock(result_mutex_);
  return async_stale_drop_count_;
}

std::size_t YOLO11_BUFF::async_out_of_order_drop_count() const
{
  std::lock_guard<std::mutex> lock(result_mutex_);
  return async_out_of_order_drop_count_;
}

// 确保队列严格按时间排序
void YOLO11_BUFF::enqueue_async_result(AsyncResult result)
{
  std::lock_guard<std::mutex> lock(result_mutex_);

  if (has_delivered_timestamp_ && result.timestamp <= last_delivered_timestamp_) {
    async_stale_drop_count_++;
    return;
  }

  const bool completed_out_of_order =
    !result_queue_.empty() && result.timestamp < result_queue_.back().timestamp;
  if (completed_out_of_order) async_out_of_order_drop_count_++;

  auto insert_pos = std::upper_bound(
    result_queue_.begin(), result_queue_.end(), result.timestamp,
    [](const auto & timestamp, const AsyncResult & queued) {
      return timestamp < queued.timestamp;
    });
  result_queue_.insert(insert_pos, std::move(result));

  while (result_queue_.size() > result_queue_limit_) {
    result_queue_.pop_front();
    async_stale_drop_count_++;
  }
}

void YOLO11_BUFF::handle_completion(RequestContext * ctx, std::exception_ptr ex_ptr)
{
  if (!ctx) return;

  if (ex_ptr) {
    try {
      std::rethrow_exception(ex_ptr);
    } catch (const std::exception & e) {
      tools::logger()->error("[YOLO11_BUFF async] Inference failed: {}", e.what());
    } catch (...) {
      tools::logger()->error("[YOLO11_BUFF async] Inference failed with unknown exception.");
    }
    release_request(ctx->id);
    return;
  }

  try {
    const auto completion_time = std::chrono::steady_clock::now();
    const ov::Tensor output = ctx->request.get_output_tensor();

    AsyncResult result;
    result.image = std::move(ctx->original);
    result.preprocess_dt_ms = ctx->preprocess_dt_ms;
    result.infer_dt_ms = std::chrono::duration<double, std::milli>(
                           completion_time - ctx->infer_start_time)
                           .count();
    const auto postprocess_start = std::chrono::steady_clock::now();
    result.objects = parse_output(output, ctx->letterbox, result.image.size());
    result.postprocess_dt_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - postprocess_start)
                                 .count();
    result.frame_count = ctx->frame_count;
    result.timestamp = ctx->timestamp;
    result.detect_dt_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - ctx->submit_time).count();

    enqueue_async_result(std::move(result));
  } catch (const std::exception & e) {
    tools::logger()->error("[YOLO11_BUFF async] Postprocess failed: {}", e.what());
  } catch (...) {
    tools::logger()->error("[YOLO11_BUFF async] Postprocess failed with unknown exception.");
  }

  release_request(ctx->id);
}

void YOLO11_BUFF::release_request(int id)
{
  if (stopping_.load()) return;

  std::lock_guard<std::mutex> lock(request_mutex_);
  free_request_ids_.push(id);
}

void YOLO11_BUFF::printInputAndOutputsInfo(const ov::Model & network)
{
  std::cout << "model name: " << network.get_friendly_name() << std::endl;

  const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
  for (const ov::Output<const ov::Node> & input : inputs) {
    std::cout << "    inputs" << std::endl;

    const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
    std::cout << "        input name: " << name << std::endl;

    const ov::element::Type type = input.get_element_type();
    std::cout << "        input type: " << type << std::endl;

    const ov::Shape shape = input.get_shape();
    std::cout << "        input shape: " << shape << std::endl;
  }

  const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
  for (const ov::Output<const ov::Node> & output : outputs) {
    std::cout << "    outputs" << std::endl;

    const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
    std::cout << "        output name: " << name << std::endl;

    const ov::element::Type type = output.get_element_type();
    std::cout << "        output type: " << type << std::endl;

    const ov::Shape shape = output.get_shape();
    std::cout << "        output shape: " << shape << std::endl;
  }
}

void YOLO11_BUFF::save(const std::string & programName, const cv::Mat & image)
{
  const std::filesystem::path saveDir = "../result/";
  if (!std::filesystem::exists(saveDir)) {
    std::filesystem::create_directories(saveDir);
  }
  const std::filesystem::path savePath = saveDir / (programName + ".jpg");
  cv::imwrite(savePath.string(), image);
}
}  // namespace auto_buff
