#include "yolo11.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
namespace
{
int yaml_int_or(const YAML::Node & yaml, const std::string & key, int default_value)
{
  return yaml[key] ? yaml[key].as<int>() : default_value;
}

std::size_t clamp_positive(int value, std::size_t default_value)
{
  return value > 0 ? static_cast<std::size_t>(value) : default_value;
}
}  // namespace

class YOLO11::ThreadPool
{
public:
  explicit ThreadPool(std::size_t thread_num)
  {
    thread_num = std::max<std::size_t>(1, thread_num);
    workers_.reserve(thread_num);
    for (std::size_t i = 0; i < thread_num; ++i) {
      workers_.emplace_back([this]() { worker_loop(); });
    }
  }

  ~ThreadPool() { stop(); }

  bool post(std::function<void()> task)
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) return false;
      tasks_.push(std::move(task));
    }
    cv_.notify_one();
    return true;
  }

  void stop()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) return;
      stopping_ = true;
    }
    cv_.notify_all();

    for (auto & worker : workers_) {
      if (worker.joinable()) worker.join();
    }
    workers_.clear();
  }

private:
  void worker_loop()
  {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return stopping_ || !tasks_.empty(); });
        if (stopping_ && tasks_.empty()) return;
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stopping_ = false;
};

struct YOLO11::RequestContext
{
  int id = -1;
  ov::InferRequest request;
  cv::Mat original;
  cv::Mat input;
  double scale = 1.0;
  int frame_count = -1;
  std::chrono::steady_clock::time_point timestamp;
  std::chrono::steady_clock::time_point submit_time;
};

YOLO11::YOLO11(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false), post_filter_(config_path)
{
  auto yaml = YAML::LoadFile(config_path);

  model_path_ = yaml["yolo11_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  int x = yaml["roi"]["x"].as<int>();
  int y = yaml["roi"]["y"].as<int>();
  int width = yaml["roi"]["width"].as<int>();
  int height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  auto model = core_.read_model(model_path_);
  const auto output_shape = model->output().get_shape();
  if (
    output_shape.size() != 3 || output_shape[0] != 1 ||
    output_shape[1] != static_cast<std::size_t>(output_feature_num_)) {
    throw std::runtime_error(fmt::format(
      "YOLO11 model output does not match the {}-class postprocess: got rank {} and {} "
      "features, expected shape [1, {}, proposals]",
      class_num_, output_shape.size(), output_shape.size() > 1 ? output_shape[1] : 0,
      output_feature_num_));
  }

  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 640, 640, 3})
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale(255.0);

  model = ppp.build();

  const int configured_requests = yaml_int_or(yaml, "openvino_num_requests", 0);
  const int configured_preprocess_threads = yaml_int_or(yaml, "openvino_preprocess_threads", 2);
  const int configured_postprocess_threads = yaml_int_or(yaml, "openvino_postprocess_threads", 1);
  result_queue_limit_ = clamp_positive(yaml_int_or(yaml, "openvino_result_queue_size", 2), 2);

  if (configured_requests > 0) {
    compiled_model_ = core_.compile_model(
      model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
      ov::hint::num_requests(static_cast<uint32_t>(configured_requests)));
    num_requests_ = static_cast<std::size_t>(configured_requests);
  } else {
    compiled_model_ = core_.compile_model(
      model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));

    try {
      num_requests_ = static_cast<std::size_t>(
        compiled_model_.get_property(ov::optimal_number_of_infer_requests));
    } catch (const std::exception & e) {
      tools::logger()->warn(
        "[YOLO11 async] Failed to query optimal infer requests: {}. Fallback to 2.", e.what());
      num_requests_ = 2;
    }
  }

  num_requests_ = std::max<std::size_t>(1, num_requests_);

  preprocess_pool_ = std::make_unique<ThreadPool>(
    clamp_positive(configured_preprocess_threads, 2));
  postprocess_pool_ = std::make_unique<ThreadPool>(
    clamp_positive(configured_postprocess_threads, 1));

  request_contexts_.reserve(num_requests_);
  for (std::size_t i = 0; i < num_requests_; ++i) {
    auto ctx = std::make_unique<RequestContext>();
    ctx->id = static_cast<int>(i);
    ctx->request = compiled_model_.create_infer_request();
    auto * raw_ctx = ctx.get();

    ctx->request.set_callback([this, raw_ctx](std::exception_ptr ex_ptr) {
      if (stopping_.load()) return;

      const bool posted = postprocess_pool_->post([this, raw_ctx, ex_ptr]() {
        handle_completion(raw_ctx, ex_ptr);
      });

      if (!posted) {
        release_request(raw_ctx->id);
      }
    });

    request_contexts_.push_back(std::move(ctx));
    free_request_ids_.push(static_cast<int>(i));
  }

  tools::logger()->info(
    "[YOLO11 async] InferRequest pool created, num_requests: {}, preprocess_threads: {}, "
    "postprocess_threads: {}, result_queue: {}",
    num_requests_, clamp_positive(configured_preprocess_threads, 2),
    clamp_positive(configured_postprocess_threads, 1), result_queue_limit_);
}

YOLO11::~YOLO11()
{
  stopping_.store(true);

  if (preprocess_pool_) preprocess_pool_->stop();

  for (auto & ctx : request_contexts_) {
    if (!ctx) continue;
    try {
      ctx->request.cancel();
    } catch (...) {
    }
    try {
      ctx->request.wait();
    } catch (...) {
    }
  }

  if (postprocess_pool_) postprocess_pool_->stop();
}

bool YOLO11::prepare_input(const cv::Mat & raw_img, cv::Mat & input, double & scale) const
{
  if (raw_img.empty()) return false;

  cv::Mat bgr_img;
  if (use_roi_) {
    cv::Rect roi_local = roi_;
    if (roi_local.width == -1) roi_local.width = raw_img.cols - roi_local.x;
    if (roi_local.height == -1) roi_local.height = raw_img.rows - roi_local.y;

    roi_local &= cv::Rect(0, 0, raw_img.cols, raw_img.rows);
    if (roi_local.width <= 0 || roi_local.height <= 0) return false;

    bgr_img = raw_img(roi_local);
  } else {
    bgr_img = raw_img;
  }

  const auto x_scale = static_cast<double>(640) / bgr_img.rows;
  const auto y_scale = static_cast<double>(640) / bgr_img.cols;
  scale = std::min(x_scale, y_scale);
  const auto h = static_cast<int>(bgr_img.rows * scale);
  const auto w = static_cast<int>(bgr_img.cols * scale);

  if (input.empty() || input.rows != 640 || input.cols != 640 || input.type() != CV_8UC3) {
    input.create(640, 640, CV_8UC3);
  }
  input.setTo(cv::Scalar(0, 0, 0));
  const auto input_roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input(input_roi), {w, h});

  return true;
}

std::list<Armor> YOLO11::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  tmp_img_ = raw_img;

  auto infer_request = compiled_model_.create_infer_request();
  auto input_tensor = infer_request.get_input_tensor();
  cv::Mat input(640, 640, CV_8UC3, input_tensor.data<uint8_t>());

  double scale = 1.0;
  if (!prepare_input(raw_img, input, scale)) {
    tools::logger()->warn("[YOLO11] Failed to prepare input image.");
    return std::list<Armor>();
  }

  infer_request.start_async();
  infer_request.wait();

  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(
    static_cast<int>(output_shape[1]), static_cast<int>(output_shape[2]), CV_32F,
    output_tensor.data<float>());
  cv::Mat output_copy = output.clone();

  return parse(scale, output_copy, raw_img, frame_count, true);
}

bool YOLO11::submit(
  const cv::Mat & img, int frame_count,
  const std::chrono::steady_clock::time_point & timestamp)
{
  if (stopping_.load()) return false;

  if (img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return false;
  }

  int id = -1;
  {
    std::lock_guard<std::mutex> lock(request_mutex_);
    if (free_request_ids_.empty()) {
      return false;  // request pool full: caller should drop this frame
    }
    id = free_request_ids_.front();
    free_request_ids_.pop();
  }

  auto * ctx = request_contexts_[id].get();
  ctx->original = img.clone();
  ctx->input.release();
  ctx->scale = 1.0;
  ctx->frame_count = frame_count;
  ctx->timestamp = timestamp;
  ctx->submit_time = std::chrono::steady_clock::now();

  const bool posted = preprocess_pool_->post([this, ctx]() {
    if (stopping_.load()) {
      release_request(ctx->id);
      return;
    }

    try {
      auto input_tensor = ctx->request.get_input_tensor();
      ctx->input = cv::Mat(640, 640, CV_8UC3, input_tensor.data<uint8_t>());
      if (!prepare_input(ctx->original, ctx->input, ctx->scale)) {
        tools::logger()->warn("[YOLO11 async] Failed to prepare input image.");
        release_request(ctx->id);
        return;
      }

      ctx->request.start_async();
    } catch (...) {
      handle_completion(ctx, std::current_exception());
    }
  });

  if (!posted) {
    release_request(id);
    return false;
  }

  return true;
}

bool YOLO11::submit(
  const cv::Mat & img, const std::chrono::steady_clock::time_point & timestamp)
{
  return submit(img, -1, timestamp);
}

bool YOLO11::fetch(YOLOAsyncResult & result)
{
  std::lock_guard<std::mutex> lock(result_mutex_);
  if (result_queue_.empty()) return false;

  result = std::move(result_queue_.front());
  result_queue_.pop_front();
  return true;
}

bool YOLO11::try_fetch(YOLOAsyncResult & result)
{
  return fetch(result);
}

void YOLO11::handle_completion(RequestContext * ctx, std::exception_ptr ex_ptr)
{
  if (!ctx) return;

  if (ex_ptr) {
    try {
      std::rethrow_exception(ex_ptr);
    } catch (const std::exception & e) {
      tools::logger()->error("[YOLO11 async] Inference failed: {}", e.what());
    } catch (...) {
      tools::logger()->error("[YOLO11 async] Inference failed with unknown exception.");
    }
    release_request(ctx->id);
    return;
  }

  try {
    auto output_tensor = ctx->request.get_output_tensor();
    auto output_shape = output_tensor.get_shape();
    cv::Mat output(
      static_cast<int>(output_shape[1]), static_cast<int>(output_shape[2]), CV_32F,
      output_tensor.data<float>());
    cv::Mat output_copy = output.clone();

    std::list<Armor> armors;
    {
      // parse() uses post_filter_ and may draw/save in debug paths; serialize it deliberately.
      std::lock_guard<std::mutex> lock(parse_mutex_);
      armors = parse(ctx->scale, output_copy, ctx->original, ctx->frame_count, false);
    }

    YOLOAsyncResult result;
    result.img = std::move(ctx->original);
    result.armors = std::move(armors);
    result.timestamp = ctx->timestamp;
    result.stamp = ctx->timestamp;
    result.detect_dt = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - ctx->submit_time).count();
    result.frame_count = ctx->frame_count;

    {
      std::lock_guard<std::mutex> lock(result_mutex_);
      while (result_queue_.size() >= result_queue_limit_) {
        result_queue_.pop_front();
      }
      result_queue_.push_back(std::move(result));
    }
  } catch (const std::exception & e) {
    tools::logger()->error("[YOLO11 async] Postprocess failed: {}", e.what());
  } catch (...) {
    tools::logger()->error("[YOLO11 async] Postprocess failed with unknown exception.");
  }

  release_request(ctx->id);
}

void YOLO11::release_request(int id)
{
  if (stopping_.load()) return;

  std::lock_guard<std::mutex> lock(request_mutex_);
  free_request_ids_.push(id);
}

std::list<Armor> YOLO11::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count, bool draw_debug)
{  // for each row: xywh + classes + keypoints
  cv::transpose(output, output);

  if (output.cols != output_feature_num_) {
    tools::logger()->warn(
      "[YOLO11] unexpected output width: {}, expected exactly {} for {} classes", output.cols,
      output_feature_num_, class_num_);
    return {};
  }

  std::vector<int> ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;
  for (int r = 0; r < output.rows; r++) {
    auto xywh = output.row(r).colRange(0, 4);
    auto scores = output.row(r).colRange(4, 4 + class_num_);
    auto one_key_points = output.row(r).colRange(4 + class_num_, output_feature_num_);

    std::vector<cv::Point2f> armor_key_points;

    double score;
    cv::Point max_point;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &max_point);

    if (score < score_threshold_) continue;

    auto x = xywh.at<float>(0);
    auto y = xywh.at<float>(1);
    auto w = xywh.at<float>(2);
    auto h = xywh.at<float>(3);
    auto left = static_cast<int>((x - 0.5 * w) / scale);
    auto top = static_cast<int>((y - 0.5 * h) / scale);
    auto width = static_cast<int>(w / scale);
    auto height = static_cast<int>(h / scale);

    bool keypoints_ok = true;
    for (int i = 0; i < keypoint_num_; i++) {
      const int base = i * keypoint_stride_;
      float x = one_key_points.at<float>(0, base) / static_cast<float>(scale);
      float y = one_key_points.at<float>(0, base + 1) / static_cast<float>(scale);
      float kp_conf = one_key_points.at<float>(0, base + 2);

      if (kp_conf < keypoint_score_threshold_) {
        keypoints_ok = false;
        break;
      }

      cv::Point2f kp = {x, y};
      armor_key_points.push_back(kp);
    }
    if (!keypoints_ok) continue;

    ids.emplace_back(max_point.x);
    confidences.emplace_back(static_cast<float>(score));
    boxes.emplace_back(left, top, width, height);
    armors_key_points.emplace_back(armor_key_points);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    sort_keypoints(armors_key_points[i]);
    if (use_roi_) {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }

    if (!post_filter_.keep(*it, bgr_img)) {
      it = armors.erase(it);
      continue;
    }

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_ && draw_debug) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

bool YOLO11::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  // 保存不确定的图案，用于神经网络的迭代
  // if (name_ok && !confidence_ok) save(armor);

  return name_ok && confidence_ok;
}

bool YOLO11::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? armor.name != ArmorName::one
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost && armor.name != ArmorName::base);

  // 保存异常的图案，用于神经网络的迭代
  // if (!name_ok) save(armor);

  return name_ok;
}

cv::Point2f YOLO11::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void YOLO11::sort_keypoints(std::vector<cv::Point2f> & keypoints)
{
  if (keypoints.size() != 4) {
    std::cout << "beyond 4!!" << std::endl;
    return;
  }

  std::sort(keypoints.begin(), keypoints.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.y < b.y;
  });

  std::vector<cv::Point2f> top_points = {keypoints[0], keypoints[1]};
  std::vector<cv::Point2f> bottom_points = {keypoints[2], keypoints[3]};

  std::sort(top_points.begin(), top_points.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.x < b.x;
  });

  std::sort(
    bottom_points.begin(), bottom_points.end(),
    [](const cv::Point2f & a, const cv::Point2f & b) { return a.x < b.x; });

  keypoints[0] = top_points[0];     // top-left
  keypoints[1] = top_points[1];     // top-right
  keypoints[2] = bottom_points[1];  // bottom-right
  keypoints[3] = bottom_points[0];  // bottom-left
}

void YOLO11::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::Rect roi_local = roi_;
    if (roi_local.width == -1) roi_local.width = img.cols - roi_local.x;
    if (roi_local.height == -1) roi_local.height = img.rows - roi_local.y;
    roi_local &= cv::Rect(0, 0, img.cols, img.rows);
    cv::rectangle(detection, roi_local, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
  cv::imshow("detection", detection);
}

void YOLO11::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, ARMOR_NAMES[armor.name], file_name);
  cv::imwrite(img_path, tmp_img_);
}

std::list<Armor> YOLO11::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count, true);
}

}  // namespace auto_aim
