#include "yolov5.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <array>
#include <stdexcept>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
YOLOV5::YOLOV5(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false), post_filter_(config_path)
{
  auto yaml = YAML::LoadFile(config_path);

  model_path_ = yaml["yolov5_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  use_traditional_ = yaml["use_traditional"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  const auto configured_backend = yaml["yolov5_backend"]
                                    ? yaml["yolov5_backend"].as<std::string>()
                                    : (yaml["inference_backend"]
                                         ? yaml["inference_backend"].as<std::string>()
                                         : "openvino");
  if (configured_backend == "onnxruntime_cuda") {
#ifdef SP_VISION_HAS_ONNXRUNTIME_CUDA
    backend_ = InferenceBackend::OnnxRuntimeCuda;
    onnxruntime_cuda_ = std::make_unique<inference::OnnxRuntimeCudaSession>(model_path_);
    tools::logger()->info("[YOLOV5] model: {}, backend: ONNX Runtime CUDA", model_path_);
    return;
#elif !defined(SP_VISION_WITH_ONNXRUNTIME)
    throw std::runtime_error(
      "yolov5_backend=onnxruntime_cuda requires a build with ONNX Runtime CUDA support");
#endif
  }
  if (configured_backend == "opencv_dnn_cuda") {
    backend_ = InferenceBackend::OpenCVDnnCuda;
    model_path_ = yaml["yolov5_onnx_model_path"].as<std::string>();
    if (model_path_.empty()) {
      throw std::runtime_error("yolov5_onnx_model_path must be set for opencv_dnn_cuda.");
    }
    dnn_net_ = cv::dnn::readNetFromONNX(model_path_);
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0) {
      throw std::runtime_error("OpenCV was built without an available CUDA device.");
    }
    dnn_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    dnn_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    tools::logger()->info("[YOLOV5] OpenCV DNN CUDA backend enabled.");
    return;
  }
  if (configured_backend == "onnxruntime" ||
      (configured_backend == "onnxruntime_cuda" && backend_ != InferenceBackend::OnnxRuntimeCuda)) {
#ifdef SP_VISION_WITH_ONNXRUNTIME
    backend_ = configured_backend == "onnxruntime_cuda"
                 ? InferenceBackend::OnnxRuntimeCuda
                 : InferenceBackend::OnnxRuntime;
    if (yaml["yolov5_onnx_model_path"])
      model_path_ = yaml["yolov5_onnx_model_path"].as<std::string>();
    if (model_path_.empty()) {
      throw std::runtime_error("yolov5_onnx_model_path must be set for the ONNX Runtime backend.");
    }

    ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "sp_vision");
    ort_session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    ort_session_options_.SetIntraOpNumThreads(1);
    if (configured_backend == "onnxruntime_cuda") {
      OrtCUDAProviderOptions cuda_options{};
      ort_session_options_.AppendExecutionProvider_CUDA(cuda_options);
    }
    ort_session_ = std::make_unique<Ort::Session>(
      *ort_env_, model_path_.c_str(), ort_session_options_);
    Ort::AllocatorWithDefaultOptions allocator;
    ort_input_name_ = ort_session_->GetInputNameAllocated(0, allocator).get();
    ort_output_name_ = ort_session_->GetOutputNameAllocated(0, allocator).get();
    tools::logger()->info("[YOLOV5] ONNX Runtime backend enabled: {}", configured_backend);
    return;
#else
    throw std::runtime_error(
      "This binary was built without ONNX Runtime. Reconfigure with "
      "-DSP_VISION_ENABLE_ONNXRUNTIME=ON.");
#endif
  }
  if (configured_backend != "openvino") {
    throw std::runtime_error("Unknown inference_backend: " + configured_backend);
  }

#ifndef SP_VISION_WITH_OPENVINO
  throw std::runtime_error("OpenVINO backend was selected, but this binary has no OpenVINO support.");
#else
  auto model = core_.read_model(model_path_);
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

  // TODO: ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
  model = ppp.build();
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
#endif
}

YOLOV5::~YOLOV5() = default;

std::list<Armor> YOLOV5::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) {  // -1 表示该维度不裁切
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1) {  // -1 表示该维度不裁切
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  auto x_scale = static_cast<double>(640) / bgr_img.rows;
  auto y_scale = static_cast<double>(640) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  // preproces
  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input(roi), {w, h});

#ifdef SP_VISION_HAS_ONNXRUNTIME_CUDA
  if (backend_ == InferenceBackend::OnnxRuntimeCuda && onnxruntime_cuda_) {
    const auto output_tensor = onnxruntime_cuda_->run_bgr_nchw(input);
    if (output_tensor.shape.size() != 3 || output_tensor.shape[0] != 1 ||
        output_tensor.shape[1] != 25200 || output_tensor.shape[2] != 22) {
      throw std::runtime_error("YOLOv5 CUDA backend received an unexpected output shape");
    }
    cv::Mat output(25200, 22, CV_32F, const_cast<float *>(output_tensor.values.data()));
    return parse(scale, output, raw_img, frame_count);
  }
#endif

#ifdef SP_VISION_WITH_ONNXRUNTIME
  if ((backend_ == InferenceBackend::OnnxRuntime ||
       backend_ == InferenceBackend::OnnxRuntimeCuda) && ort_session_) {
    // The exported YOLOv5 ONNX model expects RGB, float32, NCHW input in [0, 1].
    cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar(), true);
    std::array<int64_t, 4> input_shape{1, 3, 640, 640};
    const std::array<const char *, 1> input_names{ort_input_name_.c_str()};
    const std::array<const char *, 1> output_names{ort_output_name_.c_str()};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, blob.ptr<float>(), static_cast<size_t>(blob.total()), input_shape.data(),
      input_shape.size());
    auto outputs = ort_session_->Run(
      Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
    auto & output = outputs.front();
    const auto output_info = output.GetTensorTypeAndShapeInfo();
    const auto output_shape = output_info.GetShape();
    if (output_shape.size() != 3 || output_shape[0] != 1) {
      throw std::runtime_error("YOLOv5 ONNX output must have shape [1, proposals, features].");
    }
    cv::Mat output_mat(
      static_cast<int>(output_shape[1]), static_cast<int>(output_shape[2]), CV_32F,
      output.GetTensorMutableData<float>());
    return parse(scale, output_mat, raw_img, frame_count);
  }
#endif

  if (backend_ == InferenceBackend::OpenCVDnnCuda) {
    cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar(), true);
    dnn_net_.setInput(blob);
    cv::Mat output = dnn_net_.forward();
    if (output.dims != 3 || output.size[0] != 1) {
      throw std::runtime_error("YOLOv5 OpenCV DNN output must have shape [1, proposals, features].");
    }
    cv::Mat output_mat = output.reshape(1, output.size[1]);
    return parse(scale, output_mat, raw_img, frame_count);
  }

#ifdef SP_VISION_WITH_OPENVINO
  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);

  // infer
  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  // postprocess
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());

  return parse(scale, output, raw_img, frame_count);
#else
  throw std::runtime_error("No enabled YOLOv5 inference backend.");
#endif
}

std::list<Armor> YOLOV5::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // for each row: xywh + classess
  std::vector<int> color_ids, num_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;
  for (int r = 0; r < output.rows; r++) {
    double score = output.at<float>(r, 8);
    score = sigmoid(score);

    if (score < score_threshold_) continue;

    std::vector<cv::Point2f> armor_key_points;

    //颜色和类别独热向量
    cv::Mat color_scores = output.row(r).colRange(9, 13);     //color
    cv::Mat classes_scores = output.row(r).colRange(13, 22);  //num
    cv::Point class_id, color_id;
    int _class_id, _color_id;
    double score_color, score_num;
    cv::minMaxLoc(classes_scores, NULL, &score_num, NULL, &class_id);
    cv::minMaxLoc(color_scores, NULL, &score_color, NULL, &color_id);
    _class_id = class_id.x;
    _color_id = color_id.x;

    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 0) / scale, output.at<float>(r, 1) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 6) / scale, output.at<float>(r, 7) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 4) / scale, output.at<float>(r, 5) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 2) / scale, output.at<float>(r, 3) / scale));

    float min_x = armor_key_points[0].x;
    float max_x = armor_key_points[0].x;
    float min_y = armor_key_points[0].y;
    float max_y = armor_key_points[0].y;

    for (int i = 1; i < armor_key_points.size(); i++) {
      if (armor_key_points[i].x < min_x) min_x = armor_key_points[i].x;
      if (armor_key_points[i].x > max_x) max_x = armor_key_points[i].x;
      if (armor_key_points[i].y < min_y) min_y = armor_key_points[i].y;
      if (armor_key_points[i].y > max_y) max_y = armor_key_points[i].y;
    }

    cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

    color_ids.emplace_back(_color_id);
    num_ids.emplace_back(_class_id);
    boxes.emplace_back(rect);
    confidences.emplace_back(score);
    armors_key_points.emplace_back(armor_key_points);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    if (use_roi_) {
      armors.emplace_back(
        color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  tmp_img_ = bgr_img;
  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }
    // 使用传统方法二次矫正角点
    if (!post_filter_.check_geometry(*it)) {
      it = armors.erase(it);
      continue;
    }

    if (use_traditional_) detector_.detect(*it, bgr_img);

    if (!post_filter_.check_color(*it, bgr_img)) {
      it = armors.erase(it);
      continue;
    }

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

bool YOLOV5::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  // 保存不确定的图案，用于神经网络的迭代
  // if (name_ok && !confidence_ok) save(armor);

  return name_ok && confidence_ok;
}

bool YOLOV5::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? armor.name != ArmorName::one
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost && armor.name != ArmorName::base);

  // 保存异常的图案，用于神经网络的迭代
  // if (!name_ok) save(armor);

  return name_ok;
}

cv::Point2f YOLOV5::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void YOLOV5::draw_detections(
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
    cv::rectangle(detection, roi_, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
  cv::imshow("detection", detection);
}

void YOLOV5::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, ARMOR_NAMES[armor.name], file_name);
  cv::imwrite(img_path, tmp_img_);
}

double YOLOV5::sigmoid(double x)
{
  if (x > 0)
    return 1.0 / (1.0 + exp(-x));
  else
    return exp(x) / (1.0 + exp(x));
}

std::list<Armor> YOLOV5::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim
