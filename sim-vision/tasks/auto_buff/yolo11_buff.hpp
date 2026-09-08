#ifndef AUTO_BUFF__YOLO11_BUFF_HPP
#define AUTO_BUFF__YOLO11_BUFF_HPP
#include <yaml-cpp/yaml.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <queue>
#include <string>
#include <vector>

#include "tools/logger.hpp"
#ifdef SP_VISION_HAS_ONNXRUNTIME_CUDA
#include "tools/onnxruntime_cuda.hpp"
#endif

namespace auto_buff
{
inline const std::vector<std::string> class_names = {"red", "blue"};

class YOLO11_BUFF
{
public:
  struct Object // 检测结果
  {
    cv::Rect_<float> rect; // 还原到原图坐标
    int label; // 0=red，1=blue
    float prob; // 置信度
    std::vector<cv::Point2f> kpt; // 4个关键点
  };

  struct AsyncResult // 异步推理完成后返回的一帧结果
  {
    cv::Mat image; // 原图副本
    std::vector<Object> objects; // 这一帧识别到的所有结果
    int frame_count = -1; // 针号
    std::chrono::steady_clock::time_point timestamp; // 提交时的时间戳
    double detect_dt_ms = 0.0; // 从提交到回调完成后处理的耗时
    double preprocess_dt_ms = 0.0; // 预处理耗时
    double infer_dt_ms = 0.0; // 推理耗时
    double postprocess_dt_ms = 0.0; // 后处理耗时
  };

  YOLO11_BUFF(const std::string & config);
  ~YOLO11_BUFF();

  // 使用NMS，用来获取多个框（这是同步推理的接口，现在改用异步推理）
  std::vector<Object> get_multicandidateboxes(cv::Mat & image);

  // 寻找置信度最高的框（这是同步推理的接口，现在改用异步推理）
  std::vector<Object> get_onecandidatebox(cv::Mat & image);

  // 异步提交，传图像、帧号和时间戳
  bool submit(
    const cv::Mat & image, int frame_count,
    const std::chrono::steady_clock::time_point & timestamp);

  // 异步提交，传图像以及当前时间戳
  bool submit(const cv::Mat & image, int frame_count);

  // 异步获取，从结果队列里取一个已经完成的异步结果
  bool fetch(AsyncResult & result);

  bool try_fetch(AsyncResult & result) { return fetch(result); }

  std::size_t async_stale_drop_count() const;
  std::size_t async_out_of_order_drop_count() const;

private:
  struct LetterboxInfo // 预处理缩放
  {
    double scale = 1.0;
    double pad_x = 0.0;
    double pad_y = 0.0;
    int crop_x = 0;
    int crop_y = 0;
    int crop_side = 0;
  };

  static constexpr int INPUT_SIZE = 640;
  static constexpr int CLASS_NUM = 2;
  static constexpr int KEYPOINT_NUM = 4;
  static constexpr int KEYPOINT_STRIDE = 3; // x/y/visibility
  static constexpr int OUTPUT_CHANNELS = 4 + CLASS_NUM + KEYPOINT_NUM * KEYPOINT_STRIDE;
  static constexpr int OUTPUT_PROPOSALS = 8400;

  struct RequestContext // 每个异步请求上下文
  {
    int id = -1; // 请求编号
    ov::InferRequest request;
    cv::Mat original; // 原图副本，确保异步期间图像内存还在
    cv::Mat input; // letterbox后的640x640输入图
    LetterboxInfo letterbox; // letterbox参数
    int frame_count = -1; // 帧号
    std::chrono::steady_clock::time_point timestamp; // 时间戳
    std::chrono::steady_clock::time_point submit_time; // 提交时间
    std::chrono::steady_clock::time_point infer_start_time; // 异步推理启动时间
    double preprocess_dt_ms = 0.0; // 预处理耗时
  };

  ov::Core core_;  // 创建OpenVINO Runtime Core对象
  std::shared_ptr<ov::Model> model_;
  ov::CompiledModel compiled_model_;
  std::string backend_ = "openvino";
#ifdef SP_VISION_HAS_ONNXRUNTIME_CUDA
  std::unique_ptr<inference::OnnxRuntimeCudaSession> onnxruntime_cuda_;
#endif

  double confidence_threshold_ = 0.7;
  double nms_threshold_ = 0.4;
  double kpt_conf_threshold_ = 0.5;
  bool buff_center_roi_enable_ = false;
  int buff_center_roi_size_ = 960;
  std::size_t num_requests_ = 2; // 异步请求数量
  std::size_t result_queue_limit_ = 2; // 结果队列最大长度，满了会丢旧结果

  std::vector<std::unique_ptr<RequestContext>> request_contexts_; // 请求池
  std::queue<int> free_request_ids_; // 空闲请求池
  std::mutex request_mutex_;

  std::deque<AsyncResult> result_queue_; // 异步完成结果队列
  mutable std::mutex result_mutex_;
  std::chrono::steady_clock::time_point last_delivered_timestamp_ =
    std::chrono::steady_clock::time_point::min();
  bool has_delivered_timestamp_ = false;
  std::size_t async_stale_drop_count_ = 0;
  std::size_t async_out_of_order_drop_count_ = 0;

  std::atomic<bool> stopping_{false};
  mutable std::mutex infer_mutex_; // 未使用

  // 进行letterbox
  bool prepare_input(const cv::Mat & input_image, cv::Mat & input, LetterboxInfo & info) const;

  // 解析模型输出，做坐标还原、阈值过滤、NMS
  std::vector<Object> parse_output(
    const ov::Tensor & output, const LetterboxInfo & info, const cv::Size & image_size) const;

  // 检查输出 shape，防止跑错模型
  void validate_output_shape(const ov::Shape & shape) const;

  // 异步回调完成后的统一处理函数
  void handle_completion(RequestContext * ctx, std::exception_ptr ex_ptr);

  void enqueue_async_result(AsyncResult result);

  // 把请求id放回空闲队列
  void release_request(int id);

  // 打印模型信息, 这个函数修改自$${OPENVINO_COMMON}/utils/src/args_helper.cpp的同名函数
  void printInputAndOutputsInfo(const ov::Model & network);

  // 将image保存为"../result/$${programName}.jpg"
  void save(const std::string & programName, const cv::Mat & image);
};
}  // namespace auto_buff
#endif
