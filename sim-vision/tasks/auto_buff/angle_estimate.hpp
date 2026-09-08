#ifndef AUTO_BUFF__ANGLE_ESTIMATE_HPP
#define AUTO_BUFF__ANGLE_ESTIMATE_HPP

#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "tools/extended_kalman_filter.hpp"

namespace auto_buff
{
enum class BuffMode
{
  SMALL,
  BIG,
  LOST
};

enum class SmallRunePredictor
{
  CERES,
  TARGET_EKF
};

enum class PredictorBackend
{
  BIG_CERES,
  SMALL_CERES,
  SMALL_TARGET_EKF
};

struct LeafData
{
  double angle = 0.0;       // 世界坐标系下的扇叶角度
  double zero_angle = 0.0;  // 世界坐标系下的0号扇叶角度
  double t = 0.0;           // 相对于起始的时间
  int leaf_id = 0;          // 当前观测扇叶id
};

struct EnergyTri
{
  // 大符正弦速度模型
  // speed(t) = a * sin(w * (t - t0)) + (2.09 - a)
  // angle(t) = -a / w * cos(w * (t - t0)) + (2.09 - a) * t + c
  double a = 0.9125;
  double w = 1.942;
  double c = 0.0;
  double t0 = 0.0;
  // 小符线性模型
  // angle(t) = k * (t - t0) + b
  double k = M_PI / 3.0;
  double b = 0.0;
  BuffMode mode = BuffMode::LOST;
  int direction = 1;

  double getAngle(double time) const;
  double getSpeed(double time) const;
};

struct AngleEstimateConfig
{
  double max_match_angle = 0.30;  // 扇叶匹配允许的最大角度误差
  double min_residual_margin = 0.20; // 最佳和次优id匹配候选应该相差多少rad
  double max_id_angular_speed = 4.0; // ID匹配中运动学外推允许的最大角速度
  double max_zero_phase_innovation_rad = 0.1; // ID匹配中零号扇叶允许跳变多少rad
  double id_coast_timeout_sec = 0.30; // ID匹配中模型允许多少旧多少s
  double id_reset_timeout_sec = 0.50; // 多久没有新观测就重新建立ID epoch,会使得拟合数据清空，因为ID重新以新观测为0号
  // 是否用 Ceres/EKF 预测的零相位辅助 leaf_id 匹配：健康检查通过且比运动学
  // 结果更可信时会覆盖运动学结果。默认关闭，只用运动学匹配。
  bool use_model_prior = true;
  int model_min_confirmed_frames = 2; // 模型先验启动前的最小确认帧数
  double model_max_nis = 9.0; // EKF模型允许的最大NIS
  double min_time_change = 1e-4; // 认为时间有变化的最小间隔
  int outliers_threshold = 8;   // 没用上
  double sigma_multiple = 3.0;  // 没用上
  SmallRunePredictor small_predictor =
    SmallRunePredictor::CERES;  // 小符默认预测方式（ceres or target_ekf）

  int big_queue_size = 300;
  int small_queue_size = 200;
  int big_min_fit_size = 150;
  int small_min_fit_size = 100;
  double big_direction_lock_threshold = 1.0;
  double small_direction_lock_threshold = 1.0;
  double fitting_timeout_sec = 3.0;  // 最大允许的拟合时间

  double big_a_min = 0.780;
  double big_a_max = 1.045;
  double big_w_min = 1.884;
  double big_w_max = 2.000;
  double big_huber_delta = 0.06;
  double big_omega_a = 25.0;  // 历史先验权重（拉住上一帧的 a）
  double big_omega_w = 25.0;  // 历史先验权重（拉住上一帧的 w）
  double big_a_center = 0.9125; // 物理中枢 a 值
  double big_w_center = 1.942;  // 物理中枢 w 值
  double big_center_prior_weight = 5.0; // 物理中枢先验权重（弱锚定）
  int big_cold_start_coarse_steps = 24;  // 冷启动相角粗搜点数（φ ∈ [0,2π)）
  int big_cold_start_refine_steps = 8;   // 冷启动相角局部精搜点数

  // ceres option
  int max_iterations = 50;
  int ceres_threads = 2;
  double big_cost_threshold = 0.005;
  double small_cost_threshold = 0.001;
  double small_huber_delta = 0.1;

  // small target_ekf option
  double small_ekf_p0_angle = 0.02;
  double small_ekf_p0_speed = 1e-2;
  double small_ekf_q_accel = 1e-4;
  double small_ekf_r_angle = 0.02;

  static AngleEstimateConfig from_yaml(const std::string & config_path);
};

class AngleEstimate
{
public:
  enum class State
  {
    LOST,
    WAITTING,
    FITTING,
    FITTED
  };

  enum class IdMatchStatus
  {
    INITIALIZED,
    MATCHED,
    AMBIGUOUS,
    OUTLIER,
    OUT_OF_ORDER
  };

  enum class IdModelPriorStatus
  {
    DISABLED,
    WARMING_UP,
    EPOCH_MISMATCH,
    STALE,
    CERES_COST_INVALID,
    EKF_STATE_INVALID,
    EKF_NIS_INVALID,
    PREDICTION_INVALID,
    ACTIVE
  };

  struct IdMatchResult
  {
    IdMatchStatus status = IdMatchStatus::OUTLIER;
    int leaf_id = -1;
    double aligned_angle = std::numeric_limits<double>::quiet_NaN();
    double best_residual = std::numeric_limits<double>::infinity();
    double second_best_residual = std::numeric_limits<double>::infinity();
    std::uint64_t epoch = 0;

    bool confirmed() const
    {
      return status == IdMatchStatus::INITIALIZED || status == IdMatchStatus::MATCHED;
    }
  };

  struct AngleFitResult
  {
    bool success = false;
    BuffMode mode = BuffMode::LOST;
    EnergyTri energy_tri;
    double average_cost = 0.0;
    double solve_time_ms = 0.0;

    // 拟合诊断信息，用于区分冷启动、建模和 Ceres 求解耗时。
    double cold_start_time_ms = 0.0;
    double problem_build_time_ms = 0.0;
    double total_fit_wall_ms = 0.0;
    int ceres_iterations = 0;
    int ceres_successful_steps = 0;
    int ceres_unsuccessful_steps = 0;
    int residual_count = 0;
    double latest_t = 0.0;
    double latest_zero_angle = 0.0;
  };

  struct Snapshot
  {
    State state = State::LOST;
    EnergyTri energy_tri;
    double average_cost = 0.0;
    double last_ceres_solve_time_ms = 0.0;
    double last_small_ekf_update_time_ms = 0.0;
    std::size_t sample_count = 0;
    bool fitted = false;
    bool has_ceres_solve_time = false;
    PredictorBackend predictor_backend = PredictorBackend::BIG_CERES;
    SmallRunePredictor small_predictor = SmallRunePredictor::CERES;
  };

  struct IdMatchDebug
  {
    bool attempted = false;
    bool accepted = false;
    bool short_history = false;
    bool continuity_rejected = false;
    int previous_leaf_id = -1;
    int raw_best_leaf_id = -1;
    int continuity_leaf_id = -1;
    int selected_leaf_id = -1;
    double time = 0.0;
    double observed_angle = 0.0;
    double aligned_angle = 0.0; // 把观测角对其到连续相位后的值
    double best_residual = std::numeric_limits<double>::infinity();
    double raw_best_residual = std::numeric_limits<double>::infinity();
    double zero_innovation_limit = std::numeric_limits<double>::infinity();
    std::array<double, 5> expected_angles{};
    std::array<double, 5> residuals{};
    bool model_available = false;
    bool model_used = false;
    bool model_overrode_kinematic = false;
    IdModelPriorStatus model_prior_status = IdModelPriorStatus::DISABLED;
    int kinematic_leaf_id = -1;
    int model_leaf_id = -1;
    std::uint64_t model_epoch = 0;
    int confirmed_observations_in_epoch = 0;
    double model_age_sec = std::numeric_limits<double>::quiet_NaN();
    double model_zero_angle = std::numeric_limits<double>::quiet_NaN();
    std::array<double, 5> model_expected_angles{};
    std::array<double, 5> model_residuals{};
    int history_size = 0;
    int history_previous_leaf_id = -1;
    int history_last_leaf_id = -1;
    double history_previous_angle = std::numeric_limits<double>::quiet_NaN();
    double history_last_angle = std::numeric_limits<double>::quiet_NaN();
    double history_previous_zero_angle = std::numeric_limits<double>::quiet_NaN();
    double history_last_zero_angle = std::numeric_limits<double>::quiet_NaN();
    double history_dt = std::numeric_limits<double>::quiet_NaN();
    double extrapolation_dt = std::numeric_limits<double>::quiet_NaN();
    double estimated_zero_delta = std::numeric_limits<double>::quiet_NaN();
    double current_zero_angle = std::numeric_limits<double>::quiet_NaN();
    double zero_innovation = std::numeric_limits<double>::quiet_NaN();
    bool fused_same_timestamp = false;
  };

  struct SmallRuneEkfDebug
  {
    bool initialized_before = false;
    bool reset = false;
    bool short_dt_fusion = false;
    bool phase_realigned = false;
    bool updated = false;
    int phase_shift = 0;
    double input_angle = std::numeric_limits<double>::quiet_NaN();
    double time = std::numeric_limits<double>::quiet_NaN();
    double dt = std::numeric_limits<double>::quiet_NaN();
    double state_angle_before = std::numeric_limits<double>::quiet_NaN();
    double state_speed_before = std::numeric_limits<double>::quiet_NaN();
    double raw_innovation = std::numeric_limits<double>::quiet_NaN();
    double aligned_measurement = std::numeric_limits<double>::quiet_NaN();
    double predicted_angle = std::numeric_limits<double>::quiet_NaN();
    double predicted_speed = std::numeric_limits<double>::quiet_NaN();
    double state_angle_after = std::numeric_limits<double>::quiet_NaN();
    double state_speed_after = std::numeric_limits<double>::quiet_NaN();
    double p00 = std::numeric_limits<double>::quiet_NaN();
    double p01 = std::numeric_limits<double>::quiet_NaN();
    double p10 = std::numeric_limits<double>::quiet_NaN();
    double p11 = std::numeric_limits<double>::quiet_NaN();
    double nis = std::numeric_limits<double>::quiet_NaN();
  };

  struct CeresFitDebug
  {
    bool job_submitted = false;
    bool result_received = false;
    bool result_applied = false;
    bool result_rejected = false;
    std::uint64_t submitted_epoch = 0;
    std::uint64_t submitted_sample_version = 0;
    int submitted_sample_count = 0;
    double submitted_first_sample_time = std::numeric_limits<double>::quiet_NaN();
    double submitted_last_sample_time = std::numeric_limits<double>::quiet_NaN();
    std::uint64_t result_epoch = 0;
    std::uint64_t result_sample_version = 0;
    int result_sample_count = 0;
    double result_first_sample_time = std::numeric_limits<double>::quiet_NaN();
    double result_last_sample_time = std::numeric_limits<double>::quiet_NaN();
    double previous_k = std::numeric_limits<double>::quiet_NaN();
    double previous_b = std::numeric_limits<double>::quiet_NaN();
    double result_k = std::numeric_limits<double>::quiet_NaN();
    double result_b = std::numeric_limits<double>::quiet_NaN();
    double average_cost = std::numeric_limits<double>::quiet_NaN();
    double solve_time_ms = std::numeric_limits<double>::quiet_NaN();
    double total_fit_wall_ms = std::numeric_limits<double>::quiet_NaN();
    int residual_count = 0;
  };

  struct DebugSnapshot
  {
    Snapshot summary;
    std::uint64_t id_epoch = 0;
    std::uint64_t model_epoch = 0;
    double last_model_sample_time_abs = std::numeric_limits<double>::quiet_NaN();
    int confirmed_observations_in_epoch = 0;
    int current_leaf_id = -1;
    int lost_count = 0;
    bool direction_locked = false;
    int direction = 0;
    double direction_sum = 0.0;
    double start_time = 0.0;
    std::uint64_t latest_sample_version = 0;
    std::uint64_t submitted_sample_version = 0;
    std::uint64_t committed_fit_version = 0;
    int recent_leaf_data_count = 0;
    std::array<LeafData, 3> recent_leaf_datas{};
    IdMatchDebug id_match;
    SmallRuneEkfDebug small_ekf;
    CeresFitDebug ceres_fit;
  };

  explicit AngleEstimate(const AngleEstimateConfig & config = {});
  AngleEstimate(
    double max_match_angle, double min_time_change, int outliers_threshold, double sigma_multiple);
  ~AngleEstimate();

  AngleEstimate(const AngleEstimate &) = delete;
  AngleEstimate & operator=(const AngleEstimate &) = delete;
  AngleEstimate(AngleEstimate &&) = delete;
  AngleEstimate & operator=(AngleEstimate &&) = delete;

  void init(double leaf_angle, double time, BuffMode mode);
  IdMatchResult update(double leaf_angle, double time);
  IdMatchResult update(double leaf_angle, double time, BuffMode mode);
  double predict(double time) const;
  double predict(double time, int id) const;
  double predictDelta(double from_time, double to_time) const;
  double predictSpeed(double time) const;
  bool angleFit();
  void setDeterministicReplay(bool enabled);

  bool angleFitOnSnapshot(
    const std::deque<LeafData> & leaf_datas, const EnergyTri & tri_snapshot, State state_snapshot,
    AngleFitResult & result) const;
  bool angleFitOnSnapshot(
    const std::deque<LeafData> & leaf_datas, const EnergyTri & tri_snapshot, State state_snapshot,
    bool ever_fit_snapshot, AngleFitResult & result) const;
  void applyAngleFitResult(const AngleFitResult & result);

  State estimate_state() const { return estimate_state_; }
  bool is_lost() const { return estimate_state_ == State::LOST; }
  bool is_fitted() const { return fitted_; }
  bool has_successful_fit() const { return ever_fit_; }
  int current_leaf_id() const { return detect_leaf_id_; }
  bool direction_locked() const { return direction_locked_; }
  int direction() const { return energy_tri_.direction; }
  void reset()
  {
    estimate_state_ = State::LOST;
    energy_tri_.mode = BuffMode::LOST;
    direction_locked_ = false;
    direction_sum_ = 0.0;
    leaf_datas_.clear();
    fitted_ = false;
    ever_fit_ = false;
    small_ekf_ = SmallRuneEkf{};
  }
  std::uint64_t id_epoch() const { return id_epoch_; }
  // 上一确认帧的零号角；无历史时 nullopt
  std::optional<double> last_zero_angle() const
  {
    if (leaf_datas_.empty()) return std::nullopt;
    return leaf_datas_.back().zero_angle;
  }
  // 干跑 leaf_id 匹配：不改 leaf_datas_ / detect_leaf_id_
  IdMatchResult matchLeafIdOnly(double angle, double time_abs) const;
  // 双解消歧：各自 match→zero，用 |Δzero|+方向选 index；无历史或无法判定返回 nullopt
  std::optional<std::size_t> pickIppeByZeroContinuity(
    const std::vector<double> & angles, double time_abs,
    const std::vector<double> * reproj = nullptr) const;
  Snapshot snapshot() const;
  DebugSnapshot debug_snapshot() const;
  const IdMatchDebug & last_id_match_debug() const { return last_id_match_debug_; }
  const AngleEstimateConfig & config() const { return config_; }
  bool updateConfig(const AngleEstimateConfig & config);
  const std::deque<LeafData> & leaf_datas() const { return leaf_datas_; }
  const EnergyTri & energy_tri() const { return energy_tri_; }
  double average_cost() const { return average_cost_; }

private:
  // Main estimator state.
  AngleEstimateConfig config_;
  State estimate_state_ = State::LOST;
  int detect_leaf_id_ = 0;
  std::uint64_t id_epoch_ = 0;
  double last_confirmed_time_abs_ = std::numeric_limits<double>::quiet_NaN();
  double last_observation_time_abs_ = std::numeric_limits<double>::quiet_NaN();
  std::uint64_t model_epoch_ = 0;
  double last_model_sample_time_abs_ = std::numeric_limits<double>::quiet_NaN();
  int confirmed_observations_in_epoch_ = 0;
  IdMatchDebug last_id_match_debug_;
  std::deque<LeafData> leaf_datas_;
  double start_time_ = 0.0;
  EnergyTri energy_tri_;
  bool direction_locked_ = false;
  double direction_sum_ = 0.0;
  bool fitted_ = false;
  bool ever_fit_ = false;
  double average_cost_ = 0.0;
  double last_ceres_solve_time_ms_ = 0.0;
  double last_small_ekf_update_time_ms_ = 0.0;
  bool has_ceres_solve_time_ = false;
  int lost_count_ = 0;
  int fitting_long_time_count_ = 0;
  double fitting_start_rel_time_ = 0.0;
  std::uint64_t fit_epoch_ = 0;
  std::uint64_t latest_sample_version_ = 0;
  std::uint64_t submitted_sample_version_ = 0;
  std::uint64_t committed_fit_version_ = 0;
  std::chrono::steady_clock::time_point last_fit_submit_t_ =
    std::chrono::steady_clock::time_point::min();
  bool deterministic_replay_ = false;
  double deterministic_last_fit_time_abs_ = std::numeric_limits<double>::quiet_NaN();

  // Small rune EKF (小符 EKF).
  struct SmallRuneEkf
  {
    // 状态 x = [angle, speed]
    // 观测 z = 当前 physical_angle
    // 方向由 AngleEstimate 级的累积位移判据决定，这里只在 reset 时消费其符号作为速度先验
    void reset(double angle, double time, const AngleEstimateConfig & config, int direction);
    void reanchor(double angle, double time);
    void update(double angle, double time, const AngleEstimateConfig & config);
    double predict(double time) const;
    double speed() const;
    bool initialized() const { return initialized_; }
    SmallRuneEkfDebug debug() const { return last_debug_; }

    static constexpr double SMALL_W = M_PI / 3.0;

    bool initialized_ = false;
    double last_time_ = 0.0;
    tools::ExtendedKalmanFilter ekf_;
    Eigen::Matrix2d F_ = Eigen::Matrix2d::Identity();
    Eigen::Matrix2d Q_ = Eigen::Matrix2d::Zero();
    Eigen::Matrix<double, 1, 2> H_ = Eigen::Matrix<double, 1, 2>::Zero();
    Eigen::Matrix<double, 1, 1> R_ = Eigen::Matrix<double, 1, 1>::Identity();
    SmallRuneEkfDebug last_debug_;
  };
  SmallRuneEkf small_ekf_;
  CeresFitDebug last_ceres_fit_debug_;

  // Async Ceres fitting state for big rune Ceres (大符 Ceres) and small rune Ceres (小符 Ceres).
  struct FitJob
  {
    std::uint64_t epoch = 0;
    std::uint64_t sample_version = 0;
    std::deque<LeafData> leaf_datas;
    EnergyTri energy_tri;
    State state = State::LOST;
    bool ever_fit = false;
  };

  struct FitResult
  {
    std::uint64_t epoch = 0;
    std::uint64_t sample_version = 0;
    int sample_count = 0;
    double first_sample_time = std::numeric_limits<double>::quiet_NaN();
    double last_sample_time = std::numeric_limits<double>::quiet_NaN();
    AngleFitResult result;
    bool fit_ok = false;
  };

  mutable std::mutex fit_mutex_;
  std::condition_variable fit_cv_;
  std::thread fit_worker_;  // ceres拟合线程，避免ceres拟合时间长阻塞主线程
  bool fit_worker_quit_ = false;
  bool fit_worker_busy_ = false;
  std::optional<FitJob> pending_fit_job_;
  std::optional<FitResult> pending_fit_result_;
  mutable std::mutex config_mutex_;

  // Main update, Ceres dispatch, and worker helpers.
  void resetTrackingState(double leaf_angle, double time, BuffMode mode, bool preserve_locked_direction);
  // 扇叶 id 匹配：闭式最近邻 + 连续性门 + 可选模型先验，写入 detect_leaf_id_ 与 debug。
  // 返回带对齐角度的匹配结果；leaf_datas_ 由调用方 update() 负责写入。
  IdMatchResult updateLeafId(double angle, double time);
  IdMatchResult reinitializeIdEpoch(double leaf_angle, double time, BuffMode mode);
  EnergyTri alignFitModelPhase(EnergyTri fitted_model, double reference_time, double reference_zero_angle) const;
  double predictZero(double time) const;
  void updateDirectionLock();
  void updateState(double relative_time);
  bool useTargetEkf() const;
  bool usesAsyncCeres() const;
  bool collectFitResult();
  bool submitFitJobIfReady();
  bool runDeterministicFitIfReady();
  bool shouldSubmitFitJob(std::chrono::steady_clock::time_point now) const;
  PredictorBackend currentPredictorBackend() const;
  void fitWorkerLoop();
};

}  // namespace auto_buff

#endif  // AUTO_BUFF__ANGLE_ESTIMATE_HPP
