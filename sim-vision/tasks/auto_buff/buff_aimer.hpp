#ifndef AUTO_BUFF__AIMER_HPP
#define AUTO_BUFF__AIMER_HPP

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <string>

#include "../auto_aim/planner/planner.hpp"
#include "angle_estimate.hpp"
#include "buff_ballistic.hpp"
#include "buff_pose_filter.hpp"
#include "buff_type.hpp"
#include "io/command.hpp"
#include "io/gimbal/gimbal.hpp"

namespace auto_buff
{
struct CommandGuardDebug
{
  bool applied = false;
  bool has_fresh_last = false;
  bool yaw_limited = false;
  bool pitch_limited = false;
  double raw_dt = std::numeric_limits<double>::quiet_NaN();
  double dt = std::numeric_limits<double>::quiet_NaN();
  double reference_yaw = std::numeric_limits<double>::quiet_NaN();
  double reference_pitch = std::numeric_limits<double>::quiet_NaN();
  double raw_yaw = std::numeric_limits<double>::quiet_NaN();
  double raw_pitch = std::numeric_limits<double>::quiet_NaN();
  double yaw_delta = std::numeric_limits<double>::quiet_NaN();
  double pitch_delta = std::numeric_limits<double>::quiet_NaN();
  double limited_yaw_delta = std::numeric_limits<double>::quiet_NaN();
  double limited_pitch_delta = std::numeric_limits<double>::quiet_NaN();
};

// 大符击打扇叶选取
enum class AttackSelectionReason
{
  NONE, // 没有执行
  NO_CANDIDATES, // 没有候选扇叶
  NO_VALID_BALLISTIC, // 有候选但弹道解算无效
  INITIAL_SELECTION, // 初始选择（没有上一帧目标）
  HOLD_MISSING_PREVIOUS, // 上一帧目标暂时不可见，保持上一帧目标
  SWITCH_MISSING_PREVIOUS, // 上一帧目标暂时不可见，切换到新目标
  KEEP_CURRENT, // 上一帧目标仍是最优目标，保持上一帧目标
  HOLD_SWITCH_HYSTERESIS, // 有其他候选更好，但优势或持续帧数不足，暂不切换
  SWITCH_BETTER // 新候选优势足够且连续多帧成立，正式切换
};

struct AttackLeafCandidateDebug
{
  int leaf_id = -1;
  bool valid = false;
  double cost = std::numeric_limits<double>::quiet_NaN();
  double yaw = std::numeric_limits<double>::quiet_NaN();
  double pitch = std::numeric_limits<double>::quiet_NaN();
};

struct AttackSelectionDebug
{
  AttackSelectionReason reason = AttackSelectionReason::NONE;
  int observed_leaf_id = -1;
  int previous_attack_leaf_id = -1;
  int best_leaf_id = -1;
  int selected_leaf_id = -1;
  int pending_leaf_id = -1;
  int switch_streak = 0;
  int switch_frames = 0;
  bool previous_valid = false;
  bool better_enough = false;
  double best_cost = std::numeric_limits<double>::quiet_NaN();
  double previous_cost = std::numeric_limits<double>::quiet_NaN();
  int candidate_count = 0;
  std::array<AttackLeafCandidateDebug, 5> candidates;
};

class Aimer
{
public:
  Aimer(const std::string & config_path);

  io::Command aim(
    PowerRune & rune, double observed_time_abs, double now_time_abs, double bullet_speed,
    BuffMode mode = BuffMode::LOST);

  auto_aim::Plan mpc_aim(
    PowerRune & rune, double observed_time_abs, double now_time_abs, io::GimbalState gs,
    BuffMode mode = BuffMode::LOST);

  void notify_observation_missed();
  void reset();

  AngleEstimate::Snapshot prediction_snapshot() const;
  AngleEstimate::DebugSnapshot prediction_debug_snapshot() const;
  AngleEstimate::IdMatchDebug id_match_debug() const;
  const AngleEstimateConfig & prediction_config() const;
  bool tune_prediction_config(const AngleEstimateConfig & config);
  double prediction_angle(double abs_time) const;
  double prediction_angle(double abs_time, int leaf_id) const;
  double prediction_speed(double abs_time) const;
  std::optional<Eigen::Matrix3d> prediction_rotation_reference(double abs_time) const;
  const BuffBallisticResult & last_ballistic_result() const;
  const BuffPoseSnapshot & pose_snapshot() const;
  const BuffPoseFilterDebug & pose_debug_snapshot() const;
  Eigen::Vector3d get_averaged_normal(const Eigen::Matrix3d & rotation_world) const;
  const CommandGuardDebug & command_guard_debug() const;
  const AttackSelectionDebug & attack_selection_debug() const;
  void set_deterministic_replay(bool enabled);
  void set_runtime_solve_delay_override_ms(std::optional<double> delay_ms);
  void set_mpc_fire_gap_time_override(std::optional<double> fire_gap_time);

private:
  double fire_gap_time_;  // 最小开火间隔
  double small_bias_time_ms_ = 120.0;
  double big_bias_time_ms_ = 120.0;
  int attack_leaf_id_ = -1;
  int pending_attack_leaf_id_ = -1;
  int switch_streak_ = 0;
  std::uint64_t selection_epoch_ = 0;
  double sel_w_yaw_ = 1.0;
  double sel_w_pitch_ = 0.8;
  double sel_switch_margin_ = 0.1;
  int sel_switch_frames_ = 3;
  double hold_last_leaf_sec_ = 0.05;
  int fire_rearm_confirmed_frames_ = 3; // 自动开火:连续3帧确认开火即允许开火
  double max_yaw_rate_ = 2.0;
  double max_pitch_rate_ = 2.0;
  double max_slew_dt_sec_ = 0.02;
  int confirmed_frames_since_guard_ = 0;
  bool suppress_fire_ = false;
  std::optional<double> runtime_solve_delay_override_ms_;
  std::optional<double> mpc_fire_gap_time_override_;

  BuffMode prediction_mode_ = BuffMode::BIG;
  AngleEstimate angle_estimator_;
  BuffPoseFilter pose_filter_;
  BuffBallistic ballistic_;

  double last_yaw_ = 0;
  double last_pitch_ = 0;
  double last_raw_yaw_ = 0.0;     // 上一帧展开后的 raw_yaw，用于角度连续展开
  double last_yaw_delta_ = 0.0;   // 上一帧 yaw_delta，用于调试/低通参考
  double last_yaw_vel_ = 0.0;     // 上一帧 yaw_vel，用于低通滤波
  double last_pitch_delta_ = 0.0; // 上一帧 pitch_delta，预留

  // 弹速均值滤波（滑动平均），仅弹速变化时注入窗口，第一帧原值输出
  static constexpr int kBulletSpeedFilterWindow = 5;
  double bullet_speed_buffer_[5] = {};
  int bullet_speed_buf_idx_ = 0;
  int bullet_speed_buf_count_ = 0;
  double last_raw_bullet_speed_ = 0.0;
  BuffBallisticResult last_ballistic_result_;
  CommandGuardDebug last_command_guard_debug_;
  AttackSelectionDebug last_attack_selection_debug_;
  bool has_last_plan_sample_ = false;
  double last_plan_observed_time_abs_ = 0.0;

  std::chrono::steady_clock::time_point last_fire_t_;
  bool big_single_leaf_fire_guard_enabled_ = true;
  bool big_single_leaf_fire_guard_active_ = false;
  int previous_big_detected_leaf_count_ = -1;
  std::chrono::steady_clock::time_point big_single_leaf_start_t_;
  double max_fire_yaw_error_ = 0.05;    // rad，从 yaml 读
  double max_fire_pitch_error_ = 0.05;  // rad，从 yaml 读
  struct FireKey { std::uint64_t epoch = 0; int leaf_id = -1; bool valid = false; };
  FireKey last_fire_key_;

  // 法向量滑动窗口平均，消除 PnP 共面退化导致的帧间朝向抖动
  bool normal_avg_enabled_ = false;
  int normal_avg_window_size_ = 30;
  double normal_avg_change_threshold_ = 0.015;  // ~0.86°，低于此变化不会入窗
  static constexpr int kNormalMinSamplesForOutlier = 10;
  static constexpr double kNormalOutlierSigma = 3.0;
  mutable std::deque<Eigen::Vector3d> normal_history_;
  mutable Eigen::Vector3d normal_sum_ = Eigen::Vector3d::Zero();
  mutable Eigen::Vector3d normal_sum_sq_ = Eigen::Vector3d::Zero();
  double tuned_bias_time_ms(BuffMode mode) const;
  double filter_bullet_speed(double raw_speed);
  double resolve_runtime_solve_delay_ms(
    const std::chrono::steady_clock::time_point & solve_start) const;
  AngleEstimate::IdMatchResult update_observation(
    PowerRune & rune, double observed_time_abs, BuffMode mode);
  void annotate_observation(PowerRune & rune, std::optional<int> target_leaf_id);
  void reset_attack_selection();
  bool tracking_ready() const;
  void reset_fire_rearm();
  void note_confirmed_observation();
  bool can_hold_last_leaf_plan(double observed_time_abs) const;
  io::Command hold_last_leaf_command(double observed_time_abs) const;
  auto_aim::Plan hold_last_leaf_plan(double observed_time_abs) const;
  void limit_and_record_plan(auto_aim::Plan & plan, double observed_time_abs, const io::GimbalState & gs,
    double analytical_yaw_vel, double analytical_pitch_vel);
  void limit_and_record_command(io::Command & command, double observed_time_abs);
  bool can_aim_cached_r_center(const PowerRune & rune, double observed_time_abs) const;
  void update_big_single_leaf_fire_guard(const PowerRune & rune, BuffMode mode);
  bool should_fire();
  bool should_fire(int leaf_id, double yaw_err_rad, double pitch_err_rad, double fly_time);
  BuffBallisticResult solve_r_center(
    const PowerRune & rune, const BuffBallisticTiming & timing, double bullet_speed) const;
  void select_attack_leaf(
    const PowerRune & rune, const BuffBallisticTiming & timing,
    const Eigen::Vector3d & center_world, const Eigen::Matrix3d & rotation_world,
    double bullet_speed, const io::GimbalState & gs, BuffBallisticResult & out_best);
  // 新增：利用 Ceres 拟合曲线与弹道迭代计算平滑解析速度前馈
  void calculate_analytical_velocities(
    const BuffBallisticTiming & timing, 
    const Eigen::Vector3d & center_world, 
    const Eigen::Matrix3d & rotation_world, 
    double bullet_speed, 
    int attack_leaf_id,
    double & out_yaw_vel, 
    double & out_pitch_vel) const;

};
}  // namespace auto_buff
#endif  // AUTO_AIM__AIMER_HPP
