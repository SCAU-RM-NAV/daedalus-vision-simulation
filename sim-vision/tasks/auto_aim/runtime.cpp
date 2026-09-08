#include "runtime.hpp"

#include <stdexcept>

namespace auto_aim
{
Runtime::Runtime(const std::string & config_path)
: solver_(config_path), tracker_(config_path, solver_), planner_(config_path)
{
}

std::list<Target> Runtime::track(std::list<Armor> & armors, const RuntimeFrame & frame)
{
  const bool has_runtime_calibration =
    frame.camera_matrix != nullptr || frame.distort_coeffs != nullptr ||
    frame.R_camera2gimbal != nullptr || frame.t_camera2gimbal != nullptr;
  if (has_runtime_calibration) {
    if (frame.camera_matrix == nullptr || frame.distort_coeffs == nullptr ||
        frame.R_camera2gimbal == nullptr || frame.t_camera2gimbal == nullptr) {
      throw std::invalid_argument("runtime calibration must provide intrinsics and full extrinsics");
    }
    solver_.set_runtime_calibration(
      *frame.camera_matrix, *frame.distort_coeffs, *frame.R_camera2gimbal,
      *frame.t_camera2gimbal);
  }
  if (frame.pose_source == RuntimePoseSource::gimbal_to_world_matrix) {
    solver_.set_R_gimbal2world_matrix(frame.R_gimbal2world);
  } else {
    solver_.set_R_gimbal2world(frame.imu_quaternion);
  }
  tracker_.set_enemy_color_from_camp(frame.camp);
  return tracker_.track(armors, frame.timestamp);
}

Plan Runtime::plan(std::optional<Target> target, float bullet_speed, float pitch)
{
  return planner_.plan(std::move(target), bullet_speed, pitch);
}

Plan Runtime::process(std::list<Armor> & armors, const RuntimeFrame & frame)
{
  return process_with_debug(armors, frame).plan;
}

RuntimeDebugResult Runtime::process_with_debug(std::list<Armor> & armors, const RuntimeFrame & frame)
{
  auto targets = track(armors, frame);
  RuntimeDebugResult result;
  result.target = targets.empty() ? std::optional<Target>{} : std::optional<Target>{targets.front()};
  result.plan = plan(result.target, frame.bullet_speed, frame.pitch);
  result.aim_xyza = planner_.debug_xyza;
  return result;
}

Solver & Runtime::solver() { return solver_; }
}  // namespace auto_aim
