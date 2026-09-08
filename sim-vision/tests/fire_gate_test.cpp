#include <cassert>
#include <chrono>
#include <optional>

#include "tasks/auto_aim/fire_gate.hpp"

using Clock = std::chrono::steady_clock;

namespace
{

auto_aim::FireGateSample sample(
  Clock::time_point t0, int milliseconds, uint64_t observation_seq,
  std::optional<double> prediction_change_deg)
{
  const auto now = t0 + std::chrono::milliseconds(milliseconds);
  auto_aim::FireGateSample value;
  value.now = now;
  value.prediction_change_deg = prediction_change_deg;
  value.tracking.track_epoch = 1;
  value.tracking.observation_seq = observation_seq;
  value.tracking.observed_frames = static_cast<int>(observation_seq);
  value.tracking.confirmed = true;
  value.tracking.live_observation = true;
  value.tracking.first_observed = t0;
  value.tracking.last_observed = now;
  return value;
}

}  // namespace

int main()
{
  const auto t0 = Clock::now();
  auto_aim::FireGate gate(auto_aim::FireGateMode::enforce, 0.20, 3, 0.25);

  gate.update(sample(t0, 0, 1, std::nullopt));
  assert(!gate.ready());
  assert(!gate.allow_fire());

  // 100 Hz 对同一份Target重复规划，不能虚假增加稳定观测数。
  for (int i = 1; i <= 10; i++) gate.update(sample(t0, i * 5, 1, std::nullopt));
  assert(gate.debug().stable_observations == 0);

  gate.update(sample(t0, 210, 2, 0.10));
  gate.update(sample(t0, 220, 3, 0.10));
  gate.update(sample(t0, 230, 4, 0.10));
  assert(gate.ready());
  assert(gate.allow_fire());

  // 一次明显的预测变化必须立即关闭击发并重新预热。
  gate.update(sample(t0, 240, 5, 0.40));
  assert(!gate.ready());
  assert(gate.debug().state == auto_aim::FireGateState::warmup);

  gate.update(sample(t0, 450, 6, 0.10));
  gate.update(sample(t0, 460, 7, 0.10));
  gate.update(sample(t0, 470, 8, 0.10));
  assert(gate.ready());

  // temp_lost立即禁火；短暂恢复后只重新确认稳定观测，不重复完整预热。
  auto lost = sample(t0, 480, 8, std::nullopt);
  lost.tracking.temp_lost = true;
  lost.tracking.live_observation = false;
  gate.update(lost);
  assert(!gate.ready());
  assert(gate.debug().state == auto_aim::FireGateState::temp_lost);

  gate.update(sample(t0, 490, 9, 0.10));
  gate.update(sample(t0, 500, 10, 0.10));
  gate.update(sample(t0, 510, 11, 0.10));
  assert(gate.ready());

  auto stale = sample(t0, 900, 11, std::nullopt);
  stale.tracking.last_observed = t0 + std::chrono::milliseconds(510);
  gate.update(stale);
  assert(!gate.ready());
  assert(gate.debug().state == auto_aim::FireGateState::observation_stale);

  // shadow模式只记录成熟度，不拦截原始开火条件。
  auto_aim::FireGate shadow(auto_aim::FireGateMode::shadow, 0.20, 3, 0.25);
  shadow.update_no_target(t0);
  assert(!shadow.ready());
  assert(shadow.allow_fire());

  return 0;
}
