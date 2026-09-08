#include "hero_deploy_vt/DeployVtSender.hpp"

#include "hero_deploy_vt/DeployVtEncoder.hpp"
#include "hero_deploy_vt/conf.hpp"

#include <gst/gst.h>

#include "io/gimbal/gimbal.hpp"
#include "tools/logger.hpp"

#include <algorithm>
#include <atomic>
#include <concepts>
#include <exception>
#include <string_view>
#include <thread>
#include <type_traits>

using namespace std;

namespace hdvt {
class DeployVtSender::Impl
{
private:
  using TimePoint = DeployVtEncoder::TimePoint;

  std::atomic<bool> _inited{ false };

  std::atomic<bool> showTxProfLog{ true };
  std::atomic<i32>  txInvMs{ 20 };

  io::Gimbal& serialDevice;

  DeployVtParam param;

public:
  DeployVtEncoder::SharedPtr encoder;

private:
  std::jthread txThread;

private:
  void initTxThread();

public:
  bool inited() const { return _inited.load(); }

public:
  Impl(const YAML::Node& _config, io::Gimbal& _serialDevice);
  ~Impl();

  Impl()                       = delete;
  Impl(const Impl&)            = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&)                 = delete;
  Impl& operator=(Impl&&)      = delete;
};

static string_view rcModEnumToStr(DeployVtParam::RateControlMode mode)
{
  switch (mode) {
    case DeployVtParam::RateControlMode::CBR:
      return "CBR";
    case DeployVtParam::RateControlMode::VBR:
      return "VBR";
#if defined(USE_OLD_VAENC) && USE_OLD_VAENC == 0
    case DeployVtParam::RateControlMode::VCM:
      return "VCM";
#endif
    case DeployVtParam::RateControlMode::QVBR:
      return "QVBR";
    default:
      return "Unknown";
  }
}

static DeployVtParam::RateControlMode strToRcModeEnum(const string& str)
{
  if (str == "CBR") {
    return DeployVtParam::RateControlMode::CBR;
  } else if (str == "VBR") {
    return DeployVtParam::RateControlMode::VBR;
  } else if (str == "VCM") {
    return DeployVtParam::RateControlMode::VCM;
  } else if (str == "QVBR") {
    return DeployVtParam::RateControlMode::QVBR;
  } else {
#if defined(USE_OLD_VAENC) && USE_OLD_VAENC == 1
    return DeployVtParam::RateControlMode::QVBR;
#else
    return DeployVtParam::RateControlMode::VCM;
#endif
  }
}

template<typename T>
concept Numeric = YAML::is_numeric<std::remove_cvref_t<T>>::value == true;

template<Numeric T, Numeric Low, Numeric High>
  requires convertible_to<Low, T> && convertible_to<High, T>
static void readFromConf(T& target, const std::string& key, Low lo, High hi, const YAML::Node& conf)
{
  auto keyNode = conf[key];
  if (!keyNode || keyNode.IsNull()) {
    tools::logger()->warn("[Deploy VT] Key '{}' not found, keep default value", key);
    return;
  }

  try {
    T val = tools::read<T>(conf, key);

    T loVal = static_cast<T>(lo);
    T hiVal = static_cast<T>(hi);

    if (loVal > hiVal) {
      tools::logger()->warn("[Deploy VT] Invalid range for '{}'", key);
      return;
    }

    T clamped = clamp(val, loVal, hiVal);

    if (val != clamped) {
      tools::logger()->warn(
          "[Deploy VT] Key '{}' is out of range [{}, {}], clamp to {}",
          key,
          val,
          loVal,
          hiVal,
          clamped
      );
    } else {
      tools::logger()->info("[Deploy VT] Key '{}' = '{}'", key, clamped);
    }

    target = clamped;
  } catch (const exception& e) {
    tools::logger()->warn(
        "[Deploy VT] Failed to read key '{}': {}, keep default value", key, e.what()
    );
  }
}

template<typename T>
static void readFromConf(T& target, string key, const YAML::Node& conf)
{
  auto keyNode = conf[key];
  if (!keyNode || keyNode.IsNull()) {
    tools::logger()->warn("[Deploy VT] Key '{}' not found, keep default value", key);
    return;
  }

  try {
    target = tools::read<T>(conf, key);

    tools::logger()->info("[Deploy VT] key '{}' = '{}'", key, target);
  } catch (const exception& e) {
    tools::logger()->warn(
        "[Deploy VT] Failed to read key '{}': {}, keep default value", key, e.what()
    );
  }
}

DeployVtSender::Impl::Impl(const YAML::Node& _config, io::Gimbal& _serialDevice) :
    serialDevice(_serialDevice)
{
  if (!_config || _config.IsNull()) {
    tools::logger()->warn("[Deploy VT] Failed to construct DeployVtSender: No config passed");
    return;
  }

  YAML::Node deployConf = _config["deploy_vt_param"];

  if (!deployConf || deployConf.IsNull()) {
    tools::logger()->warn(
        "[Deploy VT] Failed to construct DeployVtSender: No 'deploy_vt_param' field"
    );
    return;
  }

  DeployVtParam p = defaultVtParam;

  readFromConf(p.cropSize, "crop_size", 1, 20'000, deployConf);
  readFromConf(p.cropXOffsetPx, "crop_x_offset_px", -20'000, 20'000, deployConf);
  readFromConf(p.targetSize, "target_size", 1, 20'000, deployConf);
  readFromConf(p.centerClearSize, "center_clear_size", 1, 20'000, deployConf);

  readFromConf(p.targetFps, "target_fps", 30, 60, deployConf);
  readFromConf(p.noFrameSampling, "no_frame_sampling", deployConf);
  readFromConf(p.targetBitrate, "target_bitrate", 30, 100, deployConf);
  readFromConf(p.bitratePercentage, "bitrate_percentage", 1, 100, deployConf);
  readFromConf(p.noFrameSampling, "no_frame_sampling", deployConf);
  readFromConf(p.bFramesCount, "b_frames_count", 0, 8, deployConf);
  readFromConf(p.gopSize, "gop_size", 0.5 * p.targetFps, 4.0 * p.targetFps, deployConf);
  readFromConf(p.maxQp, "max_qp", 1, 51, deployConf);
  readFromConf(p.minQp, "min_qp", 1, 51, deployConf);
  readFromConf(p.encodeQuality, "encode_quality", 1, 7, deployConf);
  readFromConf(p.refFrames, "ref_frames", 1, 15, deployConf);
  readFromConf(p.rcQualityFactor, "rc_quality_factor", 1, 51, deployConf);

  if constexpr (useOldVAEnc) {
    auto vaConfNode = deployConf["old_vaenc"];
    if (!vaConfNode || vaConfNode.IsNull()) {
      tools::logger()->warn("[Deploy VT] Key 'old_vaenc' not found");
      return;
    }

    readFromConf(p.cpbLength, "cpb_length", 1, 1'0000, vaConfNode);

    string rcModeStr;
    readFromConf(rcModeStr, "rc_mode", vaConfNode);
    p.rcMode = strToRcModeEnum(rcModeStr);
  } else {
    auto vaConfNode = deployConf["new_vaenc"];
    if (!vaConfNode || vaConfNode.IsNull()) {
      tools::logger()->warn("[Deploy VT] Key 'new_vaenc' not found");
      return;
    }

    readFromConf(p.cpbLength, "cpb_length", 0, 204'8000, vaConfNode);

    string rcModeStr;
    readFromConf(rcModeStr, "rc_mode", vaConfNode);
    p.rcMode = strToRcModeEnum(rcModeStr);
  }

  readFromConf(p.staticSimplify, "static_simplify", deployConf);
  readFromConf(p.bgMonochromeBitrateThres, "bg_mono_bitrate_thres", 30, 100, deployConf);
  readFromConf(p.forceBgMonochrome, "force_bg_mono", deployConf);
  readFromConf(p.motionThres, "motion_thres", deployConf);
  readFromConf(p.motionErodePx, "motion_erode_px", 0, 20, deployConf);
  readFromConf(p.motionDilatePx, "motion_dilate_px", 0, 20, deployConf);
  readFromConf(p.motionTrailFrames, "motion_trail_frames", 0, 15, deployConf);
  readFromConf(p.trailDisableMotionRatio, "trail_disable_motion_ratio", 0.0, 1.0, deployConf);
  readFromConf(p.bgUpdateAlpha, "bg_update_alpha", deployConf);
  readFromConf(p.bgBlurSigma, "bg_blur_sigma", deployConf);
  readFromConf(p.forceMonochrome, "force_mono", deployConf);

  showTxProfLog.store(tools::read<bool>(deployConf, "show_tx_prof_log"));

  DeployVtEncoder::initEncoderRuntime();

  try {
    encoder = DeployVtEncoder::create(p);
  } catch (const exception& e) {
    tools::logger()->error("[Deploy VT] Failed to construct DeployVtEncoder: {}", e.what());
  }

  encoder->start();

  _inited.store(true);

  initTxThread();
}

DeployVtSender::Impl::~Impl()
{
  txThread.request_stop();
  _inited.store(false);
}

void DeployVtSender::Impl::initTxThread()
{
  txThread = jthread([this](stop_token st) mutable {
    TimePoint lastStatTime = TimePoint{};
    TimePoint nextTxTime   = chrono::steady_clock::now();

    u64 lastStatEncBytes = 0;
    u64 txPktSent        = 0;

    auto enc = encoder;

    HeroVidPkt serialPkt;

    while (!st.stop_requested() && _inited.load()) {
      this_thread::sleep_until(nextTxTime);
      nextTxTime += chrono::milliseconds(txInvMs.load());

      vector<u8> pkt;
      if (enc->tryPullPacket(pkt)) {
        if (pkt.size() != 300) {
          tools::logger()->warn("[Deploy VT] Pulled a packet with invalid size: {}", pkt.size());
          continue;
        }

        for (int i = 0; i < 5; i++) {
          memcpy(serialPkt.payload, pkt.data() + i * 60, 60);
          serialDevice.sendRaw(reinterpret_cast<uint8_t*>(&serialPkt), sizeof(HeroVidPkt));
        }

        txPktSent++;
      } else {
        // if (serialDevice.mode() == io::GimbalMode::IDLE) {
        //   serialDevice.send(false, false, 0, 0, 0, 0, 0, 0);
        // }

        continue;
      }

      auto now = chrono::steady_clock::now();
      if (now - lastStatTime >= 1.0s && showTxProfLog.load()) {
        f64 inv_sec = chrono::duration_cast<chrono::duration<f64>>(now - lastStatTime).count();

        auto sView = enc->getStats();

        f64 encKbps =
            static_cast<f64>(sView.totalBytesEncoded - lastStatEncBytes) * 8.0 / inv_sec / 1000.0;

        tools::logger()->info(
            "[Deploy VT] [TX Prof] encBitrate={:.2f} kbps | droppedBytes={} | dropEvents={} | "
            "pktPulled={} | pktSent={}",
            encKbps,
            sView.droppedBytes,
            sView.dropEventCount,
            sView.pktPulled,
            txPktSent
        );

        lastStatTime     = now;
        lastStatEncBytes = sView.totalBytesEncoded;
      }
    }

    auto sView = enc->getStats();

    f64 totEncKB     = static_cast<f64>(sView.totalBytesEncoded) / 1000.0;
    f64 totSentKB    = static_cast<f64>(sView.pktPulled * param.packetSize) / 1000.0;
    f64 totDroppedKB = static_cast<f64>(sView.droppedBytes) / 1000.0;

    tools::logger()->info(
        "[Deploy VT] [Final] totalEncoded={}kB pktPulled={} totalSent={}kB "
        "droppedBytes={}kB dropEvents={}",
        totEncKB,
        sView.pktPulled,
        totSentKB,
        totDroppedKB,
        sView.dropEventCount
    );
  });
}

DeployVtSender::DeployVtSender(const YAML::Node& _config, io::Gimbal& _serialDevice)
{
  _impl = make_unique<Impl>(_config, _serialDevice);
}

DeployVtSender::~DeployVtSender() = default;

auto DeployVtSender::tryPushImg(const cv::Mat& img, TimePoint timepoint)
    -> DeployVtEncoder::PushStatus
{
  if (!_impl->inited()) { return DeployVtEncoder::PushStatus::PUSH_FAILED; }

  return _impl->encoder->tryPushFrame(img, timepoint);
}

bool DeployVtSender::isInit() const
{
  return _impl->inited();
}
}  // namespace hdvt
