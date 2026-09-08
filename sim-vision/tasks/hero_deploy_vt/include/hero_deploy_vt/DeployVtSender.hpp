#include "hero_deploy_vt/DeployVtEncoder.hpp"
#include "hero_deploy_vt/DeployVtParam.hpp"

#include "tools/yaml.hpp"

#include "io/gimbal/gimbal.hpp"

#include <chrono>
#include <opencv2/opencv.hpp>

#include <memory>

#define VID_PKT_SOP_LO  0x66
#define VID_PKT_SOP_HI  0x55
#define VID_PKT_EOP_LO  0x88
#define VID_PKT_EOP_HI  0x77
#define VID_PKT_PAY_LEN 60

namespace hdvt {
typedef struct __attribute__((packed)) __hero_vid_pkt_t
{
	uint8_t SOP_LO = VID_PKT_SOP_LO;
	uint8_t SOP_HI = VID_PKT_SOP_HI;
	uint8_t payload[VID_PKT_PAY_LEN];
	uint8_t EOP_LO = VID_PKT_EOP_LO;
	uint8_t EOP_HI = VID_PKT_EOP_HI;
} HeroVidPkt;

static_assert(sizeof(HeroVidPkt) == 64);

class DeployVtSender
{
public:
	using TimePoint = DeployVtEncoder::TimePoint;

private:
	class Impl;
	std::unique_ptr<Impl> _impl;

public:
	DeployVtEncoder::PushStatus tryPushImg(
			const cv::Mat& img, DeployVtEncoder::TimePoint timeStamp = std::chrono::steady_clock::now()
	);

	bool isInit() const;

public:
	DeployVtSender(const YAML::Node& _config, io::Gimbal& _serialDevice);
	~DeployVtSender();

	DeployVtSender()                                 = delete;
	DeployVtSender(const DeployVtSender&)            = delete;
	DeployVtSender& operator=(const DeployVtSender&) = delete;
	DeployVtSender(DeployVtSender&&)                 = delete;
	DeployVtSender& operator=(DeployVtSender&&)      = delete;
};
}  // namespace hdvt