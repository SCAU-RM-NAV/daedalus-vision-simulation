#pragma once

#include "hero_deploy_vt/DeployVtParam.hpp"

#include "opencv2/opencv.hpp"

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

extern "C"
{
	struct _GstElement;
	struct _GstPad;
	struct _GstBus;
	typedef struct _GstElement GstElement;
	typedef struct _GstPad     GstPad;
	typedef struct _GstBus     GstBus;
	typedef void*              gpointer;
}

namespace hdvt {
class DeployVtEncoder
{
public:
	using SharedPtr  = std::shared_ptr<DeployVtEncoder>;
	using UniPtr     = std::unique_ptr<DeployVtEncoder>;
	using PktHandler = std::function<void(const u8* data, size_t size)>;

	using TimePoint = std::chrono::steady_clock::time_point;
	using NanoSec   = std::chrono::nanoseconds;
	using Sec       = std::chrono::seconds;

public:
	enum class PushStatus : u32
	{
		OMITTED = 0,          // 忽略推送该帧 (当输入帧率大于目标帧率时)
		PREPROCESSED_FAILED,  // 预处理失败
		PUSH_FAILED,          // 推送 Gstreamer 管道失败
		SUCCESS               // 推送成功
	};

	static std::string_view pushStatusToStr(PushStatus status)
	{
		switch (status) {
			case PushStatus::OMITTED:
				return "OMITTED";
			case PushStatus::PREPROCESSED_FAILED:
				return "PREPROCESSED FAILED";
			case PushStatus::PUSH_FAILED:
				return "PUSH FAILED";
			case PushStatus::SUCCESS:
				return "SUCCESS";
			default:
				return "UNKNOWN";
		}
	}

	struct Stats
	{
		std::atomic<u64> totalBytesEncoded = 0;  // 当前总共编码的视频字节数
		std::atomic<u64> pktProduced       = 0;  // 当前总共产出的分片数量
		std::atomic<u64> pktPulled         = 0;  // 当前被拉取的分片数量
		// std::atomic<u64> droppedPkt        = 0;  // 当前被丢弃的分片数量
		std::atomic<u64> dropEventCount    = 0;  // 分片丢弃事件计数
		std::atomic<u64> droppedBytes      = 0;  // 当前丢弃的总字节数
		// std::atomic<size_t> pktQueueDepth     = 0;  // 分片缓冲队列当前深度
	};

	struct StatsView
	{
		u64 totalBytesEncoded = 0;
		// u64 pktProduced       = 0;
		u64 pktPulled         = 0;
		// u64 droppedPkt        = 0;
		u64 dropEventCount    = 0;
		u64 droppedBytes      = 0;
		// size_t pktQueueDepth     = 0;
	};

	/**
	 * @brief 获取 DeployVtEncoder 当前的状态统计
	 * 
	 * @return StatsView
	 * 
	 * @note 多线程安全
	 */
	StatsView getStats() const
	{
		return StatsView{ stats.totalBytesEncoded.load(std::memory_order_relaxed),
											stats.pktPulled.load(std::memory_order_relaxed),
											stats.dropEventCount.load(std::memory_order_relaxed),
											stats.droppedBytes.load(std::memory_order_relaxed) };
	}

private:
	using ElemRawPtr = GstElement*;

private:
	DeployVtParam param        = defaultVtParam;
	TimePoint     lastPushTime = TimePoint{};
	NanoSec       frameInterval;

private:
	cv::Mat bgGrayF32;
	cv::Mat motionErodeKernel;
	cv::Mat motionDilateKernel;

	std::atomic<bool> pollBusMsg{ true };
	Stats             stats;

	std::mutex      queueMtx;
	std::vector<u8> packetBuffer;
	// std::deque<std::vector<u8>> packetQueue;

	std::deque<cv::Mat> motionMaskHistory;
	std::deque<cv::Mat> trailFrameHistory;

	ElemRawPtr fixedPipe = nullptr;
	ElemRawPtr fixedSrc  = nullptr;
	ElemRawPtr fixedSink = nullptr;
	GstBus*    fixedBus  = nullptr;

	PktHandler handler;

private:
	cv::Mat preprocess(
			const cv::Mat& input, cv::Mat* roiDownsample = nullptr, cv::Mat* staticRemoved = nullptr
	);

public:
	/**
	 * @brief 尝试拉取渲染管线输出的分片
	 * 
	 * @param dst 
	 * @return true 拉取成功
	 * @return false 拉取失败
	 * 
	 * @note 多线程安全，分片的长度由 param.packetSize 决定，如无特殊需求，获取的分片无需再度
	 * 		 切片即可用 VideoPacket 信息包装并发布。该方法是非阻塞的。
	 */
	bool tryPullPacket(std::vector<u8>& dst);

	/**
	 * @brief 初始化 Gstreamer 运行时
	 * 
	 * @note 多线程安全，必须在所有的 DeployVtEncoder 实例 实例化之前调用一次，否则将导致
	 * 		 DeployVtEncoder 无法正常工作
	 */
	static void initEncoderRuntime();

private:
	void clampParam();

	void pollBus();

	void initPipeline();
	void deInitGst();

	static int onNewSample(GstElement* appsink, gpointer udata);

public:
	/**
	 * @brief 尝试向渲染管线推送帧
	 * 
	 * @param frame `cv::Mat`, 色彩空间要求为 BGR8
	 * @param pushTime `std::chrono::steady_clock`, 推送时间戳
	 * @return PushStatus 
	 * 
	 * @note 非多线程安全，请不要在不同线程并发地调用此函数。该方法是非阻塞的。
	 */
	PushStatus tryPushFrame(
			const cv::Mat& frame, TimePoint pushTime = std::chrono::steady_clock::now()
	);

	/**
	 * @brief 启动渲染管线
	 * 
	 * @note 非多线程安全，请不要在不同线程并发地调用此函数。
	 */
	bool start();

	/**
	 * @brief 暂停渲染管线
	 * 
	 * @note 非多线程安全，请不要在不同线程并发地调用此函数。
	 */
	bool pause();

	/**
	 * @brief 关闭渲染管线
	 * 
	 * @note 非多线程安全，请不要在不同线程并发地调用此函数。
	 * @note 调用此方法会从硬件层面上释放渲染管线资源，可以调用 DeployVtEncoder::start() 
	 * 	     重启管线
	 */
	bool stop();

	/**
	 * @brief 是否监听渲染管线事件
	 * 
	 * @param enable true 监听，false 不监听
	 * 
	 * @note 多线程安全
	 */
	void enablePollBusMsg(bool enable) { pollBusMsg.store(enable); }

public:
	/**
	 * @brief 创建新的 DeployVtEncoder 实例
	 * 
	 * @param param 
	 * 
	 * @throw `std::runtime_error` 无法创建或连接所有渲染管线元素时 
	 */
	explicit DeployVtEncoder(const DeployVtParam& param = defaultVtParam);

	/**
	 * @brief 创建新的 DeployVtEncoder 共享指针
	 * 
	 * @param param 
	 * 
	 * @return SharedPtr 
	 * 
	 * @throw `std::runtime_error` 无法创建或连接所有渲染管线元素时 
	 */
	[[nodiscard]] static SharedPtr create(const DeployVtParam& param = defaultVtParam)
	{ return std::make_shared<DeployVtEncoder>(param); }

	~DeployVtEncoder();

	DeployVtEncoder(const DeployVtEncoder&)            = delete;
	DeployVtEncoder& operator=(const DeployVtEncoder&) = delete;
	DeployVtEncoder(DeployVtEncoder&&)                 = delete;
	DeployVtEncoder& operator=(DeployVtEncoder&&)      = delete;
};
}  // namespace hdvt