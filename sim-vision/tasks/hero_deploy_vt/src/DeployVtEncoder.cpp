#include "hero_deploy_vt/DeployVtEncoder.hpp"
#include <gst/gstinfo.h>
#include <gst/gstobject.h>
#include "hero_deploy_vt/DeployVtParam.hpp"
#include "hero_deploy_vt/TypeRedef.hpp"
#include "hero_deploy_vt/conf.hpp"

#include "tools/logger.hpp"

#include "gst/app/app.h"
#include "gst/gst.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>

using namespace cv;
using namespace std;
using namespace std::chrono_literals;

// clang-format off

#ifndef __FILE_NAME__
#  include <string.h>
#  define __FILE_NAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#if defined(USE_OLD_VAENC) && USE_OLD_VAENC == 1
# define VAENC_ELEM_NAME "vaapih265enc"
#else
# define VAENC_ELEM_NAME "vah265enc"
#endif

// clang-format on

#define cliLogLn(msg) std::cerr << "[" << __FILE_NAME__ << ":" << __LINE__ << "] " << msg << "\n";

namespace hdvt {
DeployVtEncoder::DeployVtEncoder(const DeployVtParam& param) : param(param)
{
	clampParam();

	frameInterval = chrono::duration_cast<NanoSec>(1.0s / param.targetFps);

	tools::logger()->info("Frame Interval: {}", frameInterval.count());

	packetBuffer.reserve(2 * param.packetSize);

	initPipeline();
}

DeployVtEncoder::~DeployVtEncoder()
{
	deInitGst();
}

void DeployVtEncoder::deInitGst()
{
	if (fixedPipe) { gst_element_set_state(fixedPipe, GST_STATE_NULL); }

	if (fixedBus) {
		gst_object_unref(fixedBus);
		fixedBus = nullptr;
	}

	if (fixedPipe) {
		gst_object_unref(fixedPipe);
		fixedPipe = nullptr;
	}
}

static void onGstLog(
		GstDebugCategory* cat,
		GstDebugLevel     lv,
		const gchar*      file,
		const gchar*      function,
		gint              line,
		GObject*          obj,
		GstDebugMessage*  msg,
		gpointer          user_data
)
{
	if (lv > gst_debug_category_get_threshold(cat)) { return; }

	auto getMsg = [msg] {
		auto message = gst_debug_message_get(msg);
		return message ? message : "<none>";
	};

	auto getCat = [cat] {
		auto cata = gst_debug_category_get_name(cat);
		return cata ? cata : "<none>";
	};

	auto filename = file ? file : "<unknown>";

	auto objName = (obj || GST_IS_OBJECT(obj)) ? GST_OBJECT_NAME(obj) : "";
	if (!objName) { objName = "<unknown>"; }

	switch (lv) {
		case GST_LEVEL_ERROR:
			tools::logger()->error(
					"[GST:{}] [{}:{}] From {}: {}", getCat(), filename, line, objName, getMsg()
			);
			break;
		case GST_LEVEL_WARNING:
		case GST_LEVEL_FIXME:
			tools::logger()->warn(
					"[GST:{}] [{}:{}] From {}: {}", getCat(), filename, line, objName, getMsg()
			);
			break;
		case GST_LEVEL_INFO:
			tools::logger()->info(
					"[GST:{}] [{}:{}] From {}: {}", getCat(), filename, line, objName, getMsg()
			);
			break;
		case GST_LEVEL_DEBUG:
		case GST_LEVEL_LOG:
			tools::logger()->debug(
					"[GST:{}] [{}:{}] From {}: {}", getCat(), filename, line, objName, getMsg()
			);
			break;
		case GST_LEVEL_TRACE:
			tools::logger()->trace(
					"[GST:{}] [{}:{}] From {}: {}", getCat(), filename, line, objName, getMsg()
			);
			break;
		default:
			break;
	}
};

static void initGstLog()
{
	gst_debug_remove_log_function(gst_debug_log_default);
	gst_debug_add_log_function(onGstLog, nullptr, nullptr);
}

void DeployVtEncoder::initEncoderRuntime()
{
	static once_flag initFlag;

	call_once(initFlag, [] {
		gst_init(nullptr, nullptr);
		initGstLog();
	});
}

void DeployVtEncoder::clampParam()
{
	param.targetFps               = std::clamp(param.targetFps, 1u, 60u);
	param.targetBitrate           = std::clamp(param.targetBitrate, 1u, 100u);
	param.motionTrailFrames       = std::clamp(param.motionTrailFrames, 0, 15);
	param.trailDisableMotionRatio = std::clamp(param.trailDisableMotionRatio, 0.0, 1.0);
	param.motionErodePx           = std::clamp(param.motionErodePx, 0, 20);
	param.motionDilatePx          = std::clamp(param.motionDilatePx, 0, 20);
	param.encodeQuality           = std::clamp(param.encodeQuality, 1u, 7u);
	param.maxPacketQueueSize      = std::max(1, param.maxPacketQueueSize);
	param.packetSize              = std::max(1, param.packetSize);
}

Mat DeployVtEncoder::preprocess(
		const cv::Mat& input, cv::Mat* roiDownsample, cv::Mat* staticRemoved
)
{
	const i32 w = std::min(param.cropSize, input.cols);
	const i32 h = std::min(param.cropSize, input.rows);

	i32 x = (input.cols - w) / 2 + param.cropXOffsetPx;
	i32 y = (input.rows - h) / 2;

	x = std::clamp(x, 0, input.cols - w);
	y = std::clamp(y, 0, input.rows - h);

	Mat cropped = input(cv::Rect(x, y, w, h));
	Mat resized;
	cv::resize(cropped, resized, cv::Size(param.targetSize, param.targetSize), 0, 0, INTER_LINEAR);

	if (roiDownsample != nullptr) { resized.copyTo(*roiDownsample); }

	Mat working = resized;
	if (param.forceMonochrome) {
		Mat grayFull;
		cvtColor(working, grayFull, COLOR_BGR2GRAY);
		cvtColor(grayFull, working, COLOR_GRAY2BGR);
	}

	if (!param.staticSimplify) {
		if (staticRemoved != nullptr) { working.copyTo(*staticRemoved); }
		return working;
	}

	Mat gray;
	cvtColor(working, gray, COLOR_BGR2GRAY);
	if (bgGrayF32.empty()) {
		gray.convertTo(bgGrayF32, CV_32F);
		return working;
	}

	Mat bgU8;
	convertScaleAbs(bgGrayF32, bgU8);

	Mat diff;
	absdiff(gray, bgU8, diff);

	Mat motionMask;
	threshold(diff, motionMask, param.motionThres, 255, THRESH_BINARY);
	if (param.motionErodePx > 0) {
		if (motionErodeKernel.empty()) {
			const int k       = 2 * param.motionErodePx + 1;
			motionErodeKernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(k, k));
		}
		erode(motionMask, motionMask, motionErodeKernel, cv::Point(-1, -1), 1);
	}
	if (param.motionDilatePx > 0) {
		if (motionDilateKernel.empty()) {
			const int k        = 2 * param.motionDilatePx + 1;
			motionDilateKernel = cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(k, k));
		}
		dilate(motionMask, motionMask, motionDilateKernel, cv::Point(-1, -1), 1);
	}

	const double motionRatioRaw =
			static_cast<f64>(countNonZero(motionMask)) / static_cast<f64>(motionMask.total());

	const bool suppressTrail = (motionRatioRaw >= param.trailDisableMotionRatio);

	if (param.centerClearSize > 0) {
		const i32 clearHeight = std::min(param.centerClearSize, working.rows);
		const i32 y0          = std::max(0, working.rows / 2 - clearHeight / 2);
		const i32 ch          = std::min(clearHeight, working.rows - y0);
		cv::rectangle(motionMask, cv::Rect(0, y0, working.cols, ch), cv::Scalar(255), cv::FILLED);
	}

	Mat staticBase = working.clone();
	if (!param.forceMonochrome &&
			(param.targetBitrate <= param.bgMonochromeBitrateThres || param.forceBgMonochrome)) {
		Mat grayBg;
		cvtColor(staticBase, grayBg, COLOR_BGR2GRAY);
		cvtColor(grayBg, staticBase, COLOR_GRAY2BGR);
	}

	Mat blurredStatic;
	GaussianBlur(
			staticBase,
			blurredStatic,
			cv::Size(),
			std::max(0.0, param.bgBlurSigma),
			std::max(0.0, param.bgBlurSigma)
	);

	Mat focused = blurredStatic.clone();
	working.copyTo(focused, motionMask);
	if (staticRemoved) { focused.copyTo(*staticRemoved); }

	if (param.motionTrailFrames > 0) {
		motionMaskHistory.push_back(motionMask.clone());
		trailFrameHistory.push_back(working.clone());

		const size_t maxHistory = static_cast<size_t>(param.motionTrailFrames + 1);
		while (motionMaskHistory.size() > maxHistory) { motionMaskHistory.pop_front(); }
		while (trailFrameHistory.size() > maxHistory) { trailFrameHistory.pop_front(); }

		const size_t historySize = motionMaskHistory.size();
		if (!suppressTrail && historySize > 1 && historySize == trailFrameHistory.size()) {
			Mat trailMask = motionMask.clone();
			Mat trailImg  = working.clone();

			for (size_t i = 0; i < historySize - 1; ++i) {
				bitwise_or(trailMask, motionMaskHistory[i], trailMask);
				cv::max(trailImg, trailFrameHistory[i], trailImg);
			}
			trailImg.copyTo(focused, trailMask);
		}
	} else {
		motionMaskHistory.clear();
		trailFrameHistory.clear();
	}

	accumulateWeighted(gray, bgGrayF32, std::clamp(param.bgUpdateAlpha, 0.001, 0.2));
	return focused;
}

int DeployVtEncoder::onNewSample(GstElement* appsink, gpointer udata)
{
	if (!appsink || !udata) {
		tools::logger()->error("Invalid gst appsink or user data pointer");

		return GST_FLOW_ERROR;
	}

	auto self = static_cast<DeployVtEncoder*>(udata);

	g_autoptr(GstSample) sample = gst_app_sink_pull_sample(GST_APP_SINK(self->fixedSink));

	if (!sample) {
		tools::logger()->warn("Failed to pull new gst sample from gst appsink");

		return GST_FLOW_OK;
	}

	GstBuffer* buf = gst_sample_get_buffer(sample);

	if (!buf) {
		tools::logger()->warn("Failed to get gst buffer from the gst sample");

		return GST_FLOW_OK;
	}

	GstMapInfo gMap;
	if (!gst_buffer_map(buf, &gMap, GST_MAP_READ)) {
		tools::logger()->warn("Failed to make a snapshot from the gst buffer");

		return GST_FLOW_OK;
	}

	{
		lock_guard<mutex> lock(self->queueMtx);

		self->stats.totalBytesEncoded.fetch_add(gMap.size, memory_order_relaxed);
		self->packetBuffer.insert(self->packetBuffer.end(), gMap.data, gMap.data + gMap.size);
		gst_buffer_unmap(buf, &gMap);

		const u64 maxBacklogSize = self->param.maxBacklogSize;

		if (self->packetBuffer.size() > maxBacklogSize) {
			const u64 targetDrop = self->packetBuffer.size() - maxBacklogSize;
			u64       dropBytes  = targetDrop;

			for (size_t i = targetDrop; i + 4 < self->packetBuffer.size(); ++i) {
				const bool hasStartCode3 =
						(self->packetBuffer[i] == 0 && self->packetBuffer[i + 1] == 0 &&
						 self->packetBuffer[i + 2] == 1);

				const bool hasStartCode4 =
						(self->packetBuffer[i] == 0 && self->packetBuffer[i + 1] == 0 &&
						 self->packetBuffer[i + 2] == 0 && self->packetBuffer[i + 3] == 1);

				if (hasStartCode3 || hasStartCode4) {
					dropBytes = i;
					break;
				}
			}

			self->packetBuffer.erase(self->packetBuffer.begin(), self->packetBuffer.begin() + dropBytes);

			self->stats.droppedBytes.fetch_add(dropBytes, memory_order_relaxed);
			self->stats.dropEventCount.fetch_add(1, memory_order_relaxed);
		}

		// if (self->packetQueue.size() >= static_cast<size_t>(self->param.maxPacketQueueSize)) {
		// 	self->stats.droppedPkt.fetch_add(self->packetQueue.size(), std::memory_order_relaxed);
		// 	self->stats.dropEventCount.fetch_add(1, memory_order_relaxed);
		// 	self->stats.pktQueueDepth.store(0, memory_order_relaxed);

		// 	self->packetQueue.clear();
		// }

		// while (self->packetBuffer.size() >= static_cast<size_t>(self->param.packetSize)) {
		// 	self->packetQueue.emplace_back(
		// 			self->packetBuffer.begin(), self->packetBuffer.begin() + self->param.packetSize
		// 	);

		// 	self->packetBuffer.erase(
		// 			self->packetBuffer.begin(), self->packetBuffer.begin() + self->param.packetSize
		// 	);

		// 	self->stats.pktProduced.fetch_add(1, memory_order_relaxed);
		// }

		// self->stats.pktQueueDepth.store(self->packetQueue.size(), memory_order_relaxed);
	}

	return GST_FLOW_OK;
}

bool DeployVtEncoder::tryPullPacket(vector<u8>& dst)
{
	pollBus();

	if (param.useFileSink) { return false; }

	lock_guard<mutex> lock(queueMtx);

	if (packetBuffer.size() < static_cast<u64>(param.packetSize)) { return false; }

	dst.assign(packetBuffer.begin(), packetBuffer.begin() + param.packetSize);
	packetBuffer.erase(packetBuffer.begin(), packetBuffer.begin() + param.packetSize);

	stats.pktPulled.fetch_add(1, memory_order_relaxed);

	// if (packetQueue.empty()) { return false; }

	// dst = std::move(packetQueue.front());
	// packetQueue.pop_front();

	// stats.pktQueueDepth.store(packetQueue.size(), memory_order_relaxed);

	return true;
}

void DeployVtEncoder::pollBus()
{
	if (!fixedBus || !pollBusMsg.load()) { return; }

	g_autoptr(GstMessage) msg = gst_bus_pop_filtered(
			fixedBus,
			static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING | GST_MESSAGE_EOS)
	);

	if (msg) {
		switch (GST_MESSAGE_TYPE(msg)) {
			case GST_MESSAGE_EOS:
				tools::logger()->info("End of stream detected");
				break;
			case GST_MESSAGE_ERROR: {
				g_autoptr(GError) err     = nullptr;
				g_autofree gchar* debInfo = nullptr;

				gst_message_parse_error(msg, &err, &debInfo);

				if (err) {
					tools::logger()->error(
							"{}, Debug Info: {}", err->message, (debInfo ? string(debInfo) : "<none>")
					);
				}
				break;
			}
			case GST_MESSAGE_WARNING: {
				g_autoptr(GError) err     = nullptr;
				g_autofree gchar* debInfo = nullptr;

				gst_message_parse_warning(msg, &err, &debInfo);

				if (err) {
					tools::logger()->warn(
							"{}, Debug Info: {}", err->message, (debInfo ? string(debInfo) : "<none>")
					);
				}
				break;
			}
			default:
				break;
		}
	}
}

auto DeployVtEncoder::tryPushFrame(const Mat& frame, TimePoint pushTime) -> PushStatus
{
	pollBus();

	if (frame.empty()) { return PushStatus::PREPROCESSED_FAILED; }

	if (!param.noFrameSampling && (pushTime - lastPushTime < frameInterval * 0.9)) {
		// cliLogLn("Inv: " + to_string((pushTime - lastPushTime).count()) + " < " + to_string((frameInterval * 0.8).count()));

		return PushStatus::OMITTED;
	}

	lastPushTime = pushTime;

	cv::Mat processed = preprocess(frame);
	if (processed.empty()) { return PushStatus::PREPROCESSED_FAILED; }

	GstBuffer* buffer =
			gst_buffer_new_memdup(processed.data, processed.total() * processed.elemSize());

	if (buffer) {
		auto ret = gst_app_src_push_buffer(GST_APP_SRC(fixedSrc), buffer);
		if (ret == GST_FLOW_OK) { return PushStatus::SUCCESS; }
	}

	return PushStatus::PUSH_FAILED;
}

void DeployVtEncoder::initPipeline()
{
	fixedPipe = gst_pipeline_new("fixedPipe");
	fixedSrc  = gst_element_factory_make("appsrc", "fixedSrc");
	fixedSink = param.useFileSink ? gst_element_factory_make("filesink", "fixedSink")
																: gst_element_factory_make("appsink", "fixedSink");

	ElemRawPtr vconv           = gst_element_factory_make("videoconvert", "vconv");
	ElemRawPtr vconvCapsFilter = gst_element_factory_make("capsfilter", "vconvCapsFilter");
	ElemRawPtr encoder         = !param.useX265 ? gst_element_factory_make(VAENC_ELEM_NAME, "encoder")
																							: gst_element_factory_make("x265enc", "encoder");
	ElemRawPtr parser          = gst_element_factory_make("h265parse", "parser");

	if (!fixedPipe || !fixedSrc || !fixedSink || !vconv || !vconvCapsFilter || !encoder || !parser) {
		throw runtime_error("Error: Failed to create all GStreamer pipeline elements");
	}

	gst_bin_add_many(
			GST_BIN(fixedPipe), fixedSrc, vconv, vconvCapsFilter, encoder, parser, fixedSink, nullptr
	);

	if (!gst_element_link_many(
					fixedSrc, vconv, vconvCapsFilter, encoder, parser, fixedSink, nullptr
			)) {
		throw runtime_error("Error: Failed to link all Gstreamer pipeline elements");
	}

	g_autoptr(GstCaps) inCaps = gst_caps_new_simple(
			"video/x-raw",
			"format",
			G_TYPE_STRING,
			"BGR",
			"width",
			G_TYPE_INT,
			param.targetSize,
			"height",
			G_TYPE_INT,
			param.targetSize,
			"framerate",
			GST_TYPE_FRACTION,
			param.targetFps,
			1,
			nullptr
	);

	g_object_set(
			fixedSrc,
			"is-live",
			TRUE,
			"min-latency",
			0,
			"max-latency",
			-1,
			"do-timestamp",
			TRUE,
			"format",
			GST_FORMAT_TIME,
			"caps",
			inCaps,
			"stream-type",
			GST_APP_STREAM_TYPE_STREAM,
			nullptr
	);

	string typeStr;

	param.useX265 ? typeStr = "I420" : typeStr = "NV12";

	g_autoptr(GstCaps) vConvOutCaps =
			gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, typeStr.c_str(), nullptr);
	g_object_set(vconvCapsFilter, "caps", vConvOutCaps, nullptr);

	if (!param.useX265) {
		if constexpr (useOldVAEnc) {
			g_object_set(
					encoder,
					"trellis",
					TRUE,
					"bitrate",
					param.targetBitrate,
					"default-roi-delta-qp",
					0,
					"cpb-length",
					param.cpbLength,
					"keyframe-period",
					param.gopSize,
					"max-bframes",
					param.bFramesCount,
					"max-qp",
					param.maxQp,
					"min-qp",
					param.minQp,
					"target-percentage",
					param.bitratePercentage,
					"quality-factor",
					param.rcQualityFactor,
					"quality-level",
					param.encodeQuality,
					"refs",
					param.refFrames,
					"rate-control",
					static_cast<guint>(param.rcMode),
					nullptr
			);
		} else {
			g_object_set(
					encoder,
					"aud",
					TRUE,
					"trellis",
					TRUE,
					"b-frames",
					param.bFramesCount,
					"key-int-max",
					param.gopSize,
					"bitrate",
					param.targetBitrate,
					"qpi",
					param.rcQualityFactor,
					"rate-control",
					static_cast<guint>(param.rcMode),
					"target-percentage",
					param.bitratePercentage,
					"target-usage",
					param.encodeQuality,
					"max-qp",
					param.maxQp,
					"min-qp",
					param.minQp,
					"ref-frames",
					param.refFrames,
					nullptr
			);
		}
	} else {
		g_object_set(
				encoder,
				"bitrate",
				param.targetBitrate,
				"speed-preset",
				1,
				"tune",
				4,  // 4 = zerolatency
				"key-int-max",
				param.gopSize,
				"option-string",
				"intra-refresh=1:repeat-headers=1:scenecut=0:ref=4:bframes=0:aud=1:rc-lookahead=4",
				nullptr
		);
	}

	g_object_set(parser, "config-interval", -1, "disable-passthrough", TRUE, nullptr);

	if (!param.useFileSink) {
		g_autoptr(GstCaps) outCaps = gst_caps_new_simple(
				"video/x-h265",
				"stream-format",
				G_TYPE_STRING,
				"byte-stream",
				"alignment",
				G_TYPE_STRING,
				"au",
				nullptr
		);

		g_object_set(
				fixedSink,
				"caps",
				outCaps,
				"max-buffers",
				5,
				"emit-signals",
				TRUE,
#if GST_VERSION_MAJOR >= 1 && GST_VERSION_MINOR >= 28
				"leaky-type",
				GST_APP_LEAKY_TYPE_DOWNSTREAM,
#else
				"drop",
				TRUE,
#endif
				"sync",
				FALSE,
				nullptr
		);

		g_signal_connect(fixedSink, "new-sample", G_CALLBACK(onNewSample), this);
	} else {
		g_object_set(fixedSink, "location", "./out.h265", nullptr);
	}

	fixedBus = gst_element_get_bus(fixedPipe);
}

bool DeployVtEncoder::start()
{
	if (!fixedPipe) {
		tools::logger()->error("Pipeline didn't initialize");

		return false;
	}

	auto ret = gst_element_set_state(fixedPipe, GST_STATE_PLAYING);
	if (ret == GST_STATE_CHANGE_FAILURE) {
		tools::logger()->error("Error: Failed to start pipeline");

		return false;
	}

	return true;
}

bool DeployVtEncoder::pause()
{
	if (!fixedPipe) {
		tools::logger()->error("Pipeline didn't initialize");

		return false;
	}

	auto ret = gst_element_set_state(fixedPipe, GST_STATE_PAUSED);
	if (ret == GST_STATE_CHANGE_FAILURE) {
		tools::logger()->error("Failed to pause pipeline");

		return false;
	}

	return true;
}

bool DeployVtEncoder::stop()
{
	if (!fixedPipe) {
		tools::logger()->error("Pipeline didn't initialize");

		return false;
	}

	auto ret = gst_element_set_state(fixedPipe, GST_STATE_NULL);
	if (ret == GST_STATE_CHANGE_FAILURE) {
		tools::logger()->error("Failed to stop pipeline");

		return false;
	}

	return true;
}
}  // namespace hdvt
