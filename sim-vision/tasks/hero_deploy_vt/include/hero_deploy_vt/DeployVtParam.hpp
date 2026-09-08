#pragma once

#include "hero_deploy_vt/TypeRedef.hpp"
#include "hero_deploy_vt/conf.hpp"

namespace hdvt {
struct DeployVtParam
{
#if defined(USE_OLD_VAENC) && USE_OLD_VAENC == 1
	enum class RateControlMode : u32
	{
		UNKNOWN = 0,

		CBR  = 2,   // 固定码率
		VBR  = 4,   // 可变码率
		QVBR = 8,   // 基于质量的可变码率
		VCM  = CBR  // Alias of CBR
	};
#else
	enum class RateControlMode : u32
	{
		UNKNOWN = 0,

		CBR  = 2,       // 固定码率
		VBR  = 2 << 1,  // 可变码率
		VCM  = 2 << 2,  // Video Conferencing Mode (Non HRD compliant)
		QVBR = 2 << 9   // 基于质量的可变码率
	};
#endif

	// Some ref about CBR VBV and CPB:
	// https://wenchy.github.io/blogs/2015-12-11-What-are-CBR-VBV-and-CPB.html

	i32  cropSize        = 800;            // 输入裁剪尺寸 (ROI 尺寸)
	i32  cropXOffsetPx   = 100;            // ROI 相对居中位置的 X 偏移 (像素，正数向右)
	i32  targetSize      = 400;            // 输出缩放尺寸
	u32  targetFps       = 60;             // 目标编码帧率
	bool noFrameSampling = false;          // 禁用抽帧 (启用后不再主动限制帧率至 targetFps)
	u32  bFramesCount    = 0;              // 单 GOP 内 B 帧的数量
	u32  gopSize         = 2 * targetFps;  // GOP 长度
	u64  maxBacklogSize  = 45'000;         // 编码输出最大背压字节

#if defined(USE_OLD_VAENC) && USE_OLD_VAENC == 1
	// 码率控制缓冲区长度 (ms) (1 - 10000) (过低可能导致 VCM/QVBR 下码率不稳定)
	u32 cpbLength = 1000;
#else
	u32 cpbLength = 0;  // 0 = Auto calculate, 0 - 2048000
#endif

	u32 maxQp             = 45;  // 最大量化值
	u32 minQp             = 15;  // 最小量化值
	u32 encodeQuality     = 4;   // 编码质量 (越小越好/越慢) (1-7) (小于4有概率导致 pipeline 罢工)
	u32 refFrames         = 3;   // P 帧向前参考帧数
	u32 targetBitrate     = 90;  // 目标编码码率 (Kbps) (max = 120)
	u32 bitratePercentage = 66;  // 目标编码码率百分比 (仅 VBR 有效) (1-100)

	// VBR 下, realTargetBitrate = targetBitrate * bitratePercentage / 100
	//         maxBitrate = targetBitrate

#if defined(USE_OLD_VAENC) && USE_OLD_VAENC == 1
	RateControlMode rcMode = RateControlMode::QVBR;  // 码率控制模式
#else
	RateControlMode rcMode = RateControlMode::VCM;  // 码率控制模式
#endif

	u32  rcQualityFactor = 26;     // 码率控制质量因数 (仅 QVBR 有效， 1 - 51, 越小越好/越慢)
	bool useFileSink     = false;  // 是否使用文件输出

#ifndef __APPLE__
	bool useX265 = false;  // 是否使用 x265 编码器
#else
	const bool useX265 = true;
#endif

	bool staticSimplify           = true;   // 启用静态背景简化
	u32  bgMonochromeBitrateThres = 80;     // 触发背景去色的码率阈值
	bool forceBgMonochrome        = false;  // 强制剥离背景色彩
	i32  motionThres              = 14;     // 运动检测像素差值阈值
	i32  motionErodePx            = 1;      // 运动掩码腐蚀半径
	i32  motionDilatePx           = 2;      // 运动掩码膨胀半径
	i32  motionTrailFrames        = 3;      // 运动拖影保留帧数 (0为关闭)
	f64  trailDisableMotionRatio  = 0.30;   // 禁用拖影的运动区域占比阈值
	f64  bgUpdateAlpha            = 0.01;   // 背景模型学习率
	f64  bgBlurSigma              = 1.2;    // 静态背景高斯模糊强度
	i32  centerClearSize          = 30;     // 中心固定清晰横条高度（宽度为完整输出宽度）
	bool forceMonochrome          = false;  // 强制全局单色调

	i32 packetSize         = 300;  // 网络发包 payload 大小 (Byte)
	i32 maxPacketQueueSize = 40;   // 最大缓存包数 (~1s)
};

extern const DeployVtParam defaultVtParam;
extern const DeployVtParam lowBitratePreset;
}  // namespace hdvt
