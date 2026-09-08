#include "hero_deploy_vt/DeployVtEncoder.hpp"
#include "hero_deploy_vt/DeployVtParam.hpp"

#include <gst/app/app.h>
#include <gst/gst.h>

#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <deque>
#include <iostream>
#include <stop_token>
#include <thread>

using namespace std;
using namespace cv;
using namespace std::chrono;
using namespace std::chrono_literals;

atomic<bool> running{ true };

// -------------- 模拟发送端的滑动窗口限速器 ────────────────
struct TxRateLimiter
{
  double   bandwidthLimitKbytes = 15.0;
  double   windowSec            = 2.0;
  uint64_t totalSent            = 0;
  uint64_t totalDropped         = 0;

  bool trySend(size_t bytes)
  {
    const auto now       = steady_clock::now();
    const auto windowDur = duration_cast<nanoseconds>(duration<double>(windowSec));

    while (!window.empty() && (now - window.front().first) > windowDur) {
      windowBytes -= window.front().second;
      window.pop_front();
    }

    const size_t limit = static_cast<size_t>(bandwidthLimitKbytes * 1000.0 * windowSec);

    if (windowBytes + bytes > limit) { return false; }

    window.emplace_back(now, bytes);
    windowBytes += bytes;
    totalSent   += bytes;
    return true;
  }

  private:
  deque<pair<steady_clock::time_point, size_t>> window;
  size_t                                        windowBytes = 0;
};

struct DecoderDisplay
{
  GstElement* pipe   = nullptr;
  GstElement* appsrc = nullptr;

  bool init()
  {
    pipe = gst_pipeline_new("decoder_pipe");

    appsrc       = gst_element_factory_make("appsrc", "dec_src");
    auto* parser = gst_element_factory_make("h265parse", "dec_parser");
#ifndef __APPLE__
    auto* decoder = gst_element_factory_make("vah265dec", "dec_decoder");
#else
    auto* decoder = gst_element_factory_make("vtdec_hw", "dec_decoder");
#endif
    auto* convert = gst_element_factory_make("videoconvert", "dec_convert");

#ifndef __APPLE__
    auto* sink = gst_element_factory_make("autovideosink", "dec_sink");
#else
    auto* sink = gst_element_factory_make("osxvideosink", "dec_sink");
#endif

    if (!pipe || !appsrc || !parser || !decoder || !convert || !sink) {
      cerr << "Error: Failed to create decoder pipeline elements" << endl;
      return false;
    }

    g_object_set(
      appsrc, "stream-type", 0, "format", GST_FORMAT_BYTES, "is-live", TRUE, nullptr
    );
    g_object_set(parser, "config-interval", -1, "disable-passthrough", FALSE, nullptr);
    g_object_set(sink, "sync", FALSE, nullptr);

    gst_bin_add_many(GST_BIN(pipe), appsrc, parser, decoder, convert, sink, nullptr);
    if (!gst_element_link_many(appsrc, parser, decoder, convert, sink, nullptr)) {
      cerr << "Error: Failed to link decoder pipeline" << endl;
      return false;
    }

    auto ret = gst_element_set_state(pipe, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
      cerr << "Error: Failed to start decoder pipeline" << endl;
      return false;
    }
    return true;
  }

  void pushData(const uint8_t* data, size_t size)
  {
    if (!appsrc || size == 0) return;
    auto* buf = gst_buffer_new_memdup(data, size);
    gst_app_src_push_buffer(GST_APP_SRC(appsrc), buf);
  }

  void shutdown()
  {
    if (pipe) {
      gst_element_set_state(pipe, GST_STATE_NULL);
      gst_object_unref(pipe);
      pipe = nullptr;
    }
  }
};

void onSignal(int)
{ 
    running.store(false); 
}

int pull_test_main(gpointer)
{
  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);

  mutex frameMtx;
  Mat   sharedFrame;

#ifndef __APPLE__
  atomic<bool> displayRaw{ false };
#else
  constexpr atomic<bool> displayRaw{ false };
#endif
  atomic<bool> capFromFile{ false };
  atomic<bool> showProfLog{ true };

  VideoCapture cap("./tasks/hero_deploy_vt/res/test_video1.mp4");
  if (!cap.isOpened()) {
    cerr << "Error: Failed to open camera." << endl;
    return -1;
  }

  double fps = cap.get(CAP_PROP_FPS);
  if (fps <= 0.0) { fps = 30.0; }
  cout << "Source FPS: " << fps << endl;

  const auto frameInv = duration_cast<milliseconds>(1.0s / fps);
  cout << "Frame interval: " << frameInv.count() << "ms" << endl;

  double totalFrames = cap.get(CAP_PROP_FRAME_COUNT);
  bool   isFile      = (totalFrames > 0);

  cout << "Total frames: " << static_cast<int>(totalFrames) << endl;
  cout << (!isFile ? "Capture from cam" : "Capture from video file") << endl;

  capFromFile.store(isFile);

  // ── 编码器参数 ──
  auto param              = hdvt::defaultVtParam;
  param.targetFps         = fps;
  param.targetBitrate     = 80;
  param.gopSize           = 2 * fps;
  param.minQp             = 25;
  param.encodeQuality     = 4;
  param.useFileSink       = false;
  param.motionTrailFrames = 3;
  param.forceBgMonochrome = true;
  param.forceMonochrome   = false;

  hdvt::DeployVtEncoder::initEncoderRuntime();

  auto encoder = hdvt::DeployVtEncoder::create(param);
  if (!encoder->start()) {
    cerr << "Error: Failed to start encoder." << endl;
    return -1;
  }

  DecoderDisplay display;
  if (!display.init()) {
    cerr << "Error: Failed to init decoder display." << endl;
    return -1;
  }

  // ── 线程1: 采集 + 编码 ──
  jthread captureThread([&](stop_token st) {
    Mat frame;
    while (!st.stop_requested() && running) {
      cap >> frame;
      if (frame.empty()) {
        cerr << "Warning: No frame captured, exitting..." << endl;
        running = false;
        break;
      }
      encoder->tryPushFrame(frame);

      if (displayRaw.load()) {
        lock_guard<mutex> lk(frameMtx);
        frame.copyTo(sharedFrame);
      }

      if (capFromFile.load()) { this_thread::sleep_for(frameInv); }
    }
  });

  // ── 线程2: 50Hz pull + 限速 + 解码 ──
  jthread txThread([&](stop_token st) {
    TxRateLimiter limiter;
    limiter.bandwidthLimitKbytes = 15.0;
    limiter.windowSec            = 2.0;

    const auto txInterval = 20ms;
    auto       nextTxTime = steady_clock::now();

    auto     lastStatTime     = steady_clock::now();
    uint64_t lastStatSent     = 0;
    uint64_t lastStatEncBytes = 0;
    uint64_t txPktSent        = 0;
    uint64_t txPktBlocked     = 0;

    while (!st.stop_requested() && running) {
      this_thread::sleep_until(nextTxTime);
      nextTxTime += txInterval;

      vector<hdvt::u8> pkt;
      if (encoder->tryPullPacket(pkt)) {
        if (limiter.trySend(pkt.size())) {
          display.pushData(pkt.data(), pkt.size());
          txPktSent++;
        } else {
          txPktBlocked++;
          limiter.totalDropped += pkt.size();
        }
      }

      auto now = steady_clock::now();
      if (duration_cast<seconds>(now - lastStatTime).count() >= 1 && showProfLog.load()) {
        double intervalSec = duration_cast<duration<double>>(now - lastStatTime).count();

        auto   es      = encoder->getStats();
        double encKbps = static_cast<double>(es.totalBytesEncoded - lastStatEncBytes) *
                 8.0 / 1000.0 / intervalSec;
        double txKbps  = static_cast<double>(limiter.totalSent - lastStatSent) * 8.0 /
                 1000.0 / intervalSec;

        // cout << fixed << setprecision(2) << "[ENC] encBitrate=" << encKbps << "kbps"
        // 	 << " pktProduced=" << es.pktProduced << " queueDepth=" << es.pktQueueDepth
        // 	 << " dropped=" << es.droppedPkt << " dropEvents=" << es.dropEventCount
        // 	 << " | [TX] txBitrate=" << txKbps << "kbps"
        // 	 << " sent=" << txPktSent << " blocked=" << txPktBlocked
        // 	 << " totalSent=" << limiter.totalSent / 1000.0 << "kB"
        // 	 << " totalDropped=" << limiter.totalDropped / 1000.0 << "kB" << endl;

        lastStatEncBytes = es.totalBytesEncoded;
        lastStatSent     = limiter.totalSent;
        lastStatTime     = now;
      }
    }

    auto fs = encoder->getStats();
    // cout << fixed << setprecision(2)
    // 	 << "\n[Final ENC] totalEncoded=" << fs.totalBytesEncoded / 1000.0 << "kB"
    // 	 << " pktProduced=" << fs.pktProduced << " pktPulled=" << fs.pktPulled
    // 	 << " dropped=" << fs.droppedPkt << " dropEvents=" << fs.dropEventCount
    // 	 << " | [Final TX] sent=" << limiter.totalSent / 1000.0 << "kB"
    // 	 << " blocked=" << txPktBlocked << " totalDropped=" << limiter.totalDropped / 1000.0
    // 	 << "kB" << endl;
  });

  // ---- Main thread ----
  while (running) {
    if (displayRaw.load()) {
      lock_guard<mutex> lk(frameMtx);
      if (!sharedFrame.empty()) { imshow("Cam (Raw)", sharedFrame); }
    }

    if (waitKey(frameInv.count()) == 27) {
      running = false;
      break;
    }
  }

  captureThread.request_stop();
  txThread.request_stop();

  display.shutdown();
  cap.release();
  destroyAllWindows();

  return 0;
}

int main(int argc, char* argv[])
{
#if defined(__APPLE__) && TARGET_OS_MAC && !TARGET_OS_IPHONE
  return gst_macos_main_simple((GstMainFuncSimple)pull_test_main, nullptr);
#else
  return pull_test_main(nullptr);
#endif
}