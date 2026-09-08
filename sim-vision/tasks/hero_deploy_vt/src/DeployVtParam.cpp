#include "hero_deploy_vt/DeployVtParam.hpp"

namespace hdvt {
const DeployVtParam defaultVtParam   = {};
const DeployVtParam lowBitratePreset = [] {
  DeployVtParam p = defaultVtParam;

  p.targetFps         = 30;
  p.targetBitrate     = 60;
  p.bFramesCount      = 4;
  p.refFrames         = 5;
  p.gopSize           = 4 * p.targetFps;
  p.maxQp             = 45;
  p.minQp             = 25;
  p.encodeQuality     = 7;
  p.forceBgMonochrome = true;
  p.bgBlurSigma       = 3.0;

  return p;
}();
}  // namespace hdvt