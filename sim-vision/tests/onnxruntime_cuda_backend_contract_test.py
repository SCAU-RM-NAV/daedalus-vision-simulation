#!/usr/bin/env python3
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    config = (root / "configs/sentry.yaml").read_text(encoding="utf-8")
    yolov5 = (root / "tasks/auto_aim/yolos/yolov5.cpp").read_text(encoding="utf-8")
    buff = (root / "tasks/auto_buff/yolo11_buff.cpp").read_text(encoding="utf-8")

    assert "inference_backend: opencv_dnn_cuda" in config
    assert "onnxruntime_cuda" in yolov5
    assert "onnxruntime_cuda" in buff


if __name__ == "__main__":
    main()
