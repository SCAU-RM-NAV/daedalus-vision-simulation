#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    probe = root / "build" / "sim_model_probe"
    model = root / "assets" / "20260731best.xml"
    assert probe.is_file(), "build sim_model_probe before this test"
    completed = subprocess.run(
        [str(probe), str(model)], cwd=root, check=False, text=True, capture_output=True
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["model_path"] == str(model)
    assert report["inputs"]
    assert report["outputs"]
    assert "CPU" in report["compile"]
    assert "GPU" in report["compile"]


if __name__ == "__main__":
    main()
