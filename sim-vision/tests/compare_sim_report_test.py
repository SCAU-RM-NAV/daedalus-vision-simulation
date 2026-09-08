#!/usr/bin/env python3
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "compare_sim_report.py"


def metric_report(value):
    return {"metrics": {metric: value for metric in (
        "image_delivery_rate",
        "detection_match_rate",
        "pnp_position_error_m",
        "pnp_orientation_error_rad",
        "tracking_continuity",
        "end_to_end_latency_ms_mean",
        "gimbal_follow_error_rad",
        "hit_rate",
        "rune_correct_target_rate",
        "rune_wrong_target_rate",
        "rune_timeout_rate",
    )}}


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        baseline = root / "baseline.json"
        current = root / "current.json"
        output = root / "comparison" / "report.json"
        baseline.write_text(json.dumps(metric_report(1.0)), encoding="utf-8")
        current_values = metric_report(0.5)
        for metric in (
            "pnp_position_error_m",
            "pnp_orientation_error_rad",
            "end_to_end_latency_ms_mean",
            "gimbal_follow_error_rad",
            "rune_wrong_target_rate",
            "rune_timeout_rate",
        ):
            current_values["metrics"][metric] = 1.5
        current.write_text(json.dumps(current_values), encoding="utf-8")

        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--baseline", str(baseline), "--current", str(current),
             "--output", str(output)],
            check=False,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stderr
        report = json.loads(output.read_text(encoding="utf-8"))
        expected = set(metric_report(1.0)["metrics"])
        assert set(report["deltas"]) == expected
        assert set(report["degraded_metrics"]) == expected
        assert baseline.read_text(encoding="utf-8") == json.dumps(metric_report(1.0))
        assert current.read_text(encoding="utf-8") == json.dumps(current_values)


if __name__ == "__main__":
    main()
