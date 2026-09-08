#!/usr/bin/env python3
"""Read a reviewed simulation baseline and create a new comparison report.

This command intentionally never writes source, configuration, baseline, or Git state.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys


METRICS = (
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
)

LOWER_IS_BETTER = frozenset(
    (
        "pnp_position_error_m",
        "pnp_orientation_error_rad",
        "end_to_end_latency_ms_mean",
        "gimbal_follow_error_rad",
        "rune_wrong_target_rate",
        "rune_timeout_rate",
    )
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = args.baseline.resolve()
    current = args.current.resolve()
    output = args.output.resolve()
    if not baseline.is_file() or not current.is_file():
        parser.error("--baseline and --current must be existing JSON files")
    if output == baseline or output == current or baseline.is_relative_to(output.parent):
        parser.error("--output must be a new report path outside the baseline path")
    if output.exists():
        parser.error("--output already exists; comparisons never overwrite prior reports")

    baseline_report = load_json(baseline)
    current_report = load_json(current)
    baseline_metrics = baseline_report.get("metrics", {})
    current_metrics = current_report.get("metrics", {})
    deltas = {}
    degraded = []
    for metric in METRICS:
        old = baseline_metrics.get(metric)
        new = current_metrics.get(metric)
        if old is None or new is None:
            deltas[metric] = None
            continue
        delta = new - old
        deltas[metric] = delta
        if (metric in LOWER_IS_BETTER and delta > 0) or (
            metric not in LOWER_IS_BETTER and delta < 0
        ):
            degraded.append(metric)

    digest = hashlib.sha256(current.read_bytes()).hexdigest()
    comparison = {
        "kind": "simulation-regression-comparison",
        "baseline": str(baseline),
        "current": str(current),
        "current_sha256": digest,
        "deltas": deltas,
        "degraded_metrics": degraded,
        "status": "degraded" if degraded else "no_regression_detected",
        "policy": "report-only; no baseline, source, config, Git state, or commit was modified",
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
