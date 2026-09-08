#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    runner_source = (root / "src" / "vision_sim_runner.cpp").read_text(encoding="utf-8")
    debug_source = (root / "sim" / "debug_visualizer.cpp").read_text(encoding="utf-8")
    assert runner_source.index("process_with_debug(armors, runtime_frame)") < runner_source.index(
        "evaluate_armors(armors, frame->data.truth"
    )
    assert "if (candidate.team != team) continue;" in runner_source
    assert "if (nearest->armor_label == label) ++stats.detection_truth_matches;" in runner_source
    assert "ARMOR_NAMES[armor.name]" in debug_source
    assert "pnp:{}" in debug_source
    build = root / "build"
    fixture = build / "talos_ipc_fixture"
    runner = build / "vision_sim_runner"
    assert fixture.is_file(), "build talos_ipc_fixture before this test"
    assert runner.is_file(), "build vision_sim_runner before this test"

    with tempfile.TemporaryDirectory(prefix="talos_report_contract_") as directory:
        directory = Path(directory)
        meta = directory / "meta"
        image = directory / "image"
        reports = directory / "reports"
        publisher = subprocess.Popen([str(fixture), str(meta), str(image)])
        try:
            time.sleep(0.2)
            completed = subprocess.run(
                [
                    str(runner),
                    "configs/standard4.yaml",
                    f"--meta-path={meta}",
                    f"--image-pool-path={image}",
                    f"--session-dir={reports}",
                    "--scenario=infantry_stationary_armor",
                    "--max-frames=1",
                ],
                cwd=root,
                check=False,
                text=True,
                capture_output=True,
            )
            assert completed.returncode == 0, completed.stderr
            assert publisher.wait(timeout=10) == 0
        finally:
            if publisher.poll() is None:
                publisher.kill()
                publisher.wait()

        session = next(reports.iterdir())
        metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
        truth = json.loads((session / "ground_truth.jsonl").read_text(encoding="utf-8"))
        geometry = json.loads((session / "geometry.jsonl").read_text(encoding="utf-8"))
        assert metadata["algorithm_config_hash_fnv1a64"]
        assert metadata["inference_device"] in {"CPU", "GPU"}
        assert metadata["simulator"] == "Daedalus"
        assert metadata["physical_fire_enabled"] is True
        runtime_config = session / "runtime_config.yaml"
        assert runtime_config.is_file()
        assert f"device: {metadata['inference_device']}" in runtime_config.read_text(encoding="utf-8")
        assert "targets" in truth and "runes" in truth
        assert truth["target_count"] == len(truth["targets"])
        assert truth["rune_count"] == len(truth["runes"])
        assert geometry["target_count"] == len(geometry["targets"])
        assert geometry["rune_count"] == len(geometry["runes"])


if __name__ == "__main__":
    main()
