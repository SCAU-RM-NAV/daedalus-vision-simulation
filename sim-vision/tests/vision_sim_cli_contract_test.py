#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    runner = root / "build" / "vision_sim_runner"
    assert runner.is_file(), "build vision_sim_runner before this test"

    dry_run = subprocess.run(
        [
            str(runner),
            "--scenario=daedalus_cli_contract",
            "--dry-run",
        ],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    report = json.loads(dry_run.stdout)
    assert report["kind"] == "daedalus-talos-dry-run"
    assert report["simulator"] == "Daedalus"
    assert report["scenario"] == "daedalus_cli_contract"
    assert report["algorithm_config_path"] == "configs/sentry.yaml"
    assert report["simulation_overlay_path"] == "configs/daedalus_overlay.yaml"
    assert report["inference_device_override"] == "CPU"
    assert report["physical_fire_enabled"] is True
    assert report["meta_path"] == "/tmp/talos_ipc_meta"
    assert report["image_pool_path"] == "/tmp/talos_ipc_image_pool"


if __name__ == "__main__":
    main()
