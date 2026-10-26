"""Build a reviewed configuration bundle, then publish it to a new directory."""
import json
import os
import shutil
import tempfile
from pathlib import Path

import yaml

from harborrl.config.native import validate
from harborrl.trajectories.native import digest

from .presets import build_configs
from .validation import validate_mappings


def build_config_bundle(answers, *, output_directory):
    output = Path(output_directory).expanduser().resolve()
    training, deployment = build_configs(answers, output)
    result = validate_mappings(training, deployment, config_path=output / "config.yaml")
    if result["status"] == "invalid":
        raise ValueError("; ".join(f"{row['path']}: {row['message']}" for row in result["checks"] if row["status"] == "failed"))
    training = validate(training, output / "config.yaml")
    manifest = {"setup_version": 1, "preset": answers["preset"],
                "schema_versions": {"training": 2, "deployment": 1},
                "config_digest": digest(training), "deployment_digest": digest(deployment),
                "references": {"catalog": training["tasks"]["catalog"],
                               "profile": training["harness"]["profile"],
                               "sandbox_config": training["harbor"].get("environment", {}).get("config")},
                "validation": result}
    return {"files": {"config.yaml": yaml.safe_dump(training, sort_keys=False),
                      "deployment.yaml": yaml.safe_dump(deployment, sort_keys=False),
                      "setup-manifest.json": json.dumps(manifest, indent=2) + "\n"},
            "validation": result}


def publish_bundle(bundle, output_directory):
    """No overwrite; same-parent staging and an exclusive per-target lock."""
    output = Path(output_directory).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output directory already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError("Output parent directory must already exist")
    files = bundle["files"]
    if set(files) != {"config.yaml", "deployment.yaml", "setup-manifest.json"}:
        raise ValueError("Bundle must contain exactly the three documented files")
    lock = output.parent / f".{output.name}.init.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    staging = None
    try:
        os.close(descriptor)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output directory already exists: {output}")
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.init-", dir=output.parent))
        for name, content in files.items():
            with (staging / name).open("x", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output directory already exists: {output}")
        staging.rename(output)
        staging = None
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock.unlink()
    return output
