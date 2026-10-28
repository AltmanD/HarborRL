"""Exercise bootstrap output without cloning repositories or pulling images."""
import json
import os
import subprocess
import sys
from pathlib import Path


def test_backend_env_exports_quoted_absolute_paths(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text('''#!/usr/bin/env bash
set -eu
if [[ "$1" == clone ]]; then
  mkdir -p "$3/.git"
elif [[ "$1" == -C && "$3" == rev-parse ]]; then
  case "$2" in
    */slime) printf '%s\\n' 3778dbf6d1a533ab478ecf5ddaa11449a47752b2 ;;
    */Megatron-LM) printf '%s\\n' 1dcf0dafa884ad52ffb243625717a3471643e087 ;;
  esac
fi
''')
    docker = bin_dir / "docker"
    docker.write_text('#!/usr/bin/env bash\n[[ "$1" == pull ]]\n')
    git.chmod(0o755)
    docker.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if key in {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"}}
    env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
    env["SGLANG_IMAGE"] = "registry/sglang:pinned"
    prefix = "backend 'stack' $literal"
    script = Path(__file__).resolve().parents[2] / "scripts/bootstrap_backends.sh"
    subprocess.run(["bash", str(script), prefix], cwd=tmp_path, env=env,
                   check=True, capture_output=True, text=True, timeout=10)
    expected_root = tmp_path / prefix
    env_file = expected_root / "harborrl-backend.env"
    assert env_file.read_text().count("export ") == 3
    # Source in a fresh shell from a different directory, then inspect a child.
    child = subprocess.run(
        ["bash", "-c", '. "$1"\nexec "$2" -c "$3"', "bootstrap-test",
         str(env_file), sys.executable,
         'import json, os; print(json.dumps({k: os.environ.get(k) for k in '
         '("SLIME_DIR", "MEGATRON_DIR", "SGLANG_IMAGE")}))'],
        cwd=bin_dir, env={key: value for key, value in env.items() if key != "SGLANG_IMAGE"},
        check=True, capture_output=True, text=True, timeout=10)
    assert json.loads(child.stdout) == {
        "SLIME_DIR": str(expected_root / "slime"),
        "MEGATRON_DIR": str(expected_root / "Megatron-LM"),
        "SGLANG_IMAGE": "registry/sglang:pinned",
    }
