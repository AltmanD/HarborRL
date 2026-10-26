"""Generate the repository compatibility manifest from packaged metadata."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = ROOT / "harborrl/options/resources/datasets.yaml"
    target = ROOT / "configs/harbor_hub/manifests.yaml"
    expected = source.read_text(encoding="utf-8")
    if args.check:
        return int(target.read_text(encoding="utf-8") != expected)
    target.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
