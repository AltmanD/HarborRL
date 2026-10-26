"""Native task content hashing and TOML reader shared with compatibility tools."""
import hashlib
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


__all__ = ["tomllib", "tree_digest"]


def tree_digest(root: Path, *, exclude: tuple[str, ...] = ()) -> str:
    h = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if str(path.relative_to(root)) in exclude:
            continue
        if path.is_symlink():
            raise ValueError("symlinks are not supported")
        if path.is_file():
            data = path.read_bytes()
            h.update(str(path.relative_to(root)).encode() + b"\0")
            h.update(str(path.stat().st_mode & 0o777).encode() + b"\0")
            h.update(str(len(data)).encode() + b"\0" + data)
    return h.hexdigest()
