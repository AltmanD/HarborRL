"""Legacy TerminalEnv compatibility tooling; native training does not use this path."""

from dataclasses import asdict, dataclass
from pathlib import Path

from harborrl.tasks.content import tomllib, tree_digest  # compatibility exports

from .receipt import parse_reward

PROFILE = "terminal_text_v1"
REWARD_PROFILE = "harbor_reward_txt_v1"


@dataclass(frozen=True)
class TaskRef:
    dataset: str
    task: str
    source_digest: str


@dataclass
class Inspection:
    ref: TaskRef
    status: str
    reasons: list[str]
    resources: dict
    profile: str = PROFILE

    def to_dict(self):
        return asdict(self)


def inspect(task: Path, dataset: str, receipt: dict | None = None) -> Inspection:
    task = Path(task)
    ref = TaskRef(dataset, task.name, "")
    if not dataset.strip():
        return Inspection(ref, "INVALID", ["empty_dataset_identity"], {})
    try:
        ref = TaskRef(dataset, task.name, tree_digest(task))
        spec = tomllib.loads((task / "task.toml").read_text())
        for section in ("environment", "agent", "verifier"):
            if not isinstance(spec.get(section, {}), dict):
                raise TypeError("invalid section: " + section)
        for name in ("instruction.md", "environment/Dockerfile", "tests/test.sh"):
            if not (task / name).is_file():
                return Inspection(ref, "INVALID", ["missing:" + name], {})
        if not (task / "instruction.md").read_text().strip():
            return Inspection(ref, "INVALID", ["empty_instruction"], {})
    except (OSError, TypeError, ValueError) as exc:
        return Inspection(ref, "INVALID", [str(exc)], {})
    env = spec.get("environment", {})
    resources = {
        k: env[k]
        for k in ("cpus", "memory_mb", "storage_mb", "allow_internet", "gpus")
        if k in env
    }
    if any(
        (task / "environment" / n).exists()
        for n in (
            "docker-compose.yaml",
            "docker-compose.yml",
            "compose.yaml",
            "compose.yml",
        )
    ):
        return Inspection(ref, "UNSUPPORTED", ["compose_mapping_unverified"], resources)
    if env.get("gpus", 0):
        return Inspection(ref, "UNSUPPORTED", ["gpu_environment"], resources)
    if env.get("docker_image"):
        return Inspection(ref, "UNSUPPORTED", ["image_override_unverified"], resources)
    script = (task / "tests/test.sh").read_text()
    if "/logs/verifier/reward.txt" not in script:
        return Inspection(
            ref, "UNSUPPORTED", ["reward_profile_unrecognized"], resources
        )
    try:
        reward_valid = (
            receipt is not None
            and parse_reward(str(receipt.get("native_reward"))) is not None
        )
    except ValueError:
        reward_valid = False
    supported = bool(
        reward_valid
        and receipt
        and receipt.get("source_digest") == ref.source_digest
        and receipt.get("profile") == PROFILE
        and receipt.get("success") is True
        and receipt.get("native_reward") == receipt.get("materialized_reward")
        and receipt.get("execution_backend") == "terminal_env"
    )
    return Inspection(
        ref,
        "SUPPORTED" if supported else "NEEDS_PROBE",
        [] if supported else ["terminal_execution_unverified"],
        resources,
    )
