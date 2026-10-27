"""Explicit MATH component contracts and startup compatibility checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from harborrl.algorithms.grpo import GRPOAlgorithmAdapter
from harborrl.backends.slime_v032.versions import BACKEND_CONTRACT
from harborrl.harnesses.claude_code import ClaudeCodeHarnessAdapter
from harborrl.models.qwen import QwenModelAdapter
from harborrl.tasks.jsonl_provider import JsonlTaskProvider
from harborrl.tasks.provider import NativeTaskProvider


class ComponentRegistry:
    def __init__(self, dimension):
        self.dimension = dimension
        self._components = {}

    def register(self, spec):
        if spec.id in self._components:
            raise ValueError(f"duplicate {self.dimension} registration: {spec.id}")
        self._components[spec.id] = spec
        return spec

    def get(self, component_id):
        try:
            return self._components[component_id]
        except KeyError as exc:
            raise ValueError(
                f"unsupported {self.dimension}: {component_id}; "
                f"registered: {', '.join(sorted(self._components))}"
            ) from exc

    def items(self):
        return tuple(self._components[spec_id] for spec_id in sorted(self._components))


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    preset_prefix: str
    serving_backend: str
    tool_protocol: str
    gateway_protocols: frozenset[str]
    training_backend_contracts: frozenset[str]
    adapter: QwenModelAdapter
    version: str
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class AlgorithmSpec:
    id: str
    display_name: str
    training_backend_contracts: frozenset[str]
    minimum_group_size: int
    requires_audited_raw_logprobs: bool
    reduction: str
    adapter: GRPOAlgorithmAdapter
    backend_hooks: dict[str, str]
    version: str
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class TaskProviderSpec:
    id: str
    display_name: str
    execution_backends: frozenset[str]
    task_format: str
    adapter: TaskProviderAdapter
    version: str
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class HarnessSpec:
    id: str
    display_name: str
    request_protocol: str
    tool_protocol: str
    harbor_versions: frozenset[str]
    adapter: ClaudeCodeHarnessAdapter
    version: str
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]


class ModelAdapter(Protocol):
    def render_chat(self, tokenizer, converted): ...

    def parse_output(self, text, *, response_id, tools): ...


class AlgorithmAdapter(Protocol):
    def compute_advantages_and_returns(self, args, rollout_data): ...


class TaskAdapter(Protocol):
    def discover(self, tasks_dir): ...

    def inspect(self, path, *, expected_digest=None): ...

    def catalog(self, config): ...


class TaskProviderAdapter(Protocol):
    def discover(self, tasks_dir): ...

    def inspect(self, path, *, expected_digest=None): ...

    def catalog(self, config): ...

    def prompt_rows(self, entries): ...


class HarnessAdapter(Protocol):
    def validate_profile(self, profile, gateway_url): ...

    def agent_config(self, profile, gateway_url, credential, max_output_tokens=8192): ...

QWEN_MODEL = ModelSpec(
    id="qwen",
    display_name="Qwen",
    preset_prefix="qwen",
    serving_backend="sglang",
    tool_protocol="qwen-tool-call-v1",
    gateway_protocols=frozenset({"anthropic-messages-v1"}),
    training_backend_contracts=frozenset({BACKEND_CONTRACT}),
    adapter=QwenModelAdapter(),
    version="qwen3-profile-v1",
    evidence=("examples/native/train_qwen_native.yaml",),
    limitations=("preset must start with qwen", "tokenizer and template digests are checked live"),
)
GRPO_ALGORITHM = AlgorithmSpec(
    id="grpo",
    display_name="Slime GRPO",
    training_backend_contracts=frozenset({BACKEND_CONTRACT}),
    minimum_group_size=2,
    requires_audited_raw_logprobs=True,
    reduction="global-sum",
    adapter=GRPOAlgorithmAdapter(),
    backend_hooks={
        "rollout": "harborrl.backends.slime_v032.rollout.generate_rollout",
        "converter": "harborrl.backends.slime_v032.converter.convert_samples_to_train_data",
        "advantage": "harborrl.backends.slime_v032.advantage.compute_advantages_and_returns",
        "loss": "harborrl.backends.slime_v032.loss.loss_function",
        "postprocess": "harborrl.backends.slime_v032.postprocess.rollout_data_postprocess",
    },
    version="native-centered-grpo-v1",
    evidence=("harborrl/backends/slime_v032/loss.py",),
    limitations=("requires complete reward-varied groups", "only Slime v0.3.2 hooks are accepted"),
)
HARBOR_JSONL_TASK_PROVIDER = TaskProviderSpec(
    id="harbor_jsonl",
    display_name="Harbor JSONL",
    execution_backends=frozenset({"harbor_job"}),
    task_format="native_single_step_jsonl_v1",
    adapter=JsonlTaskProvider(),
    version="jsonl-catalog-lock-v1",
    evidence=("examples/native/catalog.jsonl",),
    limitations=(
        "one task lock must be encoded per JSONL line",
        "tasks must be fetched before cataloging",
        "task selectors use catalog IDs",
    ),
)
NATIVE_TASK_PROVIDER = TaskProviderSpec(
    id="native_harbor",
    display_name="Harbor",
    execution_backends=frozenset({"harbor_job"}),
    task_format="native_single_step_text_v1",
    adapter=NativeTaskProvider(),
    version="catalog-lock-v1",
    evidence=("examples/native/catalog.json",),
    limitations=("tasks must be fetched before cataloging", "task selectors use catalog IDs"),
)
CLAUDE_CODE_HARNESS = HarnessSpec(
    id="claude_code",
    display_name="Claude Code",
    request_protocol="anthropic-messages-v1",
    tool_protocol="qwen-tool-call-v1",
    harbor_versions=frozenset({"0.23.0"}),
    adapter=ClaudeCodeHarnessAdapter(),
    version="claude-code-session-audit-v1",
    evidence=("harborrl/rollout/harbor_job/audit.py",),
    limitations=("exact CLI version is required", "ambient credentials are rejected"),
)
class MATHRegistry:
    def __init__(self):
        self.models = ComponentRegistry("model")
        self.algorithms = ComponentRegistry("algorithm")
        self.tasks = ComponentRegistry("task provider")
        self.harnesses = ComponentRegistry("harness")
        self.models.register(QWEN_MODEL)
        self.algorithms.register(GRPO_ALGORITHM)
        self.tasks.register(NATIVE_TASK_PROVIDER)
        self.tasks.register(HARBOR_JSONL_TASK_PROVIDER)
        self.harnesses.register(CLAUDE_CODE_HARNESS)

    def model_for_preset(self, preset):
        for spec in self.models.items():
            if str(preset).startswith(spec.preset_prefix):
                return spec
        raise ValueError(
            f"unsupported model preset: {preset}; "
            f"registered: {', '.join(spec.id for spec in self.models.items())}"
        )

    def algorithm_for_contract(self, contract):
        for spec in self.algorithms.items():
            if contract in spec.training_backend_contracts:
                return spec
        raise ValueError(f"no registered algorithm supports backend contract: {contract}")


MATH_REGISTRY = MATHRegistry()


@dataclass(frozen=True)
class MATHSelection:
    model: ModelSpec
    algorithm: AlgorithmSpec
    task: TaskProviderSpec
    harness: HarnessSpec

    def asdict(self):
        return {
            "model": _describe(self.model),
            "algorithm": _describe(self.algorithm),
            "task": _describe(self.task),
            "harness": _describe(self.harness),
        }


def _describe(spec):
    return {
        "id": spec.id,
        "display_name": spec.display_name,
        "version": spec.version,
        "evidence": list(spec.evidence),
        "limitations": list(spec.limitations),
    }


def select_math(config):
    return MATHSelection(
        model=MATH_REGISTRY.model_for_preset(config["model"]["args_file"]),
        algorithm=MATH_REGISTRY.algorithm_for_contract(
            config["training"]["backend_contract"]
        ),
        task=MATH_REGISTRY.tasks.get(
            "harbor_jsonl"
            if str(config.get("tasks", {}).get("catalog", "")).endswith(".jsonl")
            else "native_harbor"
        ),
        harness=MATH_REGISTRY.harnesses.get(config["harness"]["name"]),
    )


def validate_combination(selection, config):
    contract = config["training"]["backend_contract"]
    if contract not in selection.model.training_backend_contracts:
        raise ValueError(
            f"model {selection.model.id} does not support training backend {contract}"
        )
    if contract not in selection.algorithm.training_backend_contracts:
        raise ValueError(
            f"algorithm {selection.algorithm.id} does not support training backend {contract}"
        )
    if config["execution"]["backend"] not in selection.task.execution_backends:
        raise ValueError(
            f"task provider {selection.task.id} does not support execution backend "
            f"{config['execution']['backend']}"
        )
    if config["harbor"]["version"] not in selection.harness.harbor_versions:
        raise ValueError(
            f"harness {selection.harness.id} does not support Harbor "
            f"{config['harbor']['version']}"
        )
    if selection.model.tool_protocol != selection.harness.tool_protocol:
        raise ValueError("model and harness tool protocols do not match")
    if selection.harness.request_protocol not in selection.model.gateway_protocols:
        raise ValueError(
            f"model gateway does not support harness protocol {selection.harness.request_protocol}"
        )
    if config["sampling"]["group_size"] < selection.algorithm.minimum_group_size:
        raise ValueError(
            f"algorithm {selection.algorithm.id} requires group size >= "
            f"{selection.algorithm.minimum_group_size}"
        )
    if (
        selection.algorithm.requires_audited_raw_logprobs
        and not config["gateway"]["audited_raw_logprobs"]
    ):
        raise ValueError(
            f"algorithm {selection.algorithm.id} requires audited raw logprobs"
        )
    return selection


def list_components():
    return {
        dimension: [_describe(spec) for spec in registry.items()]
        for dimension, registry in (
            ("models", MATH_REGISTRY.models),
            ("algorithms", MATH_REGISTRY.algorithms),
            ("tasks", MATH_REGISTRY.tasks),
            ("harnesses", MATH_REGISTRY.harnesses),
        )
    }
