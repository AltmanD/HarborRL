"""CPU contracts for the explicit MATH component interfaces."""
import json
from dataclasses import replace

import pytest

from harborrl.math_registry import (
    BACKEND_CONTRACT,
    list_components,
    select_math,
    validate_combination,
)
from harborrl.models.qwen import QwenModelAdapter
from harborrl.tasks.jsonl_provider import JsonlTaskProvider
from harborrl.tasks.provider import NativeTaskProvider


def math_config():
    return {
        "execution": {"backend": "harbor_job"},
        "harness": {"name": "claude_code"},
        "harbor": {"version": "0.23.0"},
        "gateway": {"audited_raw_logprobs": True},
        "model": {"args_file": "qwen3-8B"},
        "training": {"backend_contract": BACKEND_CONTRACT},
        "sampling": {"group_size": 2},
    }


def test_registry_lists_versioned_default_components():
    components = list_components()
    assert [spec["id"] for spec in components["models"]] == ["qwen"]
    assert [spec["id"] for spec in components["algorithms"]] == ["grpo"]
    assert [spec["id"] for spec in components["tasks"]] == ["harbor_jsonl", "native_harbor"]
    assert [spec["id"] for spec in components["harnesses"]] == ["claude_code"]
    for dimension in components.values():
        for spec in dimension:
            assert spec["version"]
            assert spec["evidence"]
            assert spec["limitations"]

def test_combination_checks_reject_incompatible_evidence_and_groups():
    config = math_config()
    selection = select_math(config)
    assert validate_combination(selection, config) is selection

    unaudited = {**config, "gateway": {"audited_raw_logprobs": False}}
    with pytest.raises(ValueError, match="requires audited raw logprobs"):
        validate_combination(selection, unaudited)

    larger_group = replace(selection.algorithm, minimum_group_size=3)
    with pytest.raises(ValueError, match="requires group size"):
        validate_combination(replace(selection, algorithm=larger_group), config)

    class Tokenizer:
        def get(self, key):
            assert key == "input_ids"
            return [1, 2]

        def apply_chat_template(self, *args, **kwargs):
            assert kwargs["enable_thinking"] is False
            return self

    adapter = QwenModelAdapter()
    converted = {"messages": [{"role": "user", "content": "run"}], "tools": []}
    assert adapter.render_chat(Tokenizer(), converted) == [1, 2]
    blocks = adapter.parse_output(
        '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>',
        response_id="message",
        tools=[{"function": {"name": "bash"}}],
    )
    assert blocks[0]["type"] == "tool_use"


def test_jsonl_task_provider_is_selected_by_catalog_format(tmp_path):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("jsonl instruction\n")
    (task / "environment/Dockerfile").write_text("FROM debian:bookworm\n")
    (task / "tests/test.sh").write_text("true\n")
    provider = JsonlTaskProvider()
    digest = provider.inspect(task)["task_digest"]
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(json.dumps({
        "id": "task", "revision": "1", "path": str(task),
        "task_digest": digest, "reward_profile": {"key": "reward"},
    }) + "\n")
    config = {**math_config(), "tasks": {"catalog": str(catalog)}}
    selection = select_math(config)
    assert selection.task.id == "harbor_jsonl"
    entries = provider.catalog(config)
    rows = provider.prompt_rows(entries)
    assert rows[0]["task"] == "jsonl instruction\n"
    assert rows[0]["metadata"]["task_provider"] == "harbor_jsonl"

    incompatible = {**config, "execution": {"backend": "process"}}
    with pytest.raises(ValueError, match="does not support execution backend"):
        validate_combination(select_math(incompatible), incompatible)

    catalog.write_text("{broken\n")
    with pytest.raises(ValueError, match="invalid JSONL catalog line"):
        provider.catalog(config)


def test_native_task_provider_locks_and_materializes_prompts(tmp_path):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("original instruction\n")
    (task / "environment/Dockerfile").write_text("FROM debian:bookworm\n")
    (task / "tests/test.sh").write_text("true\n")
    provider = NativeTaskProvider()
    digest = provider.inspect(task)["task_digest"]
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([{
        "id": "task", "revision": "1", "path": str(task),
        "task_digest": digest, "reward_profile": {"key": "reward"},
    }]))
    config = {**math_config(), "tasks": {"catalog": str(catalog)}}
    entries = provider.catalog(config)
    rows = provider.prompt_rows(entries)
    assert rows[0]["task"] == "original instruction\n"
    assert rows[0]["metadata"]["native_task"]["task_digest"] == digest
