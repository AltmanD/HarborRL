"""Regression coverage for the full coordinator success path and empty finals."""
import asyncio
import json
import sys

import pytest
from test_native_contracts import identity, make_ir
from test_native_lifecycle import PROFILE

from harborrl.gateway.trace import TraceRegistry
from harborrl.harnesses.claude_code import ClaudeCodeHarnessAdapter
from harborrl.rollout.harbor_job.coordinator import run_attempt
from harborrl.trajectories.native import RewardProfile, digest


def test_casefold_tool_alias_must_be_unique():
    from harborrl.gateway.messages import ProtocolError, parse_qwen_output
    with pytest.raises(ProtocolError, match="ambiguous"):
        parse_qwen_output('<tool_call>{"name":"BASH","arguments":{}}</tool_call>',
                          response_id="r", tools=[{"function": {"name": "Bash"}},
                                                  {"function": {"name": "bash"}}])


@pytest.mark.parametrize("empty", [False, True])
def test_coordinator_publishes_complete_success(tmp_path, monkeypatch, empty):
    ident = identity()
    trial_id = "trial-" + ident.attempt_id
    root = tmp_path / "attempt"
    trial = root / "runner/harbor/trial"
    template = make_ir(tmp_path / "fixture", nturns=2)

    class Registry(TraceRegistry):
        def register(self, ident, trial_id, **kwargs):
            token = super().register(ident, trial_id, **kwargs)
            turns = template["turns"]
            for index, original in enumerate(turns):
                aid, request, _, _ = self.begin(token)
                turn = {**original, "request_id": request, "delivery": "sent"}
                if empty and index == 1:
                    turn["response"] = {"content": []}
                    turn["response_digest"] = digest(turn["response"])
                self.finish(aid, request, turn)
                self.delivery(aid, turn["response_id"], "sent")
            session = trial / "agent/sessions/projects/session.jsonl"
            session.parent.mkdir(parents=True)
            session.write_text("\n".join(json.dumps({
                "type": "assistant", "sessionId": "session", "message": {
                    "id": turn["response_id"], "role": "assistant",
                    "content": turn["response"]["content"]}})
                for turn in turns[:1 if empty else 2]))
            (trial / "agent/claude-code.txt").write_text(json.dumps({
                "type": "result", "subtype": "success", "is_error": False,
                "result": "", "stop_reason": "end_turn", "session_id": "session",
                "terminal_reason": "completed"}))
            (trial / "verifier").mkdir()
            (trial / "verifier/reward.json").write_text('{"reward": 1}')
            (trial / "result.json").write_text(json.dumps({
                "id": trial_id, "finished_at": "2026-10-19T00:00:00Z",
                "verifier_environment_mode": "shared",
                "verifier_result": {"rewards": {"reward": 1}}}))
            return token

    worker = tmp_path / "worker.py"
    terminal = {"type": "terminal", "trial_id": trial_id,
                "trial_dir": str(trial), "result_path": str(trial / "result.json")}
    worker.write_text("import json, sys\njson.loads(sys.stdin.readline())\n"
                      f"print({json.dumps({'type': 'prepared', 'trial_id': trial_id})!r}, flush=True)\n"
                      "sys.stdin.readline()\n"
                      f"print({json.dumps(terminal)!r}, flush=True)\n")
    spawn = asyncio.create_subprocess_exec

    async def fake_spawn(*args, **kwargs):
        return await spawn(sys.executable, "-S", str(worker), **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    registry = Registry(tmp_path / "trace")
    ir = asyncio.run(run_attempt(
        ident, task_path=tmp_path, profile=PROFILE, gateway_url="http://gateway",
        runner_python=sys.executable, root=root, registry=registry,
        reward_profile=RewardProfile()))
    assert ir["readiness"] == "CONTRACT_READY"
    assert json.loads((root / "trajectory.v2.json").read_text()) == ir
    assert (root / "evaluation.json").is_file()
    assert not (root / "cleanup-required.json").exists()
    assert not (root / "trace-unsealed.json").exists()
    assert registry.attempts[ident.attempt_id]["turns"] == []
    assert registry.seal(ident.attempt_id) == (ir["turns"], ir["trace_seal"])
    assert all(turn["delivery"] == "consumed_confirmed" for turn in ir["turns"])


@pytest.mark.parametrize("change", [
    {"content": [{"type": "text", "text": "missing"}]},
    {"finish_reason": "length"}, {"is_final": False},
])
def test_empty_final_rejects_other_missing_responses(tmp_path, change):
    class Registry:
        def unconsumed_responses(self, aid):
            return [{"response_id": "r", "content": [], "finish_reason": "end_turn",
                     "is_final": True, **change}]
    with pytest.raises(ValueError, match="only one empty final"):
        ClaudeCodeHarnessAdapter().audit_empty_final(tmp_path / "absent", Registry(), "a", {})


@pytest.mark.parametrize("change", [{"session_id": "wrong"}, {"is_error": True},
    {"subtype": "error_max_turns"}, {"result": "nonempty"}, {"stop_reason": "max_tokens"}])
def test_empty_final_requires_cli_consumption_evidence(tmp_path, change):
    class Registry:
        def unconsumed_responses(self, aid):
            return [{"response_id": "r", "content": [], "finish_reason": "end_turn", "is_final": True}]
    path = tmp_path / "claude-code.txt"
    path.write_text(json.dumps({"type": "result", "subtype": "success", "is_error": False,
        "result": "", "stop_reason": "end_turn", "session_id": "session",
        "terminal_reason": "completed", **change}))
    with pytest.raises(ValueError, match="matching session result"):
        ClaudeCodeHarnessAdapter().audit_empty_final(path, Registry(), "a", {"session_id": "session"})
