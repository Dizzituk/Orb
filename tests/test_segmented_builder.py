# FILE: tests/test_segmented_builder.py
# Purpose: Derek phase 5 gate — order, scope enforcement, isolation, recombination, re-dispatch, escalation.
# Called-by: pytest
# Depends-on: app.pipeline_v2.segmented
# Last-renovated: 2026-07-04
"""Gate 5: mocked-LLM integration tests for the segmented builder. The LLM
is faked; everything else is real — scoped writes land on a temp greenfield
root, the ledger persists, the integrator py_compiles real files."""

import asyncio
import json

import pytest

from app.pipeline_v2 import llm_tools
from app.pipeline_v2.build_targets import BuildTargetProfile
from app.pipeline_v2.segmented.scheduler import SchedulingError, schedule_order, schedule_waves
from app.pipeline_v2.segmented.segmented_builder import run_segmented_builder
from app.pipeline_v2.segmented.worker import COMPLETE_MARKER, path_in_scope
from app.pot_spec.grounded.segment_schemas import SegmentSpec
from app.pot_spec.grounded._segment_schemas_utils_1 import InterfaceContract, SegmentManifest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _profile(tmp_path) -> BuildTargetProfile:
    return BuildTargetProfile(
        project_id="derek-probe-toy",
        project_name="Derek Probe Toy",
        project_root=str(tmp_path),
        language="python",
        build_system="pip",
        framework="cli",
        source_root="",
        package_name="",
        architecture_pattern="flat",
    )


def _toy_manifest() -> SegmentManifest:
    seg1 = SegmentSpec(
        segment_id="seg-01-core", title="Core maths module",
        file_scope=["core.py"],
        requirements=["Implement add(a, b) returning a + b"],
        exposes=InterfaceContract(class_names=[], method_signatures=["def add(a, b)"],
                                  endpoint_paths=[], export_names=["add"]),
    )
    seg2 = SegmentSpec(
        segment_id="seg-02-cli", title="CLI wrapper",
        file_scope=["cli.py"],
        dependencies=["seg-01-core"],
        requirements=["CLI that adds two numbers using core.add"],
        consumes=InterfaceContract(class_names=[], method_signatures=[],
                                   endpoint_paths=[], export_names=["add"]),
    )
    m = SegmentManifest(segments=[seg1, seg2])
    m.total_segments = 2
    m.total_files = 2
    return m


GOOD_CORE = "def add(a, b):\n    return a + b\n"
GOOD_CLI = (
    "import sys\nfrom core import add\n\n"
    "def main():\n    print(add(int(sys.argv[1]), int(sys.argv[2])))\n\n"
    "if __name__ == '__main__':\n    main()\n"
)
BROKEN_CORE = "def add(a, b:\n    return a + b\n"  # syntax error


def _completion(files, exposes=None):
    return (
        f"{COMPLETE_MARKER}\n"
        + json.dumps({
            "files_written": files,
            "exposes": exposes or {},
            "decisions": [{"key": "style", "value": "stdlib only", "why": "toy"}],
            "notes": "done",
        })
    )


class FakeLLM:
    """Simulates worker models: writes files through the REAL scoped executor."""

    def __init__(self, behaviours):
        # behaviours: {segment_id: [round0_files, round1_files, ...]} where
        # files is {path: content}; last entry repeats for later rounds.
        self.behaviours = behaviours
        self.calls = []       # (segment_id, model, stage)
        self.rounds = {}

    def _segment_of(self, prompt: str) -> str:
        import re as _re
        m = _re.search(r"larger job: (\S+)", prompt)
        return m.group(1) if m else "?"

    async def run_tool_loop(self, *, system_prompt, initial_user_message, provider,
                            model, max_iterations, max_tokens, on_tool_call=None,
                            on_text=None, existing_messages=None, reasoning=None,
                            stage=None, job_id=None, tool_executor=None,
                            max_total_tokens=None):
        sid = self._segment_of(initial_user_message)
        self.calls.append((sid, model, stage))
        n = self.rounds.get(sid, 0)
        self.rounds[sid] = n + 1
        plan = self.behaviours.get(sid) or [{}]
        files = plan[min(n, len(plan) - 1)]
        written = []
        for path, content in files.items():
            if on_tool_call:
                on_tool_call("write_file", {"path": path, "content": content})
            result = await tool_executor("write_file", {"path": path, "content": content})
            if not str(result).startswith("SCOPE VIOLATION"):
                written.append(path)
        text = _completion(written)
        if on_text:
            on_text(text)
        return [{"role": "assistant", "content": text}], 1000, 200


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTRA_STAGE_BUILDER_WORKER_PROVIDER", "openai")
    monkeypatch.setenv("ASTRA_STAGE_BUILDER_WORKER_MODEL", "hands-standin-mini")
    monkeypatch.setenv("FRONTIER_PROVIDER", "anthropic")
    monkeypatch.setenv("FRONTIER_MODEL", "apex-escalation-model")
    monkeypatch.setenv("ASTRA_SEGMENT_MAX_REDISPATCH", "2")
    monkeypatch.setenv("ASTRA_V2_LEDGER_ENABLED", "true")
    yield


def _run(manifest, tmp_path, fake, monkeypatch):
    monkeypatch.setattr(llm_tools, "run_tool_loop", fake.run_tool_loop)
    job_dir = str(tmp_path / "jobdir")
    return asyncio.run(run_segmented_builder(
        spec={}, manifest=manifest, scaffold=None,
        job_dir=job_dir, job_id="probe-job-1",
        profile=_profile(tmp_path),
    ))


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def test_scheduler_topological_order_with_tiebreak():
    m = _toy_manifest()
    order = [s.segment_id for s in schedule_order(m.segments)]
    assert order == ["seg-01-core", "seg-02-cli"]
    waves = schedule_waves(m.segments)
    assert [[s.segment_id for s in w] for w in waves] == [["seg-01-core"], ["seg-02-cli"]]


def test_scheduler_rejects_cycles():
    a = SegmentSpec(segment_id="a", title="a", dependencies=["b"])
    b = SegmentSpec(segment_id="b", title="b", dependencies=["a"])
    with pytest.raises(SchedulingError):
        schedule_order([a, b])


def test_scope_matcher():
    assert path_in_scope("core.py", ["core.py"])
    assert path_in_scope("src/game/core.py", ["core.py"]) is True  # suffix-tolerant
    assert not path_in_scope("evil.py", ["core.py"])
    # p7 review fix: parent-traversal can never suffix-match its way in
    assert not path_in_scope("../core.py", ["core.py"])
    assert not path_in_scope("core.py/../../evil.py", ["core.py"])
    assert not path_in_scope("..\\..\\core.py", ["core.py"])
    assert path_in_scope("src/./core.py", ["core.py"])  # benign dot segments still fine


# ---------------------------------------------------------------------------
# End-to-end (mocked LLM, real files/ledger/integrator)
# ---------------------------------------------------------------------------

def test_happy_path_order_isolation_recombination(env, tmp_path, monkeypatch):
    fake = FakeLLM({
        "seg-01-core": [{"core.py": GOOD_CORE}],
        "seg-02-cli": [{"cli.py": GOOD_CLI}],
    })
    result = _run(_toy_manifest(), tmp_path, fake, monkeypatch)

    # Order: core before cli, one fresh context each
    assert [c[0] for c in fake.calls] == ["seg-01-core", "seg-02-cli"]
    # Workers ran on the HANDS stand-in with worker attribution
    assert all(c[1] == "hands-standin-mini" for c in fake.calls)
    assert all(c[2] == "builder_worker" for c in fake.calls)
    # Real files on the greenfield root
    assert (tmp_path / "core.py").read_text(encoding="utf-8") == GOOD_CORE
    assert (tmp_path / "cli.py").read_text(encoding="utf-8") == GOOD_CLI
    # Recombination: integrator green, BuildResult contract satisfied
    assert result.success, result.errors
    assert sorted(result.all_files_written) == ["cli.py", "core.py"]
    assert result.total_input_tokens == 2000
    # Ledger holds worker decisions
    ledger_file = tmp_path / "jobdir" / "decision_ledger.json"
    assert ledger_file.exists()
    entries = json.loads(ledger_file.read_text(encoding="utf-8"))["entries"]
    assert any(e.get("stage") == "builder_worker" and e.get("type") == "decision" for e in entries)


def test_scope_violation_rejected_and_flagged(env, tmp_path, monkeypatch):
    fake = FakeLLM({
        "seg-01-core": [{"core.py": GOOD_CORE, "outside/hack.py": "x = 1\n"}],
        "seg-02-cli": [{"cli.py": GOOD_CLI}],
    })
    result = _run(_toy_manifest(), tmp_path, fake, monkeypatch)
    assert not (tmp_path / "outside" / "hack.py").exists(), "out-of-scope write must not land"
    assert (tmp_path / "core.py").exists()
    assert any("outside file_scope" in e or "scope" in e.lower() for e in result.errors)


def test_failure_redispatch_then_pass(env, tmp_path, monkeypatch):
    fake = FakeLLM({
        "seg-01-core": [{"core.py": BROKEN_CORE}, {"core.py": GOOD_CORE}],  # broken then fixed
        "seg-02-cli": [{"cli.py": GOOD_CLI}],
    })
    result = _run(_toy_manifest(), tmp_path, fake, monkeypatch)
    # seg-01 ran twice: initial + one re-dispatch with failure evidence
    assert fake.rounds["seg-01-core"] == 2
    assert result.success, result.errors
    # Failure evidence flagged in the ledger
    entries = json.loads((tmp_path / "jobdir" / "decision_ledger.json").read_text(encoding="utf-8"))["entries"]
    assert any(e.get("stage") == "builder_integrator" and e.get("type") == "flag" for e in entries)


def test_escalation_to_apex_after_bounded_redispatches(env, tmp_path, monkeypatch):
    fake = FakeLLM({
        "seg-01-core": [{"core.py": BROKEN_CORE}],  # always broken
        "seg-02-cli": [{"cli.py": GOOD_CLI}],
    })
    result = _run(_toy_manifest(), tmp_path, fake, monkeypatch)
    seg1_calls = [c for c in fake.calls if c[0] == "seg-01-core"]
    # initial + 2 re-dispatches + 1 escalation = 4
    assert len(seg1_calls) == 4
    assert seg1_calls[-1][1] == "apex-escalation-model"
    assert seg1_calls[-1][2] == "builder_escalation"
    assert not result.success
    assert any("syntax" in e.lower() or "integration" in e.lower() for e in result.errors)
