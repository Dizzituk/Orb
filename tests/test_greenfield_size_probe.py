# FILE: tests/test_greenfield_size_probe.py
# Purpose: live10 — greenfield size analysis probes ONLY the greenfield root, never the ASTRA repos.
# Called-by: pytest
# Depends-on: app.pot_spec.grounded.size_analyzer, app.pot_spec.grounded._spec_runner_segmentation
# Last-renovated: 2026-07-05
"""First live planned-manifest run (2026-07-05 00:13): the size analyzer
resolved 16 planned Tetris files against DEFAULT_PROJECT_ROOTS (D:\\Orb,
D:\\orb-desktop) through the sandbox-routed fs — sandbox down, 15s timeout
per probe, ~8 minutes of wedge for files that cannot exist anywhere yet."""

from app.pot_spec.grounded import size_analyzer
from app.pot_spec.grounded import _spec_runner_segmentation as seg

PLANNED = ["src\\main.py", "src\\board.py", "src\\tetromino.py"]


def _record_probes(monkeypatch):
    probed = []

    def fake_isfile(path):
        probed.append(path.replace("\\", "/"))
        return False

    monkeypatch.setattr(size_analyzer, "_sbx_isfile", fake_isfile)
    return probed


def test_explicit_roots_probe_only_those_roots(monkeypatch, tmp_path):
    probed = _record_probes(monkeypatch)
    root = str(tmp_path / "Tazzas Tetris")

    result = size_analyzer.analyze_file_sizes(
        PLANNED, spec_markdown="# spec", project_roots=[root],
    )

    assert all(p.startswith(root.replace("\\", "/")) for p in probed), probed
    assert not any("/Orb/" in p or "orb-desktop" in p for p in probed), probed
    assert result.source_files_analyzed == 0  # nothing exists yet — CREATE targets


def test_default_roots_still_used_without_override(monkeypatch):
    """Non-greenfield callers keep the existing behaviour."""
    probed = _record_probes(monkeypatch)
    size_analyzer.analyze_file_sizes(["src\\one.py"], spec_markdown="# spec")
    assert probed, "default roots should be probed when no override given"
    assert any("Orb" in p or "orb-desktop" in p for p in probed)


def test_run_size_analysis_threads_greenfield_root(monkeypatch, tmp_path):
    probed = _record_probes(monkeypatch)
    root = str(tmp_path / "NewGame")

    seg._run_size_analysis(list(PLANNED), "# spec", greenfield_root=root)

    assert probed, "greenfield root should be probed"
    assert all(p.startswith(root.replace("\\", "/")) for p in probed), probed
