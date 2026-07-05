# FILE: tests/test_host_launcher.py
# Purpose: live12 — src/ entrypoint discovery + deterministic Launch.bat at project root.
# Called-by: pytest
# Depends-on: app.pipeline_v2.host_launcher, app.pipeline_v2.verifier_agent.host_perception
# Last-renovated: 2026-07-05
"""First full E2E judged FAIL solely on 'no entrypoint found' — the entry was
src/main.py and only root-level names were searched. Also pins Taz's standing
requirement: every external build ships a double-clickable Launch.bat."""

import sys

from app.pipeline_v2.host_launcher import ensure_launcher, LAUNCHER_NAME
from app.pipeline_v2.verifier_agent.host_perception import guess_entrypoint


def test_root_main_still_wins(tmp_path):
    (tmp_path / "main.py").write_text("print('root')", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('src')", encoding="utf-8")
    cmd = guess_entrypoint(str(tmp_path))
    assert cmd == [sys.executable, str(tmp_path / "main.py")]


def test_src_main_found_when_root_empty(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('src')", encoding="utf-8")
    cmd = guess_entrypoint(str(tmp_path))
    assert cmd == [sys.executable, str(tmp_path / "src" / "main.py")]


def test_single_py_in_src_fallback(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "tetris_game.py").write_text("print('game')", encoding="utf-8")
    cmd = guess_entrypoint(str(tmp_path))
    assert cmd == [sys.executable, str(tmp_path / "src" / "tetris_game.py")]


def test_no_entry_returns_none(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("", encoding="utf-8")
    assert guess_entrypoint(str(tmp_path)) is None


def test_launcher_written_and_points_at_entry(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('go')", encoding="utf-8")
    out = ensure_launcher(str(tmp_path), "Tazza's Tetris")
    assert out is not None
    text = (tmp_path / LAUNCHER_NAME).read_text(encoding="utf-8")
    assert sys.executable in text
    assert str(tmp_path / "src" / "main.py") in text
    assert "|| pause" in text
    assert "Tazza's Tetris" in text


def test_launcher_overwritten_each_build(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('go')", encoding="utf-8")
    (tmp_path / LAUNCHER_NAME).write_text("stale", encoding="utf-8")
    ensure_launcher(str(tmp_path), "Game")
    assert "stale" not in (tmp_path / LAUNCHER_NAME).read_text(encoding="utf-8")


def test_no_entry_no_launcher(tmp_path):
    assert ensure_launcher(str(tmp_path), "Empty") is None
    assert not (tmp_path / LAUNCHER_NAME).exists()
