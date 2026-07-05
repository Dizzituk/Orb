# FILE: tests/test_cost_pricing_derek.py
# Purpose: Derek phase 1 — pricing data file, unknown-model null semantics, env overrides.
# Called-by: pytest
# Depends-on: app.cost.cost_pricing
# Last-renovated: 2026-07-04
"""Pricing lookup tests for the Derek phase-1 cost telemetry build."""

import json

import pytest

from app.cost import cost_pricing


@pytest.fixture(autouse=True)
def _fresh_pricing():
    """Reload pricing around every test so cache state never leaks."""
    cost_pricing.reload_pricing()
    yield
    cost_pricing.reload_pricing()


def test_pricing_loads_from_data_file():
    src = cost_pricing.get_pricing_source()
    assert src["source"] == "file", f"expected data-file pricing, got {src}"
    table = cost_pricing.get_all_pricing()
    # Every model named in the build spec as live must be present.
    for model in (
        "claude-sonnet-5", "claude-opus-4-8", "claude-fable-5",
        "gpt-5.5", "gpt-5.4", "gpt-5.4-mini",
        "gemini-3.1-pro-preview", "claude-haiku-4-5-20251001",
        "qwen2.5-3b",
    ):
        assert model in table, f"{model} missing from pricing data file"


def test_known_model_prices():
    cost = cost_pricing.calculate_cost("claude-sonnet-5", 1_000_000, 1_000_000)
    assert cost == pytest.approx(18.0)
    # gpt-5.5 must no longer price at silent 0.00 (the pre-fix bug)
    assert cost_pricing.calculate_cost("gpt-5.5", 1_000_000, 0) > 0
    assert cost_pricing.calculate_cost("gemini-3.1-pro-preview", 1_000_000, 0) > 0


def test_unknown_model_returns_none_not_zero():
    assert cost_pricing.calculate_cost("model-that-does-not-exist-xyz", 1000, 1000) is None
    assert cost_pricing.get_model_price("model-that-does-not-exist-xyz") is None


def test_local_models_price_zero_not_null():
    # Local is free but KNOWN — tokens still recorded at 0.0, never null.
    assert cost_pricing.calculate_cost("qwen2.5-3b", 500_000, 50_000) == 0.0
    assert cost_pricing.calculate_cost("vllm_local", 1000, 1000) == 0.0


def test_fuzzy_match_prefers_longest_key():
    # A dated mini variant must match the mini entry, not the shorter base model.
    mini = cost_pricing.get_model_price("gpt-5.4-mini-2026-01-01")
    base = cost_pricing.get_model_price("gpt-5.4")
    assert mini is not None and base is not None
    assert mini.input_per_million < base.input_per_million


def test_env_path_override_and_missing_file_fallback(tmp_path, monkeypatch):
    # Custom file wins
    custom = tmp_path / "pricing.json"
    custom.write_text(json.dumps({
        "models": {"my-model": {"input_per_million": 1.0, "output_per_million": 2.0}}
    }), encoding="utf-8")
    monkeypatch.setenv("ASTRA_MODEL_PRICING_PATH", str(custom))
    cost_pricing.reload_pricing()
    assert cost_pricing.calculate_cost("my-model", 1_000_000, 1_000_000) == pytest.approx(3.0)
    # The default table's models are gone under the custom file
    assert cost_pricing.get_model_price("claude-sonnet-5") is None

    # Missing file falls back to the built-in table with a WARN (never raises)
    monkeypatch.setenv("ASTRA_MODEL_PRICING_PATH", str(tmp_path / "nope.json"))
    n = cost_pricing.reload_pricing()
    assert n > 0
    assert cost_pricing.get_pricing_source()["source"] == "built-in fallback"
    assert cost_pricing.calculate_cost("gpt-5.4", 1_000_000, 0) == pytest.approx(2.5)


def test_per_model_env_override(monkeypatch):
    monkeypatch.setenv("ORB_PRICE_GPT_5_5_INPUT", "9.99")
    cost_pricing.reload_pricing()
    assert cost_pricing.calculate_cost("gpt-5.5", 1_000_000, 0) == pytest.approx(9.99)
