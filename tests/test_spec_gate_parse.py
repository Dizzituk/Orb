# FILE: tests/test_spec_gate_parse.py
import pytest

from app.pot_spec.spec_gate import parse_spec_gate_output


def test_parse_spec_gate_output_extracts_json():
    text = "noise\n{\n  \"goal\": \"x\",\n  \"requirements\": {\"must\": [], \"should\": [], \"can\": []},\n  \"constraints\": {},\n  \"acceptance_tests\": [],\n  \"open_questions\": [],\n  \"recommendations\": [],\n  \"repo_snapshot\": null\n}\ntrailing"
    draft = parse_spec_gate_output(text)
    assert draft.goal == "x"
