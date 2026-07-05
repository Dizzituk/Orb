# FILE: tests/test_bridge_study_flow_block.py
"""STUDY FLOW prompt block (2026-07-03): bridge/voice sources only.

Live transcript problem: three user turns per lecture transition (announce
-> act-without-content -> summarise on a third prompt) plus a needless
"which browser session?" question. The block tells the model to act in the
same turn; it must reach bridge sources and never the desktop or Room.
"""
from app.bridge.study_flow_prompt import STUDY_FLOW_BLOCK, study_flow_block


class TestStudyFlowBlockGating:
    def test_present_for_bridge_tts(self):
        assert study_flow_block("bridge-tts") == STUDY_FLOW_BLOCK

    def test_present_for_bridge_chat(self):
        assert study_flow_block("bridge") == STUDY_FLOW_BLOCK

    def test_absent_for_desktop_room_and_none(self):
        assert study_flow_block("desktop") == ""
        assert study_flow_block("room") == ""
        assert study_flow_block("") == ""
        assert study_flow_block(None) == ""


class TestStudyFlowBlockContent:
    def test_names_the_real_study_tools(self):
        for name in ("coursera_resume", "coursera_next_item", "coursera_read_lesson"):
            assert name in STUDY_FLOW_BLOCK

    def test_forbids_announce_only_turns(self):
        assert "Never end a turn having only announced an action" in STUDY_FLOW_BLOCK

    def test_forbids_the_session_question(self):
        assert "Never ask which browser" in STUDY_FLOW_BLOCK
