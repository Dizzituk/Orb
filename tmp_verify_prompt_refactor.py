"""Quick verification script — checks the prompt_builders refactor."""
from app.llm.routing.prompt_builders import (
    build_system_prompt,
    build_messages,
    build_full_context,
    _CONVERSATIONAL_GUIDELINES,
    _IMAGE_GEN_MARKER_INSTRUCTIONS,
)

print("Parent module imports: OK")
print(f"_CONVERSATIONAL_GUIDELINES: {len(_CONVERSATIONAL_GUIDELINES)} chars")
print(f"_IMAGE_GEN_MARKER_INSTRUCTIONS: {len(_IMAGE_GEN_MARKER_INSTRUCTIONS)} chars")
print()
print("=== Voice rewrite verification ===")
print("VOICE AND REGISTER section present:", "## VOICE AND REGISTER" in _CONVERSATIONAL_GUIDELINES)
print("Banned openers section present:    ", "Banned openers" in _CONVERSATIONAL_GUIDELINES)
print("Match-the-topic guidance present:  ", "Match the register of the topic" in _CONVERSATIONAL_GUIDELINES)
print("Match-the-input guidance present:  ", "Match the register of the input" in _CONVERSATIONAL_GUIDELINES)
print()
print("=== Old patterns removed ===")
acknowledge_line = "\n- Acknowledge the request"
print("Acknowledge-the-request bullet removed:", acknowledge_line not in _CONVERSATIONAL_GUIDELINES)
print("Old 'Got it' example removed:          ", 'GOOD: "Got it' not in _CONVERSATIONAL_GUIDELINES)
print("Old 'Makes sense' example removed:     ", 'GOOD: "Makes sense' not in _CONVERSATIONAL_GUIDELINES)
print("Rule 4 tightened (no reflex restate):  ", "Summarise scope back ONLY when ambiguous" in _CONVERSATIONAL_GUIDELINES)
print()
print("=== Preserved sections (sanity) ===")
print("ONE QUESTION AT A TIME still present:  ", "ONE QUESTION AT A TIME" in _CONVERSATIONAL_GUIDELINES)
print("Android disambiguation still present:  ", "ANDROID PROJECT DISAMBIGUATION" in _CONVERSATIONAL_GUIDELINES)
print("File generation still present:         ", "FILE GENERATION" in _CONVERSATIONAL_GUIDELINES)
print("Image marker instructions intact:      ", "[IMAGE_PROMPT]:" in _IMAGE_GEN_MARKER_INSTRUCTIONS)
