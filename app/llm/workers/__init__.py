# Purpose: Local agent-worker pool package (Nat now; the 26B fleet later).
# Workers are GENERIC: they take a job over the standard call_llm_text path and
# know nothing about their callers (memory, debug, orchestrator, ...).
