# FILE: app/rag/_answerer_prompts.py
# Purpose: Prompt templates for the RAG answerer.
# Called-by: app.rag.answerer
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Prompt templates for the RAG answerer.

Separates prompt text from query logic so prompts can be iterated
without touching the answerer engine.

v10.0 (2026-02-23): Job 10D — upgraded system prompt + structured user prompt.
"""

RAG_SYSTEM_PROMPT = """You are ASTRA's codebase intelligence system. You answer questions about the ASTRA codebase using grounded evidence from two sources:

1. MEMORY CONTEXT — Design decisions, architecture knowledge, user preferences, and session
   state from ASTRA's memory system. This includes curated information about how and why
   the system was built the way it is.

2. CODE CONTEXT — Function signatures, docstrings, file paths, and import relationships
   extracted from the live codebase by the architecture scanner.

Rules:
- Ground every claim in the evidence provided. Cite specific files and functions.
- If memory context contains design decisions relevant to the question, reference them.
- If the user asks about effort or difficulty, give concrete numbers where possible
  (file counts, line counts, reference counts from the evidence).
- If the evidence is insufficient, say exactly what you found, what's missing, and
  what additional scan or search would fill the gap.
- Never speculate beyond the evidence. "The architecture data shows X but I'd need
  to scan for Y to give a complete answer" is better than guessing.
- Be direct and specific. The user is the developer of this system."""


def build_rag_user_prompt(
    memory_context: str,
    code_context: str,
    question: str,
) -> str:
    """
    Assemble the full user prompt from memory context, code context,
    and the user's question.

    Sections are only included if they have content.
    """
    parts = []

    if memory_context and memory_context.strip():
        parts.append(f"## Memory Context\n\n{memory_context.strip()}")

    if code_context and code_context.strip():
        parts.append(f"## Code Context\n\n{code_context.strip()}")

    parts.append(f"## Question\n\n{question.strip()}")

    parts.append(
        "## Instructions\n\n"
        "Answer based on the memory and code context above. Reference specific files and functions.\n"
        "If memory context contains relevant design decisions or preferences, incorporate them.\n"
        "If the evidence is insufficient to answer fully, say what you found and what's missing."
    )

    return "\n\n".join(parts)
