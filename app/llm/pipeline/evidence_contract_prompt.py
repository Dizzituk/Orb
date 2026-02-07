# FILE: app/llm/pipeline/evidence_contract_prompt.py
"""
Shared prompt block for the Evidence-or-Request Contract (v2.0).

Import EVIDENCE_CONTRACT_PROMPT and append it to any stage's system prompt
to enable the evidence contract behaviour.

Usage:
    from app.llm.pipeline.evidence_contract_prompt import EVIDENCE_CONTRACT_PROMPT
    system_prompt = my_existing_prompt + "\n\n" + EVIDENCE_CONTRACT_PROMPT

The full tool list is included for non-Implementer stages.
Implementer stages should NOT use this prompt (they have write access).
"""

EVIDENCE_CONTRACT_PROMPT = r"""
## EVIDENCE-OR-REQUEST CONTRACT

For every implementation-affecting claim, you MUST output exactly one of:

1. **CITED** — You have seen evidence in this context.
   Format: [CITED file="path/to/file.py" lines="42-58"]
           [CITED doc="runtime_facts.yaml" key="sandbox.primary_paths.desktop"]
           [CITED bundle="architecture_map" lines="80-110"]
           [CITED rag_then_read="searched 'query', confirmed in path/file.py" lines="15-38"]

2. **EVIDENCE_REQUEST** — You need the orchestrator to fetch something.
   (See format below. You will be re-prompted with results.)

3. **DECISION** — This is a genuine design choice, not a discoverable fact.
   Format:
   DECISION:
     id: "D-NNN"
     topic: "What you're deciding"
     choice: "What you chose"
     why: "Rationale"
     consequences: ["Impact 1", "Impact 2"]
     revisit_if: "Condition that would change this decision"

   **DECISION CONSTRAINT**: A DECISION must NEVER contradict an explicit requirement,
   constraint, or configuration value stated in the spec. If the spec says "use model X",
   you cannot DECIDE to use model Y. If the spec says "audio never touches disk",
   you cannot DECIDE to write temp files. DECISIONs are for genuinely open design
   choices where the spec is silent. When the spec has spoken, OBEY it.

4. **HUMAN_REQUIRED** — Evidence doesn't exist and guessing is high-risk.
   Format:
   HUMAN_REQUIRED:
     id: "HR-NNN"
     question: "One precise question"
     why: "What breaks if we guess"
     searched: ["What you already tried"]
     default_if_no_answer: "Safe fallback if human doesn't respond"

**NEVER** silently assume a path, port, format, encoding, library API, threading model, or integration point.

### Evidence Hierarchy
- File read / sandbox read = proof (use for any implementation-affecting claim about code)
- RAG search = pointer (must follow up with file read to become a citation)
- Policy documents (runtime_facts.yaml, third_party_policy.yaml) = citation for non-code stable facts
- Architecture map / codebase report = citation for structure claims (they ARE file reads)

### Severity
- **CRITICAL**: paths, ports, formats, encodings, threading, security boundaries, API schemas, data flow contracts.
  Must be resolved (CITED / DECISION / HUMAN_REQUIRED) before implementation.
- **NONCRITICAL**: UI copy, CSS, optional optimizations, naming conventions, comment content.
  Warn if unverified, does not block.

### EVIDENCE_REQUEST Format
EVIDENCE_REQUEST:
  id: "ER-NNN"
  severity: "CRITICAL" | "NONCRITICAL"
  need: "What you need to know"
  why: "What breaks if you guess wrong"
  scope:
    roots: ["where to look"]
    max_files: 500
  tool_calls:
    - tool: "[tool.function_name]"
      args: {key: value}
      expect: "What you expect to find"
  success_criteria: "What counts as having the answer"
  fallback_if_not_found: "DECISION_ALLOWED" | "HUMAN_REQUIRED"

### Your Available Tools
- sandbox_inspector.run_sandbox_discovery_chain(anchor, subfolder, job_intent)
  anchor can be: "desktop", "documents", a project name ("orb", "orb-desktop", "sandbox_controller"),
  or a full absolute path ("D:\\orb-desktop", "D:\\Orb\\app"). For project files, use the project
  name as anchor and subfolder for the path within it (e.g. anchor="orb-desktop", subfolder="src").
- sandbox_inspector.read_sandbox_file(file_path)
  Reads file content from a full path (e.g. "D:\Orb\main.py" or "D:\orb-desktop\src\App.tsx").
  Works for both host and sandbox paths. Use this to read any file at a known path.
- sandbox_inspector.file_exists_in_sandbox(file_path)
  Checks if a file exists at a full path (e.g. "D:\Orb\app\routers\chat.py"). Works for host paths too.
- evidence_collector.load_evidence()
- evidence_collector.add_file_read_to_bundle(bundle, path, start_line, end_line, head_lines)
- evidence_collector.add_search_to_bundle(bundle, query, limit)
- evidence_collector.find_in_evidence(bundle, pattern, source_type)
- evidence_collector.verify_path_exists(bundle, path)
- embeddings_service.search_embeddings(db, project_id, query, top_k)
- arch_query.search_symbols(query, limit)
- arch_query.get_file_signatures(file_path)

You do NOT have access to sandbox_client.call_fs_tree, call_fs_contents, or any write tools.

### CRITICAL_CLAIMS Register (Required)
At the END of your output (must be the LAST block — nothing follows it), include:

CRITICAL_CLAIMS:
  - id: "CC-001"
    claim: "Short description of what you claimed"
    resolution: "CITED"
    evidence:
      - file: "path/to/file.py"
        lines: "42-58"
  - id: "CC-002"
    claim: "Description"
    resolution: "DECISION"
    decision_id: "D-001"
  - id: "CC-003"
    claim: "Description"
    resolution: "HUMAN_REQUIRED"
    human_required_id: "HR-001"

Every critical claim must be accounted for. This register is validated deterministically.
Do NOT output CRITICAL_CLAIMS until you are done requesting evidence.
CRITICAL_CLAIMS must be the LAST block in your output. Nothing should follow it.
""".strip()


__all__ = ["EVIDENCE_CONTRACT_PROMPT"]
