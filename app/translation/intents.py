# FILE: app/translation/intents.py
"""
Canonical intent definitions for ASTRA Translation Layer.
This is the CLOSED SET of allowed intents. No free-form execution.

v1.3 (2026-01): Added LATEST_ARCHITECTURE_MAP and LATEST_CODEBASE_REPORT_FULL intents
v1.2 (2026-01): Fixed Overwatcher gating - removed change_set_id requirement,
                added proper trigger patterns for "run overwatcher"
v1.1 (2026-01): Added Spec Gate flow intents (WEAVER_BUILD_SPEC, SEND_TO_SPEC_GATE)
"""
from __future__ import annotations
from typing import Dict, List, Optional
from .schemas import CanonicalIntent, IntentDefinition


# =============================================================================
# CANONICAL INTENT DEFINITIONS
# =============================================================================

INTENT_DEFINITIONS: Dict[CanonicalIntent, IntentDefinition] = {
    
    # -------------------------------------------------------------------------
    # ARCHITECTURE COMMANDS
    # -------------------------------------------------------------------------
    
    CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES: IntentDefinition(
        intent=CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES,
        trigger_phrases=[
            "CREATE ARCHITECTURE MAP",  # ALL CAPS required
            "Astra, command: CREATE ARCHITECTURE MAP",
            "astra, command: CREATE ARCHITECTURE MAP",
        ],
        trigger_patterns=[
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?CREATE ARCHITECTURE MAP$",  # ALL CAPS required
            r"^CREATE ARCHITECTURE MAP\s+for\s+",  # With target
        ],
        requires_context=[],  # Optional: repo target, job_id
        requires_confirmation=False,
        description="Generate a full architecture map including full folder + file structure",
        behavior=(
            "Generate a full architecture map including:\n"
            "- System diagram\n"
            "- Components and relationships\n"
            "- Full folder + file tree\n"
            "Output as job artifact in jobs/<job_id>/arch/arch_vN.md\n"
            "Use for deep analysis or onboarding."
        ),
    ),
    
    CanonicalIntent.ARCHITECTURE_MAP_STRUCTURE_ONLY: IntentDefinition(
        intent=CanonicalIntent.ARCHITECTURE_MAP_STRUCTURE_ONLY,
        trigger_phrases=[
            "Create architecture map",   # Normal/lowercase
            "create architecture map",
            "Astra, command: Create architecture map",
            "astra, command: create architecture map",
        ],
        trigger_patterns=[
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?[Cc]reate [Aa]rchitecture [Mm]ap$",  # Case-insensitive but NOT all caps
            r"^[Cc]reate [Aa]rchitecture [Mm]ap\s+for\s+",
        ],
        requires_context=[],
        requires_confirmation=False,
        description="Generate a logical architecture map ONLY (no file tree dump)",
        behavior=(
            "Generate a logical architecture map ONLY.\n"
            "NO file tree dump.\n"
            "Based on current Code Atlas.\n"
            "Output same location, lighter content."
        ),
    ),
    
    CanonicalIntent.ARCHITECTURE_UPDATE_ATLAS_ONLY: IntentDefinition(
        intent=CanonicalIntent.ARCHITECTURE_UPDATE_ATLAS_ONLY,
        trigger_phrases=[
            "update architecture",
            "Update architecture",
            "Astra, command: update architecture",
            "astra, command: update architecture",
        ],
        trigger_patterns=[
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?[Uu]pdate [Aa]rchitecture$",
            r"^[Uu]pdate your [Aa]rchitecture$",
            r"^[Rr]efresh [Aa]rchitecture$",
        ],
        requires_context=[],
        requires_confirmation=False,
        description="Refresh Code Atlas / repo understanding (internal memory only)",
        behavior=(
            "Refresh Code Atlas / repo understanding.\n"
            "Internal memory ONLY.\n"
            "NO markdown outputs.\n"
            "NO Claude involvement unless summaries needed.\n"
            "Uses incremental update."
        ),
    ),
    
    # -------------------------------------------------------------------------
    # SANDBOX CONTROL
    # -------------------------------------------------------------------------
    
    CanonicalIntent.START_SANDBOX_ZOMBIE_SELF: IntentDefinition(
        intent=CanonicalIntent.START_SANDBOX_ZOMBIE_SELF,
        trigger_phrases=[
            "Start your zombie",
            "start your zombie",
        ],
        trigger_patterns=[
            r"^[Ss]tart your [Zz]ombie$",
            r"^[Ss]tart the [Zz]ombie$",
            r"^[Ll]aunch [Zz]ombie$",
            r"^[Ss]pin up [Zz]ombie$",
        ],
        requires_context=[],  # sandbox_id optional
        requires_confirmation=False,
        description="Start Windows Sandbox Zombie instance & agent",
        behavior=(
            "Host should:\n"
            "1. Detect if sandbox already running\n"
            "2. If not:\n"
            "   - Start Windows Sandbox with predefined config\n"
            "   - Launch sandbox agent\n"
            "   - Agent clones repo into sandbox working dir\n"
            "   - Agent phones home\n"
            "3. Host marks sandbox READY\n"
            "\n"
            "Never writes to main repo.\n"
            "Promotion remains manual via GitHub Desktop."
        ),
    ),
    
    CanonicalIntent.SCAN_SANDBOX_STRUCTURE: IntentDefinition(
        intent=CanonicalIntent.SCAN_SANDBOX_STRUCTURE,
        trigger_phrases=[
            # Primary triggers
            "scan sandbox",
            "Scan sandbox",
            "SCAN SANDBOX",
            # Astra command variants
            "Astra, command: scan sandbox",
            "astra, command: scan sandbox",
            # Legacy full phrase
            "SCAN SANDBOX STRUCTURE",
            "Scan sandbox structure",
            "scan sandbox structure",
        ],
        trigger_patterns=[
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?[Ss]can\s+[Ss]andbox$",
            r"^SCAN SANDBOX(?: STRUCTURE)?$",
            r"^[Ss]can sandbox structure$",
            r"^[Ss]can the sandbox(?: structure)?$",
        ],
        requires_context=[],
        requires_confirmation=False,
        description=(
            "Trigger a read-only scan of the sandbox filesystem "
            "(Desktop, Documents, Downloads, backend, frontend) "
            "and build a flattened Sandbox Index."
        ),
        behavior=(
            "Host should:\n"
            "1. Call the Architecture Query Service sandbox project scan endpoint\n"
            "2. Walk configured sandbox roots (Desktop/Documents/Downloads/backend/frontend)\n"
            "3. Apply ignore rules (node_modules, .git, .venv, dist, build, logs, etc.)\n"
            "4. Store a flat list of file entries (path/name/ext/zone) in memory/cache\n"
            "5. Expose the index via /sandbox/index for later LOCATE FILE/FOLDER commands.\n"
            "\n"
            "STRICTLY READ-ONLY: no file writes, no deletes, no recycle-bin operations."
        ),
    ),

    
    # -------------------------------------------------------------------------
    # SPEC GATE FLOW (v1.1)
    # -------------------------------------------------------------------------
    
    CanonicalIntent.WEAVER_BUILD_SPEC: IntentDefinition(
        intent=CanonicalIntent.WEAVER_BUILD_SPEC,
        trigger_phrases=[
            # Natural conversation triggers
            "How does that look all together",
            "How does that look all together?",
            "how does that look all together",
            "how does that look all together?",
            # Explicit weave commands
            "Weave this into a spec",
            "weave this into a spec",
            "Build spec from ramble",
            "build spec from ramble",
            "Compile the spec",
            "compile the spec",
            # Summary/consolidation triggers
            "Put that all together",
            "put that all together",
            "Consolidate that into a spec",
            "consolidate that into a spec",
        ],
        trigger_patterns=[
            r"^[Hh]ow does that look all together\??$",
            r"^[Ww]eave (?:this|that) into a spec$",
            r"^[Bb]uild (?:a )?spec from (?:the )?ramble$",
            r"^[Cc]ompile (?:the )?spec$",
            r"^[Pp]ut (?:that|this|it) all together$",
            r"^[Cc]onsolidate (?:that|this) into a spec$",
            r"^[Ss]ummarize (?:the|my) ramble into a spec$",
            r"^[Tt]urn (?:this|that) into a spec$",
        ],
        requires_context=[],  # Will pull from conversation memory
        requires_confirmation=False,
        description="Trigger Weaver to build a candidate spec from ramble/conversation",
        behavior=(
            "Weaver (GPT-5.2 latest) is triggered to:\n"
            "1. Pull all relevant ramble material from memory\n"
            "2. Build one coherent candidate spec\n"
            "3. Read it back to user in natural language\n"
            "4. Point out obvious weak spots or contradictions\n"
            "5. Integrate any clarifications\n"
            "6. Produce a refined candidate spec\n"
            "\n"
            "Does NOT send to Spec Gate automatically.\n"
            "User must explicitly say 'Send to Spec Gate' after review."
        ),
    ),
    
    CanonicalIntent.SEND_TO_SPEC_GATE: IntentDefinition(
        intent=CanonicalIntent.SEND_TO_SPEC_GATE,
        trigger_phrases=[
            # Explicit send commands
            "Send that to Spec Gate",
            "send that to Spec Gate",
            "Send to Spec Gate",
            "send to Spec Gate",
            "Okay, send that to Spec Gate",
            "okay, send that to Spec Gate",
            "Ok, send that to Spec Gate",
            "ok, send that to Spec Gate",
            # Validate commands
            "Validate the spec",
            "validate the spec",
            "Run Spec Gate",
            "run Spec Gate",
            # Submit commands
            "Submit spec for validation",
            "submit spec for validation",
        ],
        trigger_patterns=[
            r"^(?:[Oo]k(?:ay)?,?\s*)?[Ss]end (?:that|this|it) to [Ss]pec ?[Gg]ate$",
            r"^[Ss]end to [Ss]pec ?[Gg]ate$",
            r"^[Vv]alidate (?:the )?spec$",
            r"^[Rr]un [Ss]pec ?[Gg]ate$",
            r"^[Ss]ubmit (?:the )?spec(?: for validation)?$",
            r"^[Ss]pec ?[Gg]ate[,:]?\s*validate$",
        ],
        requires_context=[],  # Auto-fetches latest draft spec
        requires_confirmation=False,  # Not high-stakes until pipeline runs
        description="Send refined candidate spec to Spec Gate for validation",
        behavior=(
            "Spec Gate (GPT-5.2 Pro) receives the refined candidate spec.\n"
            "It evaluates for:\n"
            "- Completeness\n"
            "- Consistency\n"
            "- Unambiguous behaviour\n"
            "- Safety / risk coverage\n"
            "\n"
            "Either:\n"
            "- Approves directly (spec_valid=true, spec_id, spec_hash)\n"
            "- Returns structured questions to patch real gaps\n"
            "\n"
            "Questions go through Mediator (GPT-5.2 latest) for user interaction.\n"
            "NO automatic pipeline execution - just validation."
        ),
    ),
    
    # -------------------------------------------------------------------------
    # HIGH-STAKES PIPELINE CONTROL (require confirmation)
    # -------------------------------------------------------------------------
    
    CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB: IntentDefinition(
        intent=CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB,
        trigger_phrases=[
            "Run critical pipeline",
            "run critical pipeline",
            "Execute critical pipeline",
            "Start the pipeline",
            "start the pipeline",
        ],
        trigger_patterns=[
            r"^[Rr]un (?:the )?[Cc]ritical [Pp]ipeline$",
            r"^[Rr]un (?:the )?[Cc]ritical [Pp]ipeline for job\s+",
            r"^[Ee]xecute (?:the )?[Cc]ritical [Pp]ipeline$",
            r"^[Ss]tart the pipeline$",
        ],
        requires_context=["job_id", "spec_id"],  # MUST have validated spec
        requires_confirmation=True,
        confirmation_prompt=(
            "⚠️ HIGH-STAKES OPERATION\n"
            "You are about to run the Critical Pipeline for job {job_id}.\n"
            "Spec: {spec_id}\n"
            "This will execute the full verification and execution flow.\n"
            "\n"
            "Type 'Yes' to confirm."
        ),
        description="Execute the critical pipeline for a validated spec",
        behavior=(
            "Execute full critical pipeline:\n"
            "1. Verify spec_valid=true and spec_id exists\n"
            "2. Intent detected and restated\n"
            "3. User confirms 'Yes'\n"
            "4. Only then execution begins\n"
            "\n"
            "NO silent execution.\n"
            "Requires validated spec from Spec Gate."
        ),
    ),
    
    # -------------------------------------------------------------------------
    # OVERWATCHER (v1.2 - FIXED GATING)
    # -------------------------------------------------------------------------
    # Overwatcher is the SYSTEM SUPERVISOR, not just a pipeline stage.
    # It should run when:
    #   - A validated spec exists (spec_id + spec_hash)
    #   - Critical Pipeline completed for that spec
    # It should NOT hard-require:
    #   - change_set_id (Overwatcher derives/creates this)
    #   - "no blocking issues" (Overwatcher evaluates these itself)
    # -------------------------------------------------------------------------
    
    CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES: IntentDefinition(
        intent=CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES,
        trigger_phrases=[
            # Primary triggers (v1.2)
            "run overwatcher",
            "Run overwatcher",
            "Run Overwatcher",
            "execute overwatcher",
            "Execute overwatcher",
            "Execute Overwatcher",
            "start overwatcher",
            "Start overwatcher",
            "Start Overwatcher",
            # Astra command variants
            "Astra, command: run overwatcher",
            "astra, command: run overwatcher",
            "Astra command run overwatcher",
            # Legacy triggers
            "Execute overwatcher changes",
            "Apply overwatcher changes",
            # Send to overwatcher
            "send to overwatcher",
            "Send to overwatcher",
            "Send to Overwatcher",
        ],
        trigger_patterns=[
            # Primary patterns (v1.2)
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?[Rr]un\s+[Oo]verwatcher$",
            r"^(?:[Aa]stra[,:]?\s*)?[Ee]xecute\s+[Oo]verwatcher$",
            r"^(?:[Aa]stra[,:]?\s*)?[Ss]tart\s+[Oo]verwatcher$",
            r"^[Oo]verwatcher\s+run$",
            r"^[Rr]un\s+the\s+[Oo]verwatcher$",
            r"^[Ii]nvoke\s+[Oo]verwatcher$",
            r"^[Tt]rigger\s+[Oo]verwatcher$",
            r"^[Ss]end\s+to\s+[Oo]verwatcher$",
            # Legacy patterns
            r"^[Ee]xecute [Oo]verwatcher [Cc]hanges$",
            r"^[Aa]pply [Oo]verwatcher [Cc]hanges$",
            r"^[Oo]verwatcher[,:]\s*execute$",
        ],
        # v1.2: REMOVED change_set_id - Overwatcher resolves/derives this internally
        # Overwatcher-specific gating happens in gates.py check_overwatcher_gate()
        requires_context=[],
        requires_confirmation=False,  # Overwatcher handles its own safety checks
        confirmation_prompt=None,
        description="Run Overwatcher supervisor to execute approved changes",
        behavior=(
            "Overwatcher is the SYSTEM SUPERVISOR.\n"
            "\n"
            "Responsibilities:\n"
            "- Final safety + correctness decision after Critical Pipeline\n"
            "- Reasoning over remaining issues with full system awareness\n"
            "- Coordinating execution in Windows Sandbox\n"
            "- Supervising implementation jobs (Implementer = Claude Sonnet)\n"
            "- Validating outputs and logging evidence\n"
            "- Deciding whether work is acceptable to proceed\n"
            "\n"
            "Gating (checked in gates.py):\n"
            "- REQUIRES: validated spec (spec_id + spec_hash) resolvable\n"
            "- REQUIRES: Critical Pipeline completed for that spec\n"
            "- NOT REQUIRED: change_set_id (Overwatcher derives internally)\n"
            "- NOT REQUIRED: zero blocking issues (Overwatcher evaluates these)\n"
            "\n"
            "Blocking issue handling:\n"
            "- Overwatcher reads critic's blocking list\n"
            "- Reasons about each with system-level knowledge\n"
            "- Decides severity itself (hard-stop vs proceed-with-warning)\n"
            "- Logs override decisions in evidence bundle\n"
            "\n"
            "NO fallback to chat if context missing - structured error instead."
        ),
    ),
    
    # -------------------------------------------------------------------------
    # FEEDBACK (no action, just logging)
    # -------------------------------------------------------------------------
    
    CanonicalIntent.USER_BEHAVIOR_FEEDBACK: IntentDefinition(
        intent=CanonicalIntent.USER_BEHAVIOR_FEEDBACK,
        trigger_phrases=[
            "Astra, feedback:",
            "astra, feedback:",
            "ASTRA, feedback:",
        ],
        trigger_patterns=[
            r"^[Aa]stra,?\s*feedback:\s*",
        ],
        requires_context=[],
        requires_confirmation=False,
        description="User feedback for behavior tuning",
        behavior=(
            "Log structured feedback event.\n"
            "Behavior tuning occurs ONLY in sandbox.\n"
            "NEVER triggers any commands."
        ),
    ),
    
    # -------------------------------------------------------------------------
    # RAG CODEBASE QUERY (v1.3)
    # -------------------------------------------------------------------------
    
    CanonicalIntent.RAG_CODEBASE_QUERY: IntentDefinition(
        intent=CanonicalIntent.RAG_CODEBASE_QUERY,
        trigger_phrases=[
            # Explicit codebase search
            "search codebase:",
            "Search codebase:",
            "codebase search:",
            "Codebase search:",
            "ask about codebase:",
            "Ask about codebase:",
            # Question prefixes
            "In the codebase,",
            "in the codebase,",
            "In this codebase,",
            "in this codebase,",
            # Index commands
            "index the architecture",
            "Index the architecture",
            "index architecture",
            "Index architecture",
            "run RAG index",
            "Run RAG index",
        ],
        trigger_patterns=[
            r"^(?:[Ss]earch|[Aa]sk about)\s+(?:the\s+)?codebase:\s*(.+)$",
            r"^[Cc]odebase\s+(?:search|query):\s*(.+)$",
            r"^[Ii]n (?:the|this) codebase,?\s+(.+)$",
            r"^[Ww]hat (?:functions?|classes?|methods?)\s+.+\?$",
            r"^[Ww]here is\s+.+\s+(?:implemented|defined|located)\?$",
            r"^[Hh]ow does\s+(?:the\s+)?.+\s+(?:work|function)\?$",
            r"^[Ss]how me\s+(?:the\s+)?.+\s+(?:code|implementation|function)$",
            r"^[Ii]ndex (?:the )?(?:architecture|codebase|RAG)$",
            r"^[Rr]un RAG index$",
        ],
        requires_context=[],
        requires_confirmation=False,
        description="Search codebase using RAG and answer questions",
        behavior=(
            "RAG-powered codebase Q&A:\n"
            "1. Parse the question from the message\n"
            "2. Search indexed code signatures for relevant chunks\n"
            "3. Build context from top matches\n"
            "4. Answer using LLM with codebase context\n"
            "\n"
            "Trigger phrases:\n"
            "- 'search codebase: <question>'\n"
            "- 'In the codebase, <question>'\n"
            "- 'What functions handle X?'\n"
            "- 'Where is Y implemented?'\n"
            "\n"
            "Index commands:\n"
            "- 'index the architecture' - triggers /rag/index\n"
            "\n"
            "Prerequisites: Run 'CREATE ARCHITECTURE MAP' first."
        ),
    ),
    
    # -------------------------------------------------------------------------
    # EMBEDDING MANAGEMENT (v1.3)
    # -------------------------------------------------------------------------
    
    CanonicalIntent.EMBEDDING_STATUS: IntentDefinition(
        intent=CanonicalIntent.EMBEDDING_STATUS,
        trigger_phrases=[
            "embedding status",
            "Embedding status",
            "embeddings status",
            "check embeddings",
            "Astra, command: embedding status",
            "astra, command: embedding status",
        ],
        trigger_patterns=[
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?[Ee]mbedding[s]?\s+status$",
            r"^[Cc]heck\s+embedding[s]?$",
            r"^[Ee]mbedding[s]?\s+progress$",
            r"^[Hh]ow\s+are\s+embeddings\s+doing$",
        ],
        requires_context=[],
        requires_confirmation=False,
        description="Check current embedding generation status",
        behavior=(
            "Display embedding status:\n"
            "- Total chunks in DB\n"
            "- Embedded vs pending count\n"
            "- Current tier being processed\n"
            "- Last run timestamp\n"
            "- Any errors"
        ),
    ),
    
    CanonicalIntent.GENERATE_EMBEDDINGS: IntentDefinition(
        intent=CanonicalIntent.GENERATE_EMBEDDINGS,
        trigger_phrases=[
            "generate embeddings",
            "Generate embeddings",
            "run embeddings",
            "start embeddings",
            "embed chunks",
            "Astra, command: generate embeddings",
            "astra, command: generate embeddings",
        ],
        trigger_patterns=[
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?[Gg]enerate\s+embedding[s]?$",
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?[Rr]un\s+embedding[s]?$",
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?[Ss]tart\s+embedding[s]?$",
            r"^[Ee]mbed\s+(?:the\s+)?(?:code\s+)?chunks$",
        ],
        requires_context=[],
        requires_confirmation=False,
        description="Trigger manual embedding generation",
        behavior=(
            "Queue background embedding job:\n"
            "- Uses priority ordering (Tier1 first)\n"
            "- Incremental (skips already-embedded chunks)\n"
            "- Non-blocking (returns immediately)\n"
            "- Use 'embedding status' to check progress"
        ),
    ),
    
    # -------------------------------------------------------------------------
    # FILESYSTEM QUERY (v1.4)
    # -------------------------------------------------------------------------
    
    CanonicalIntent.FILESYSTEM_QUERY: IntentDefinition(
        intent=CanonicalIntent.FILESYSTEM_QUERY,
        trigger_phrases=[
            # Patterns are used instead - see tier0_rules.py check_filesystem_query_trigger()
        ],
        trigger_patterns=[
            # Detection handled by check_filesystem_query_trigger() in tier0_rules.py
            # for more complex prefix/path validation
        ],
        requires_context=[],
        requires_confirmation=False,
        description="Query scan index for filesystem listing or search",
        behavior=(
            "Query the architecture_file_index DB table (from scan sandbox) to:\n"
            "- List contents of a directory (folders first, then files)\n"
            "- Find files/folders by name pattern\n"
            "- Return full paths\n"
            "\n"
            "Constraints:\n"
            "- Only allowed roots: D:\\ and C:\\Users\\dizzi\n"
            "- NEVER run shell commands or mention running dir/grep\n"
            "- Hard cap 200 entries (show '+N more' if truncated)\n"
            "- Uses DB index first, fallback to sandbox /fs/tree if needed"
        ),
    ),
    
    # -------------------------------------------------------------------------
    # CODEBASE REPORT (v1.5)
    # -------------------------------------------------------------------------
    
    CanonicalIntent.CODEBASE_REPORT: IntentDefinition(
        intent=CanonicalIntent.CODEBASE_REPORT,
        trigger_phrases=[
            "codebase report fast",
            "Codebase report fast",
            "CODEBASE REPORT FAST",
            "codebase report full",
            "Codebase report full",
            "CODEBASE REPORT FULL",
            "Astra, command: codebase report fast",
            "astra, command: codebase report fast",
            "Astra, command: codebase report full",
            "astra, command: codebase report full",
        ],
        trigger_patterns=[
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?[Cc]odebase\s+[Rr]eport\s+[Ff]ast$",
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?[Cc]odebase\s+[Rr]eport\s+[Ff]ull$",
        ],
        requires_context=[],
        requires_confirmation=False,
        description="Generate codebase hygiene/bloat/drift report",
        behavior=(
            "Generate a read-only codebase report scanning D:\\Orb and D:\\orb-desktop.\n"
            "\n"
            "FAST mode:\n"
            "- Metadata scan (size + mtime)\n"
            "- Line counting for text files\n"
            "- Bloat offenders (largest/longest files)\n"
            "- Floating files and suspect folders\n"
            "- Incremental changes since last report\n"
            "\n"
            "FULL mode:\n"
            "- All FAST mode features\n"
            "- Absolute path detection in content\n"
            "- Blocked folder reference detection\n"
            "- Duplicate filename heuristics\n"
            "\n"
            "Output: D:\\Orb.architecture\\CODEBASE_REPORT_<MODE>_<TIMESTAMP>.md/json\n"
            "\n"
            "Does NOT trigger embeddings, scans, indexing, or schema changes."
        ),
    ),
    
    # -------------------------------------------------------------------------
    # LATEST REPORT RESOLVER (v1.3)
    # -------------------------------------------------------------------------
    
    CanonicalIntent.LATEST_ARCHITECTURE_MAP: IntentDefinition(
        intent=CanonicalIntent.LATEST_ARCHITECTURE_MAP,
        trigger_phrases=[
            "latest architecture map",
            "Latest architecture map",
            "Astra, command: latest architecture map",
            "astra, command: latest architecture map",
            "Orb, command: latest architecture map",
            "orb, command: latest architecture map",
        ],
        trigger_patterns=[
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?[Ll]atest\s+[Aa]rchitecture\s+[Mm]ap$",
            r"^(?:[Oo]rb[,:]?\s*)?(?:command[:\s]+)?[Ll]atest\s+[Aa]rchitecture\s+[Mm]ap$",
        ],
        requires_context=[],
        requires_confirmation=False,
        description="Resolve and display the latest architecture map from D:\\Orb\\.architecture\\",
        behavior=(
            "Resolve the latest ARCHITECTURE_MAP*.md file by mtime.\n"
            "\n"
            "Location: D:\\Orb\\.architecture\\\n"
            "Patterns: ARCHITECTURE_MAP.md, ARCHITECTURE_MAP_*.md\n"
            "\n"
            "Returns:\n"
            "- File path and metadata (mtime, size)\n"
            "- Content preview (first 100 lines)\n"
            "\n"
            "Read-only operation. Never hardcodes timestamped filenames."
        ),
    ),
    
    CanonicalIntent.LATEST_CODEBASE_REPORT_FULL: IntentDefinition(
        intent=CanonicalIntent.LATEST_CODEBASE_REPORT_FULL,
        trigger_phrases=[
            "latest codebase report full",
            "Latest codebase report full",
            "Astra, command: latest codebase report full",
            "astra, command: latest codebase report full",
            "Orb, command: latest codebase report full",
            "orb, command: latest codebase report full",
            # Short forms
            "latest codebase report",
            "Latest codebase report",
        ],
        trigger_patterns=[
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?[Ll]atest\s+[Cc]odebase\s+[Rr]eport(?:\s+[Ff]ull)?$",
            r"^(?:[Oo]rb[,:]?\s*)?(?:command[:\s]+)?[Ll]atest\s+[Cc]odebase\s+[Rr]eport(?:\s+[Ff]ull)?$",
        ],
        requires_context=[],
        requires_confirmation=False,
        description="Resolve and display the latest FULL codebase report from D:\\Orb\\.architecture\\",
        behavior=(
            "Resolve the latest CODEBASE_REPORT_FULL_*.md file by mtime.\n"
            "\n"
            "Location: D:\\Orb\\.architecture\\\n"
            "Patterns: CODEBASE_REPORT_FULL.md, CODEBASE_REPORT_FULL_*.md\n"
            "\n"
            "IMPORTANT: Only matches FULL reports (not FAST).\n"
            "MD-first: ignores JSON files.\n"
            "\n"
            "Returns:\n"
            "- File path and metadata (mtime, size)\n"
            "- Content preview (first 100 lines)\n"
            "\n"
            "Read-only operation. Never hardcodes timestamped filenames."
        ),
    ),
    
    # -------------------------------------------------------------------------
    # MULTI-FILE OPERATIONS (v1.4 - Level 3)
    # -------------------------------------------------------------------------
    
    CanonicalIntent.MULTI_FILE_SEARCH: IntentDefinition(
        intent=CanonicalIntent.MULTI_FILE_SEARCH,
        trigger_phrases=[],  # Patterns handled by check_multi_file_trigger in tier0_rules.py
        trigger_patterns=[],
        requires_context=[],
        requires_confirmation=False,
        description="Search across multiple files for patterns or content",
        behavior=(
            "Read-only multi-file search operation.\n"
            "\n"
            "Triggers:\n"
            "- 'find all X in the codebase'\n"
            "- 'list files containing X'\n"
            "- 'search codebase for X'\n"
            "\n"
            "Returns:\n"
            "- List of matching files\n"
            "- Preview of matches\n"
            "- Total count of occurrences\n"
            "\n"
            "Does NOT modify any files."
        ),
    ),
    
    CanonicalIntent.MULTI_FILE_REFACTOR: IntentDefinition(
        intent=CanonicalIntent.MULTI_FILE_REFACTOR,
        trigger_phrases=[],  # Patterns handled by check_multi_file_trigger in tier0_rules.py
        trigger_patterns=[],
        requires_context=[],
        requires_confirmation=True,  # Write operation requires confirmation
        confirmation_prompt=(
            "⚠️ MULTI-FILE REFACTOR\n"
            "You are about to replace '{search_pattern}' with '{replacement_pattern}' "
            "in {total_files} files.\n"
            "\n"
            "Type 'Yes' to confirm."
        ),
        description="Search and replace across multiple files",
        behavior=(
            "Multi-file search and replace operation.\n"
            "\n"
            "Triggers:\n"
            "- 'replace X with Y everywhere'\n"
            "- 'change all X to Y'\n"
            "- 'rename X to Y across the codebase'\n"
            "\n"
            "REQUIRES CONFIRMATION before execution.\n"
            "\n"
            "Process:\n"
            "1. Discovery phase finds all matching files\n"
            "2. Shows preview and asks for confirmation\n"
            "3. On confirmation, processes each file\n"
            "4. Reports results with success/failure counts"
        ),
    ),
    
    # -------------------------------------------------------------------------
    # SEGMENT LOOP (v1.8 - Phase 2 Pipeline Segmentation)
    # -------------------------------------------------------------------------
    
    CanonicalIntent.RUN_SEGMENT_LOOP: IntentDefinition(
        intent=CanonicalIntent.RUN_SEGMENT_LOOP,
        trigger_phrases=[
            "run segments",
            "Run segments",
            "run segment loop",
            "Run segment loop",
            "execute segments",
            "Execute segments",
        ],
        trigger_patterns=[
            r"^[Rr]un\s+(?:the\s+)?segments?$",
            r"^[Ee]xecute\s+(?:the\s+)?segments?$",
            r"^[Rr]un\s+segment\s+loop$",
            r"^[Ss]egment\s+loop$",
            r"^[Rr]un\s+segmented\s+job$",
        ],
        requires_context=[],
        requires_confirmation=False,
        description="Execute segmented job through the pipeline segment by segment",
        behavior=(
            "Execute a segmented job through the pipeline:\n"
            "1. Load segments from validated spec\n"
            "2. Process each segment through critical pipeline\n"
            "3. Track progress and report results\n"
            "\n"
            "Requires validated spec with segments."
        ),
    ),
    
    # -------------------------------------------------------------------------
    # v5.4 PHASE 1B: UNIFIED PIPELINE COMMAND
    # -------------------------------------------------------------------------
    # Merges RUN_CRITICAL_PIPELINE_FOR_JOB and RUN_SEGMENT_LOOP.
    # Always runs through the segment loop (which handles both single and
    # multi-segment manifests). Trigger phrases cover both old commands.
    
    CanonicalIntent.RUN_PIPELINE: IntentDefinition(
        intent=CanonicalIntent.RUN_PIPELINE,
        trigger_phrases=[
            "Run the pipeline",
            "run the pipeline",
            "Run pipeline",
            "run pipeline",
            "Run critical pipeline",
            "run critical pipeline",
            "Execute pipeline",
            "execute pipeline",
            "Start the pipeline",
            "start the pipeline",
            "Run segments",
            "run segments",
            "Execute segments",
            "execute segments",
        ],
        trigger_patterns=[
            r"^[Rr]un (?:the )?pipeline$",
            r"^[Rr]un (?:the )?[Cc]ritical [Pp]ipeline$",
            r"^[Rr]un (?:the )?[Cc]ritical [Pp]ipeline for job\s+",
            r"^[Ee]xecute (?:the )?pipeline$",
            r"^[Ee]xecute (?:the )?[Cc]ritical [Pp]ipeline$",
            r"^[Ss]tart the pipeline$",
            r"^[Rr]un\s+(?:the\s+)?segments?$",
            r"^[Ee]xecute\s+(?:the\s+)?segments?$",
            r"^[Rr]un\s+segment\s+loop$",
            r"^[Rr]un\s+segmented\s+job$",
        ],
        requires_context=["job_id", "spec_id"],
        requires_confirmation=True,
        confirmation_prompt=(
            "⚠️ HIGH-STAKES OPERATION\n"
            "You are about to run the pipeline for the latest validated spec.\n"
            "This will generate architecture and write files to your project.\n\n"
            "Type 'confirm' or 'yes' to proceed."
        ),
        description="Run the unified pipeline (handles both single and multi-segment jobs)",
        behavior=(
            "Execute the validated spec through the pipeline:\n"
            "1. Load the segment manifest (always present after SpecGate)\n"
            "2. Process each segment through critical pipeline → critique → overwatcher\n"
            "3. For single-segment jobs, runs the loop once (no extra overhead)\n"
            "4. For multi-segment jobs, processes in dependency order with evidence threading\n"
            "\n"
            "Requires validated spec."
        ),
    ),
    
    # -------------------------------------------------------------------------
    # v5.13: IMPLEMENT SEGMENTS — Phase 2 (execution only, no architecture)
    # -------------------------------------------------------------------------
    # Separated from RUN_PIPELINE to prevent accidental auto-execution.
    # RUN_PIPELINE (= "run segments") generates architectures and stops.
    # IMPLEMENT_SEGMENTS (= "implement segments") executes approved architectures.
    
    CanonicalIntent.IMPLEMENT_SEGMENTS: IntentDefinition(
        intent=CanonicalIntent.IMPLEMENT_SEGMENTS,
        trigger_phrases=[
            "Implement segments",
            "implement segments",
            "Implement the segments",
            "implement the segments",
            "Execute implementations",
            "execute implementations",
            "Run implementations",
            "run implementations",
        ],
        trigger_patterns=[
            r"^[Ii]mplement\s+(?:the\s+)?segments?$",
            r"^[Ee]xecute\s+(?:the\s+)?implementations?$",
            r"^[Rr]un\s+(?:the\s+)?implementations?$",
            r"^[Ii]mplement\s+(?:the\s+)?(?:approved\s+)?(?:architecture|arch)s?$",
        ],
        requires_context=[],
        requires_confirmation=False,
        description="Implement approved segments through Overwatcher + Implementer",
        behavior=(
            "Execute APPROVED segments through implementation:\n"
            "1. Load the segment manifest and state\n"
            "2. Skip PENDING segments (architecture not yet generated)\n"
            "3. Execute only APPROVED segments through Overwatcher + Implementer\n"
            "4. Write files to the project\n"
            "\n"
            "Requires segments to be APPROVED first (via 'run segments')."
        ),
    ),
    
    # -------------------------------------------------------------------------
    # CHAT (no action)
    # -------------------------------------------------------------------------
    
    CanonicalIntent.CHAT_ONLY: IntentDefinition(
        intent=CanonicalIntent.CHAT_ONLY,
        trigger_phrases=[],  # Default fallback
        trigger_patterns=[],
        requires_context=[],
        requires_confirmation=False,
        description="Normal conversation - no backend actions",
        behavior=(
            "Chat mode - NO backend actions.\n"
            "Used for:\n"
            "- Normal conversation\n"
            "- Thinking\n"
            "- Explanations\n"
            "- Planning\n"
            "- Storytelling\n"
            "- Meta-discussion"
        ),
    ),
}


def get_intent_definition(intent: CanonicalIntent) -> IntentDefinition:
    """Get the definition for a canonical intent."""
    return INTENT_DEFINITIONS[intent]


def get_all_command_intents() -> List[CanonicalIntent]:
    """Get all intents that are actual commands (not chat or feedback)."""
    return [
        intent for intent in CanonicalIntent 
        if intent not in (CanonicalIntent.CHAT_ONLY, CanonicalIntent.USER_BEHAVIOR_FEEDBACK)
    ]


def get_high_stakes_intents() -> List[CanonicalIntent]:
    """Get all intents that require confirmation."""
    return [
        intent for intent, defn in INTENT_DEFINITIONS.items()
        if defn.requires_confirmation
    ]


def get_spec_gate_flow_intents() -> List[CanonicalIntent]:
    """Get intents related to the Spec Gate flow."""
    return [
        CanonicalIntent.WEAVER_BUILD_SPEC,
        CanonicalIntent.SEND_TO_SPEC_GATE,
        CanonicalIntent.RUN_PIPELINE,  # v5.4: unified
        CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB,  # v5.4: deprecated alias
    ]


def get_overwatcher_intents() -> List[CanonicalIntent]:
    """Get intents related to Overwatcher."""
    return [
        CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES,
    ]


def get_intent_by_trigger_phrase(phrase: str) -> Optional[CanonicalIntent]:
    """
    Exact match lookup for trigger phrases.
    Returns None if no exact match found.
    """
    for intent, defn in INTENT_DEFINITIONS.items():
        if phrase in defn.trigger_phrases:
            return intent
    return None
