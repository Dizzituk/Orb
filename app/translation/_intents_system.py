# FILE: app/translation/_intents_system.py
"""
System & pipeline intent definitions.
Architecture, Sandbox, Spec Gate, Pipeline, Overwatcher, Segments, Refactor.
"""
from __future__ import annotations
from typing import Dict
from .schemas import CanonicalIntent, IntentDefinition


SYSTEM_INTENTS: Dict[CanonicalIntent, IntentDefinition] = {

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
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?CREATE ARCHITECTURE MAP$",
            r"^CREATE ARCHITECTURE MAP\s+for\s+",
        ],
        requires_context=[],
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
            "Create architecture map",
            "create architecture map",
            "Astra, command: Create architecture map",
            "astra, command: create architecture map",
        ],
        trigger_patterns=[
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?[Cc]reate [Aa]rchitecture [Mm]ap$",
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
        requires_context=[],
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
            "scan sandbox",
            "Scan sandbox",
            "SCAN SANDBOX",
            "Astra, command: scan sandbox",
            "astra, command: scan sandbox",
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
            "How does that look all together",
            "How does that look all together?",
            "how does that look all together",
            "how does that look all together?",
            "Weave this into a spec",
            "weave this into a spec",
            "Build spec from ramble",
            "build spec from ramble",
            "Compile the spec",
            "compile the spec",
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
        requires_context=[],
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
            "Send that to Spec Gate",
            "send that to Spec Gate",
            "Send to Spec Gate",
            "send to Spec Gate",
            "Okay, send that to Spec Gate",
            "okay, send that to Spec Gate",
            "Ok, send that to Spec Gate",
            "ok, send that to Spec Gate",
            "Validate the spec",
            "validate the spec",
            "Run Spec Gate",
            "run Spec Gate",
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
        requires_context=[],
        requires_confirmation=False,
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
        requires_context=["job_id", "spec_id"],
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

    CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES: IntentDefinition(
        intent=CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES,
        trigger_phrases=[
            "run overwatcher",
            "Run overwatcher",
            "Run Overwatcher",
            "execute overwatcher",
            "Execute overwatcher",
            "Execute Overwatcher",
            "start overwatcher",
            "Start overwatcher",
            "Start Overwatcher",
            "Astra, command: run overwatcher",
            "astra, command: run overwatcher",
            "Astra command run overwatcher",
            "Execute overwatcher changes",
            "Apply overwatcher changes",
            "send to overwatcher",
            "Send to overwatcher",
            "Send to Overwatcher",
        ],
        trigger_patterns=[
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?[Rr]un\s+[Oo]verwatcher$",
            r"^(?:[Aa]stra[,:]?\s*)?[Ee]xecute\s+[Oo]verwatcher$",
            r"^(?:[Aa]stra[,:]?\s*)?[Ss]tart\s+[Oo]verwatcher$",
            r"^[Oo]verwatcher\s+run$",
            r"^[Rr]un\s+the\s+[Oo]verwatcher$",
            r"^[Ii]nvoke\s+[Oo]verwatcher$",
            r"^[Tt]rigger\s+[Oo]verwatcher$",
            r"^[Ss]end\s+to\s+[Oo]verwatcher$",
            r"^[Ee]xecute [Oo]verwatcher [Cc]hanges$",
            r"^[Aa]pply [Oo]verwatcher [Cc]hanges$",
            r"^[Oo]verwatcher[,:]\s*execute$",
        ],
        requires_context=[],
        requires_confirmation=False,
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
    # v5.13: IMPLEMENT SEGMENTS — Phase 2
    # -------------------------------------------------------------------------

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
    # CODEBASE REFACTOR (v1.9 - Self-refactoring loop)
    # -------------------------------------------------------------------------

    CanonicalIntent.REFACTOR_CODEBASE: IntentDefinition(
        intent=CanonicalIntent.REFACTOR_CODEBASE,
        trigger_phrases=[
            "Astra, refactor yourself",
            "astra, refactor yourself",
            "ASTRA, REFACTOR YOURSELF",
            "Refactor yourself",
            "refactor yourself",
            "Astra, command: refactor codebase",
            "astra, command: refactor codebase",
            "refactor codebase",
            "Refactor codebase",
            "Start refactor",
            "start refactor",
            "*refactor",
        ],
        trigger_patterns=[
            r"^(?:[Aa]stra[,:]?\s+)?[Rr]efactor\s+(?:yourself|codebase)$",
            r"^[Ss]tart\s+(?:the\s+)?refactor(?:ing)?$",
            r"^\*refactor$",
        ],
        requires_context=[],
        requires_confirmation=True,
        confirmation_prompt=(
            "This will start the refactor loop — scanning your codebase, "
            "extracting symbols from oversized files, and boot-checking after "
            "each pass. The loop runs until all files are at minimum viable size "
            "or a boot check fails. Proceed?"
        ),
        description="Run the scan-do-rescan codebase refactor loop",
        behavior=(
            "Runs the autonomous refactor loop:\n"
            "1. Scan codebase, pick easiest oversized file\n"
            "2. Surgical extraction (zero LLM, deterministic)\n"
            "3. Boot check — if fail, rollback and stop\n"
            "4. Update RAG with new file state\n"
            "5. Rescan and repeat\n\n"
            "Uses surgical_extractor for all extractions.\n"
            "Chips away at monoliths across multiple passes.\n"
            "Stops when scan returns zero oversized files."
        ),
    ),
}
