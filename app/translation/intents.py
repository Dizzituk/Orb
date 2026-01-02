# FILE: app/translation/intents.py
"""
Canonical intent definitions for ASTRA Translation Layer.
This is the CLOSED SET of allowed intents. No free-form execution.

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
        ],
        trigger_patterns=[
            r"^CREATE ARCHITECTURE MAP$",  # Exact match, all caps
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
        ],
        trigger_patterns=[
            r"^[Cc]reate [Aa]rchitecture [Mm]ap$",  # Case-insensitive but NOT all caps
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
        ],
        trigger_patterns=[
            r"^[Uu]pdate [Aa]rchitecture$",
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
        CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB,
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
