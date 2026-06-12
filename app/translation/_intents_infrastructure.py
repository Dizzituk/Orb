# FILE: app/translation/_intents_infrastructure.py
# Purpose: Intent definitions: Infrastructure domain.
# Called-by: app.translation.intents
# Depends-on: app.translation.schemas
# Last-renovated: 2026-06-11
"""
Intent definitions: Infrastructure domain.
Architecture, sandbox, spec gate, high-stakes pipeline, overwatcher, feedback.
"""
from __future__ import annotations
from typing import Dict
from .schemas import CanonicalIntent, IntentDefinition


INFRASTRUCTURE_INTENTS: Dict[CanonicalIntent, IntentDefinition] = {

    # -------------------------------------------------------------------------
    # ARCHITECTURE COMMANDS
    # -------------------------------------------------------------------------

    CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES: IntentDefinition(
        intent=CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES,
        trigger_phrases=[
            "CREATE ARCHITECTURE MAP",
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
            "send that to weaver",
            "Send that to weaver",
            "send to weaver",
            "Send to weaver",
            "send that to the weaver",
            "Send that to the weaver",
            "send to the weaver",
            "Send to the weaver",
            "sent a weaver",
            "Sent a weaver",
            "sent to weaver",
            "Sent to weaver",
            "sent to the weaver",
            "Sent to the weaver",
            "send out to weaver",
            "Send out to weaver",
            "send out to the weaver",
            "Send out to the weaver",
            "weave that together",
            "Weave that together",
        ],
        trigger_patterns=[
            r"^[Hh]ow does that look all together\??$",
            r"^[Ww]eave (?:this|that) (?:into a spec|together)$",
            r"^[Bb]uild (?:a )?spec from (?:the )?ramble$",
            r"^[Cc]ompile (?:the )?spec$",
            r"^[Pp]ut (?:that|this|it) all together$",
            r"^[Cc]onsolidate (?:that|this) into a spec$",
            r"^[Ss]ummarize (?:the|my) ramble into a spec$",
            r"^[Tt]urn (?:this|that) into a spec$",
            r"^(?:[Aa]stra[,:]?\s*)?[Ss]en[dt] (?:that|this|it|out|a) (?:to )?(?:the )?[Ww]eaver\.?$",
            r"^(?:[Aa]stra[,:]?\s*)?[Ss]en[dt] (?:out )?to (?:the )?[Ww]eaver\.?$",
            r"^(?:[Hh]ow does that (?:look|come) (?:all together|along)\??$)",
        ],
        requires_context=[],
        requires_confirmation=True,
        confirmation_prompt=(
            "I'll send our conversation to the Weaver to build a structured spec.\n"
            "Confirm to proceed."
        ),
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
            "sent a spec gate",
            "Sent a spec gate",
            "sent to spec gate",
            "Sent to spec gate",
            "sent that spec gate",
            "Sent that spec gate",
            "send a spec gate",
            "Send a spec gate",
            "Validate the spec",
            "validate the spec",
            "Run Spec Gate",
            "run Spec Gate",
            "Submit spec for validation",
            "submit spec for validation",
        ],
        trigger_patterns=[
            r"^(?:[Oo]k(?:ay)?,?\s*)?[Ss]en[dt] (?:that|this|it|a) (?:to )?[Ss]pec ?[Gg]ate\.?$",
            r"^[Ss]en[dt] (?:a |to )?(?:the )?[Ss]pec ?[Gg]ate\.?$",
            r"^[Vv]alidate (?:the )?spec$",
            r"^[Rr]un [Ss]pec ?[Gg]ate$",
            r"^[Ss]ubmit (?:the )?spec(?: for validation)?$",
            r"^[Ss]pec ?[Gg]ate[,:]?\s*validate$",
        ],
        requires_context=[],
        requires_confirmation=True,
        confirmation_prompt=(
            "I'll send the spec to Spec Gate for validation.\n"
            "Confirm to proceed."
        ),
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

    # =========================================================================
    # PROJECT REGISTRY (v1.7)
    # =========================================================================

    CanonicalIntent.SCAN_PROJECT_ARCHITECTURE: IntentDefinition(
        intent=CanonicalIntent.SCAN_PROJECT_ARCHITECTURE,
        trigger_phrases=[
            "scan project architecture",
            "Astra, scan the architecture",
            "astra, scan the architecture",
            "Astra, refresh the architecture",
            "astra, refresh the architecture",
            "Astra, rescan the project",
            "astra, rescan the project",
            "Astra, update the architecture",
            "scan project",
            "rescan project",
        ],
        trigger_patterns=[
            r"(?:scan|refresh|rescan|update)\s+(?:the\s+)?(?:project\s+)?architecture",
            r"(?:scan|rescan)\s+(?:the\s+)?project",
        ],
        requires_confirmation=True,
        description="Scan a registered project's file structure and save to architecture DB",
        behavior=(
            "1. Identify which project from context or message\n"
            "2. Scan the project's root paths via sandbox controller\n"
            "3. Save file index to architecture DB (scoped by project)\n"
            "4. Generate ARCHITECTURE_MAP.md in project folder\n"
            "5. Update project registry scan metadata"
        ),
    ),

    CanonicalIntent.REGISTER_PROJECT: IntentDefinition(
        intent=CanonicalIntent.REGISTER_PROJECT,
        trigger_phrases=[
            "register project",
            "Astra, register project",
            "astra, register project",
            "add project",
            "Astra, add project",
        ],
        trigger_patterns=[
            r"(?:register|add)\s+(?:a\s+)?(?:new\s+)?project",
        ],
        requires_confirmation=True,
        description="Register a new external project for architecture tracking",
        behavior=(
            "Registers a new project with name, root paths, type, and language.\n"
            "The project will then be scannable and queryable."
        ),
    ),

    # -------------------------------------------------------------------------
    # PROJECT SCOPING (v2.2 - Conversational project creation)
    # -------------------------------------------------------------------------

    CanonicalIntent.PROJECT_SCOPE_START: IntentDefinition(
        intent=CanonicalIntent.PROJECT_SCOPE_START,
        trigger_phrases=[
            "start a new project",
            "new project",
            "I want to build something new",
            "let's start a new app",
            "new app",
            "new android app",
            "I've got a new idea",
            "i want to start a new project",
            "let's start a new project",
            "i want to build a new app",
            "Astra, new project",
            "astra, new project",
            "Astra, let's start a new project",
            "astra, let's start a new project",
            "I want to start off a new project",
            "i want to start off a new project",
            "Hey, Astra, I want to start a new project",
            "Hey Astra, I want to start a new project",
            "I want to build a phone app",
            "i want to build a phone app",
            "I want to build an app",
            "i want to build an app",
            "I want to build an android app",
            "i want to build an android app",
            "I want to make an app",
            "build me an app",
            "build a phone app",
            "build an android app",
            "Astra, I want to build a phone app",
            "astra, I want to build a phone app",
        ],
        trigger_patterns=[
            r"^(?:[Aa]stra[,:]?\s*)?(?:I want to |let's |let me )?(?:start|begin|create)(?:\s+off)?\s+a new (?:project|app|build)",
            r"^(?:[Aa]stra[,:]?\s*)?[Nn]ew (?:project|app|android app|build)$",
            r"^(?:[Aa]stra[,:]?\s*)?[Ii]'ve got (?:a |an )?new (?:idea|project|app)",
            r"^(?:[Aa]stra[,:]?\s*)?[Ii] want to build (?:a |an )?new",
            r"^(?:[Aa]stra[,:]?\s*)?[Ll]et's build (?:a |an )?new",
            r"^(?:[Aa]stra[,:]?\s*)?[Ii] want to build (?:a |an )?(?:phone|android|web|mobile|desktop) ?app",
            r"^(?:[Aa]stra[,:]?\s*)?[Ii] want to (?:make|create|build) (?:a |an )?(?:new )?app",
            r"^(?:[Aa]stra[,:]?\s*)?[Ll]et's (?:make|create|build) (?:a |an )?(?:phone|android|web|mobile) ?app",
            r"^(?:[Aa]stra[,:]?\s*)?(?:build|make|create) (?:me )?(?:a |an )?(?:phone|android|web|mobile|new) ?app",
        ],
        requires_context=[],
        requires_confirmation=True,
        confirmation_prompt=(
            "You want to start scoping a new project.\n"
            "This will enter a conversational flow where I'll ask you about:\n"
            "- Project type (Android app, web app, ASTRA feature)\n"
            "- Project name\n"
            "- What it does\n\n"
            "Ready to begin?"
        ),
        description="Enter multi-turn project scoping conversation",
        behavior=(
            "Enter project scoping mode. Escalate to reasoning model (GPT-5.4).\n"
            "Ask the user ONE question at a time:\n"
            "1. What kind of project? (Android app, web app, ASTRA feature)\n"
            "2. What should we call it?\n"
            "3. Brief description of what it does\n"
            "4. Any specific tech requirements?\n\n"
            "Once enough info gathered, summarise and ask user to confirm.\n"
            "On confirmation: create project folder, register BuildTargetProfile,\n"
            "and store in ProjectSession for pipeline routing.\n\n"
            "User triggers Weaver with phrases like:\n"
            "'send that to weaver', 'how does that look all together',\n"
            "'weave that together', etc."
        ),
    ),

    CanonicalIntent.LIST_REGISTERED_PROJECTS: IntentDefinition(
        intent=CanonicalIntent.LIST_REGISTERED_PROJECTS,
        trigger_phrases=[
            "list projects",
            "show projects",
            "Astra, list projects",
            "astra, list projects",
            "what projects are registered",
            "Astra, what projects do you know about",
        ],
        trigger_patterns=[
            r"(?:list|show|what)\s+(?:registered\s+)?projects",
        ],
        requires_confirmation=False,
        description="List all registered external projects",
        behavior="Display all active registered projects with scan status.",
    ),
}
