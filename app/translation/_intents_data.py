# FILE: app/translation/_intents_data.py
"""
Data & query intent definitions.
Feedback, RAG, Embedding, Filesystem, Codebase Report, Multi-file, Quarantine.
"""
from __future__ import annotations
from typing import Dict
from .schemas import CanonicalIntent, IntentDefinition


DATA_INTENTS: Dict[CanonicalIntent, IntentDefinition] = {

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
            "search codebase:",
            "Search codebase:",
            "codebase search:",
            "Codebase search:",
            "ask about codebase:",
            "Ask about codebase:",
            "In the codebase,",
            "in the codebase,",
            "In this codebase,",
            "in this codebase,",
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
        trigger_phrases=[],
        trigger_patterns=[],
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
            "FAST mode: Metadata scan, line counting, bloat offenders.\n"
            "FULL mode: All FAST + content scanning, duplicate detection.\n"
            "\n"
            "Output: D:\\Orb\\.architecture\\CODEBASE_REPORT_<MODE>_<TIMESTAMP>.md/json"
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
            "Location: D:\\Orb\\.architecture\\\n"
            "Returns file path, metadata, and content preview.\n"
            "Read-only operation."
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
            "latest codebase report",
            "Latest codebase report",
        ],
        trigger_patterns=[
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?[Ll]atest\s+[Cc]odebase\s+[Rr]eport(?:\s+[Ff]ull)?$",
            r"^(?:[Oo]rb[,:]?\s*)?(?:command[:\s]+)?[Ll]atest\s+[Cc]odebase\s+[Rr]eport(?:\s+[Ff]ull)?$",
        ],
        requires_context=[],
        requires_confirmation=False,
        description="Resolve and display the latest FULL codebase report",
        behavior=(
            "Resolve the latest CODEBASE_REPORT_FULL_*.md file by mtime.\n"
            "Only matches FULL reports (not FAST). MD-first.\n"
            "Read-only operation."
        ),
    ),

    # -------------------------------------------------------------------------
    # MULTI-FILE OPERATIONS (v1.4 - Level 3)
    # -------------------------------------------------------------------------

    CanonicalIntent.MULTI_FILE_SEARCH: IntentDefinition(
        intent=CanonicalIntent.MULTI_FILE_SEARCH,
        trigger_phrases=[],
        trigger_patterns=[],
        requires_context=[],
        requires_confirmation=False,
        description="Search across multiple files for patterns or content",
        behavior=(
            "Read-only multi-file search operation.\n"
            "Does NOT modify any files."
        ),
    ),

    CanonicalIntent.MULTI_FILE_REFACTOR: IntentDefinition(
        intent=CanonicalIntent.MULTI_FILE_REFACTOR,
        trigger_phrases=[],
        trigger_patterns=[],
        requires_context=[],
        requires_confirmation=True,
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
            "REQUIRES CONFIRMATION before execution."
        ),
    ),

    # -------------------------------------------------------------------------
    # RAG LIFECYCLE (v2.0 - Quarantine purge & status)
    # -------------------------------------------------------------------------

    CanonicalIntent.PURGE_QUARANTINE: IntentDefinition(
        intent=CanonicalIntent.PURGE_QUARANTINE,
        trigger_phrases=[
            "Astra, command: purge quarantine",
            "astra, command: purge quarantine",
            "Astra, purge quarantine",
            "astra, purge quarantine",
            "purge quarantine",
            "Purge quarantine",
            "purge quarantined entries",
            "Purge quarantined entries",
            "clean quarantine",
            "Clean quarantine",
        ],
        trigger_patterns=[
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?[Pp]urge\s+(?:the\s+)?quarantin(?:e|ed)(?:\s+entries)?$",
            r"^(?:[Aa]stra[,:]?\s*)?[Cc]lean\s+(?:the\s+)?quarantine$",
        ],
        requires_context=[],
        requires_confirmation=True,
        confirmation_prompt=(
            "This will permanently delete all quarantined RAG entries "
            "across arch_code_chunks, architecture_file_index, and rag_entries. "
            "This cannot be undone. Proceed?"
        ),
        description="Permanently remove quarantined RAG entries after refactoring",
        behavior=(
            "Purge all quarantined entries:\n"
            "1. Delete quarantined arch_code_chunks\n"
            "2. Delete quarantined architecture_file_index entries\n"
            "3. Delete quarantined rag_entries\n"
            "4. Remove .quarantined/ directories from disk\n"
            "5. Report counts of purged items"
        ),
    ),

    CanonicalIntent.QUARANTINE_STATUS: IntentDefinition(
        intent=CanonicalIntent.QUARANTINE_STATUS,
        trigger_phrases=[
            "Astra, command: quarantine status",
            "astra, command: quarantine status",
            "quarantine status",
            "Quarantine status",
            "show quarantine",
            "Show quarantine",
            "how many quarantined",
            "How many quarantined",
        ],
        trigger_patterns=[
            r"^(?:[Aa]stra[,:]?\s*)?(?:command[:\s]+)?(?:[Ss]how\s+)?quarantine\s+status$",
            r"^(?:[Hh]ow\s+many|[Ss]how)\s+quarantined(?:\s+entries)?$",
        ],
        requires_context=[],
        requires_confirmation=False,
        description="Show counts of quarantined entries across all RAG tables",
        behavior=(
            "Display quarantine statistics:\n"
            "1. Count quarantined arch_code_chunks\n"
            "2. Count quarantined architecture_file_index entries\n"
            "3. Count quarantined rag_entries\n"
            "4. Report total and per-table counts"
        ),
    ),
}
