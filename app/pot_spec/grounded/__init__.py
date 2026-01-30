# FILE: app/pot_spec/grounded/__init__.py
# v1.27 (2026-01-28): Added MultiFileOperation for Level 3 multi-file operations
# v1.26 (2026-01-28): Added file_discovery module for multi-file operations
"""
SpecGate Grounded Modules Package (v1.25)

This package contains the modularized components of spec_gate_grounded.py:

- spec_models: Data structures (GroundedPOTSpec, GroundedQuestion, etc.)
- domain_detection: Domain/intent classification and decision forks
- job_classification: Job kind determination (micro, repo, arch, scan)
- scan_operations: SCAN_ONLY security and parameter extraction
- sandbox_discovery: Sandbox file resolution and output mode detection
- tech_stack_detection: Implementation stack anchoring (v1.11)
- qa_processing: Q&A file handling and LLM reply generation
- evidence_gathering: v1.25 Evidence-First filesystem validation
- file_discovery: v1.0 Multi-file pattern search (Level 3)

Version: v1.27 (Added MultiFileOperation)
"""

# spec_models
from .spec_models import (
    QuestionCategory,
    GroundedFact,
    GroundedAssumption,
    GroundedQuestion,
    MultiFileOperation,
    GroundedPOTSpec,
    MIN_QUESTIONS,
    MAX_QUESTIONS,
)

# domain_detection
from .domain_detection import (
    DOMAIN_KEYWORDS,
    MOBILE_APP_FORK_BANK,
    detect_domains,
    extract_unresolved_ambiguities,
    extract_decision_forks,
)

# job_classification
from .job_classification import (
    EVIDENCE_CONFIG,
    classify_job_kind,
    classify_job_size,
)

# scan_operations
from .scan_operations import (
    DEFAULT_SCAN_EXCLUSIONS,
    SAFE_DEFAULT_SCAN_ROOTS,
    FORBIDDEN_SCAN_ROOTS,
    validate_scan_roots,
    extract_scan_params,
)

# sandbox_discovery
from .sandbox_discovery import (
    OutputMode,
    SUBFOLDER_STOPWORDS,
    extract_sandbox_hints,
    detect_output_mode,
    extract_replacement_text,
)

# tech_stack_detection
from .tech_stack_detection import (
    STACK_DETECTION_PATTERNS,
    STACK_CHOICE_INDICATORS,
    CONFIRMATION_PATTERNS,
    detect_implementation_stack,
)

# qa_processing
from .qa_processing import (
    analyze_qa_file,
    detect_simple_instruction,
    generate_reply_from_content,
)

# evidence_gathering
from .evidence_gathering import (
    FilesystemEvidenceSource,
    FileEvidence,
    EvidencePackage,
    EVIDENCE_ALLOWED_ROOTS,
    EVIDENCE_FORBIDDEN_PATHS,
    ANCHOR_RESOLUTION_MAP,
    resolve_path_enhanced,
    extract_path_references,
    detect_file_structure,
    resolve_and_validate_path,
    gather_filesystem_evidence,
    format_evidence_for_prompt,
)

# file_discovery (v1.0 - Level 3 multi-file operations)
from .file_discovery import (
    LineMatch,
    FileMatch,
    DiscoveryResult,
    discover_files,
    discover_files_by_extension,
    DEFAULT_ROOTS,
    DEFAULT_EXCLUSIONS,
    DEFAULT_FILE_EXTENSIONS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_MAX_RESULTS,
)

# spec_generation (main entry point)
from .spec_generation import (
    parse_weaver_intent,
    ground_intent_with_evidence,
    generate_grounded_questions,
    build_pot_spec_markdown,
    run_spec_gate_grounded,
)


__all__ = [
    # spec_models
    "QuestionCategory",
    "GroundedFact",
    "GroundedAssumption",
    "GroundedQuestion",
    "MultiFileOperation",
    "GroundedPOTSpec",
    "MIN_QUESTIONS",
    "MAX_QUESTIONS",
    # domain_detection
    "DOMAIN_KEYWORDS",
    "MOBILE_APP_FORK_BANK",
    "detect_domains",
    "extract_unresolved_ambiguities",
    "extract_decision_forks",
    # job_classification
    "EVIDENCE_CONFIG",
    "classify_job_kind",
    "classify_job_size",
    # scan_operations
    "DEFAULT_SCAN_EXCLUSIONS",
    "SAFE_DEFAULT_SCAN_ROOTS",
    "FORBIDDEN_SCAN_ROOTS",
    "validate_scan_roots",
    "extract_scan_params",
    # sandbox_discovery
    "OutputMode",
    "SUBFOLDER_STOPWORDS",
    "extract_sandbox_hints",
    "detect_output_mode",
    "extract_replacement_text",
    # tech_stack_detection
    "STACK_DETECTION_PATTERNS",
    "STACK_CHOICE_INDICATORS",
    "CONFIRMATION_PATTERNS",
    "detect_implementation_stack",
    # qa_processing
    "analyze_qa_file",
    "detect_simple_instruction",
    "generate_reply_from_content",
    # evidence_gathering
    "FilesystemEvidenceSource",
    "FileEvidence",
    "EvidencePackage",
    "EVIDENCE_ALLOWED_ROOTS",
    "EVIDENCE_FORBIDDEN_PATHS",
    "ANCHOR_RESOLUTION_MAP",
    "resolve_path_enhanced",
    "extract_path_references",
    "detect_file_structure",
    "resolve_and_validate_path",
    "gather_filesystem_evidence",
    "format_evidence_for_prompt",
    # file_discovery
    "LineMatch",
    "FileMatch",
    "DiscoveryResult",
    "discover_files",
    "discover_files_by_extension",
    "DEFAULT_ROOTS",
    "DEFAULT_EXCLUSIONS",
    "DEFAULT_FILE_EXTENSIONS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RESULTS",
    # spec_generation
    "parse_weaver_intent",
    "ground_intent_with_evidence",
    "generate_grounded_questions",
    "build_pot_spec_markdown",
    "run_spec_gate_grounded",
]

__version__ = "1.27"
