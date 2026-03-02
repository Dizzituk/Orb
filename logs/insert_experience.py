"""
Insert experience patterns learned from sg-2a3c378d Education Tab pipeline run.
These cover gaps not captured by the cohesion auto-fix distillation.
"""
import sys, os
sys.path.insert(0, r'D:\Orb')
os.chdir(r'D:\Orb')

from app.db import SessionLocal, init_db
init_db()
db = SessionLocal()

from app.experience.experience_store import create_pattern

patterns = [
    # 1. Frontend scope detection failure
    {
        "category": "architecture_decision",
        "stage": "spec_runner",
        "description": (
            "SpecGate scope detection uses a static keyword dict (SCOPE_FRONTEND) that only matches "
            "literal words like 'frontend', 'electron', 'the ui'. When the user describes visual elements "
            "(dark theme, cards, progress bars, dashboard layout) or Weaver generates .tsx/.jsx file references, "
            "scope detection fails because these frontend indicators are not in the keyword dict. "
            "This causes FINAL PATHS to drop the frontend root (D:\\orb-desktop), and the LLM invents "
            "non-existent paths like D:\\Orb\\frontend\\src\\components\\."
        ),
        "root_cause": "static_keyword_scope_detection",
        "resolution": (
            "Add content-aware scope detection (Step 3c) that checks for .tsx, .jsx, .vue, .svelte, "
            "react, component, tailwind, dashboard.tsx, card.tsx in the text. If any are found, "
            "set has_frontend_scope=True before the Step 3b injection runs. Long-term: replace keyword "
            "dict with INDEX.json zone lookups."
        ),
        "job_type": "architecture",
        "language": "python",
        "file_scope": "app/pot_spec/grounded/_spec_runner_utils_12.py",
        "source_job_id": "sg-2a3c378d",
        "initial_confidence": 0.9,
    },
    # 2. Codebase convention drift - PK types
    {
        "category": "architecture_decision",
        "stage": "critical_pipeline",
        "description": (
            "LLM chose UUID string primary keys for new Education models, but existing codebase models "
            "use Integer auto-increment PKs. The architecture prompt did not instruct the LLM to match "
            "existing codebase conventions for primary key types, auth patterns, and naming."
        ),
        "root_cause": "missing_convention_matching_directive",
        "resolution": (
            "Added CODEBASE CONVENTION MATCHING section to architecture system prompt in prompt_builder.py. "
            "Directs LLM to match existing PK types, auth patterns, naming conventions, import style, "
            "error handling, and config patterns. Deviation requires a DECISION block with justification."
        ),
        "job_type": "architecture",
        "language": "python",
        "file_scope": "app/llm/critical_pipeline/prompt_builder.py",
        "source_job_id": "sg-2a3c378d",
        "initial_confidence": 0.8,
    },
    # 3. Missing frontend service bridge
    {
        "category": "architecture_decision",
        "stage": "cohesion_check",
        "description": (
            "Frontend React components reference a frontend API service (educationApi.ts) for HTTP calls "
            "to backend endpoints, but no segment created this file. Multiple components import from "
            "'../../services/educationApi' which does not exist, causing import errors. When a backend "
            "API segment creates endpoints, a corresponding frontend service bridge file should be "
            "included in the segmentation."
        ),
        "root_cause": "missing_frontend_service_bridge",
        "resolution": (
            "SpecGate segmentation should detect when both backend API routes and frontend UI components "
            "exist in the same job, and auto-generate a segment for the frontend API service layer "
            "(e.g. src/services/xxxApi.ts) that bridges the two. Alternatively, the cohesion check "
            "should flag unresolved frontend imports to non-existent service files."
        ),
        "job_type": "architecture",
        "language": "typescript",
        "file_scope": "src/services/educationApi.ts",
        "source_job_id": "sg-2a3c378d",
        "initial_confidence": 0.7,
    },
    # 4. Dark theme styling not carried through
    {
        "category": "architecture_decision",
        "stage": "critical_pipeline",
        "description": (
            "Weaver captured dark theme visual requirements (#111217 bg, #1E1D2B cards, #4ADE80 accents) "
            "from the user's description and screenshot context, but the architecture and implementation "
            "briefs generated generic CSS class names without any actual color values, theme variables, "
            "or reference to the existing design system. Visual styling intent from Weaver is lost "
            "by the time it reaches the implementer."
        ),
        "root_cause": "visual_context_lost_in_pipeline",
        "resolution": (
            "The architecture prompt should extract and forward Weaver's vision context (color values, "
            "spacing, font weights, existing component CSS patterns) into the architecture doc. "
            "The implementation brief should include a STYLING section with concrete CSS variable "
            "references or inline style values from the existing theme."
        ),
        "job_type": "architecture",
        "language": "typescript",
        "source_job_id": "sg-2a3c378d",
        "initial_confidence": 0.7,
    },
    # 5. Spec compliance false positive on frontend segments
    {
        "category": "architecture_decision",
        "stage": "critique",
        "description": (
            "The deterministic spec-compliance checker flagged a BLOCKING issue (SPEC-COMPLIANCE-001) on "
            "seg-05-education-tab-core-ui because the architecture used TypeScript/React stack but the "
            "broader job's discussed_stack included Python/FastAPI. This is a false positive: frontend "
            "segments in a full-stack job legitimately use only TypeScript/React. The checker does not "
            "distinguish between segment scope and parent job scope."
        ),
        "root_cause": "spec_compliance_ignores_segment_scope",
        "resolution": (
            "The spec-compliance deterministic check should compare the segment's architecture stack "
            "against the segment-scoped spec (which specifies file extensions), not the full parent "
            "job's discussed_stack. If a segment only contains .tsx files, TypeScript/React is the "
            "correct and only valid stack regardless of what other segments use."
        ),
        "job_type": "architecture",
        "language": "typescript",
        "file_scope": "app/llm/pipeline/critique_parts/spec_compliance.py",
        "source_job_id": "sg-2a3c378d",
        "initial_confidence": 0.85,
    },
    # 6. Weaver conversation depth improves output quality
    {
        "category": "strategy",
        "stage": "weaver",
        "description": (
            "A 7-message Weaver conversation produced a spec with 30 integration points, correct "
            "dual-root path discovery, and 7 well-scoped segments. A single-prompt version of the "
            "same job produced only 15 integration points, incorrect paths, and 3 segments. "
            "Weaver's prodding questions draw out visual requirements, navigation patterns, "
            "interaction details, and domain context that a single prompt misses."
        ),
        "root_cause": "insufficient_requirements_gathering",
        "resolution": (
            "Weaver should always attempt at least 3-4 rounds of clarifying questions before "
            "generating the final job description, especially for UI-heavy features. Key areas "
            "to probe: visual styling/theme, navigation flow, data sources, interaction patterns, "
            "and phasing (what's manual now vs automated later)."
        ),
        "source_job_id": "sg-2a3c378d",
        "initial_confidence": 0.75,
    },
    # 7. Sanitiser stripping valid sub-router files
    {
        "category": "architecture_decision",
        "stage": "critical_pipeline",
        "description": (
            "The architecture sanitiser stripped 3 sub-router files (ingestion_router.py, "
            "analogy_router.py, curriculum_router.py) from seg-04 as 'hallucination / out-of-scope', "
            "even though the architecture's own DECISION D-003 justified the split to meet the 20KB "
            "file size constraint. The sanitiser doesn't check whether added files have a corresponding "
            "DECISION block justifying them."
        ),
        "root_cause": "sanitiser_ignores_decision_justified_files",
        "resolution": (
            "The sanitiser should check for a DECISION block that references added files before "
            "stripping them. If a DECISION explicitly states 'file X is being added to meet size "
            "constraints', the sanitiser should allow it. Alternatively, the segment spec should "
            "include a 'may_add_files_for_size' flag."
        ),
        "job_type": "architecture",
        "language": "python",
        "file_scope": "app/orchestrator/architecture_sanitiser.py",
        "source_job_id": "sg-2a3c378d",
        "initial_confidence": 0.7,
    },
]

count = 0
for p in patterns:
    try:
        create_pattern(db, **p)
        count += 1
        print(f"  Created: {p['description'][:80]}...")
    except Exception as e:
        print(f"  FAILED: {e}")

db.commit()
db.close()
print(f"\nDone: {count}/{len(patterns)} patterns inserted")
