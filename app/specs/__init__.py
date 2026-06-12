# FILE: app/specs/__init__.py
# Purpose: ASTRA Specs Module
# Called-by: app.db, app.llm._sg_stream_persist, app.llm._spec_gate_stream_utils_2, app.pot_spec._spec_gate_persistence_utils_2 (+2 more)
# Depends-on: app.specs.models, app.specs.schema, app.specs.service
# Last-renovated: 2026-06-11
"""
ASTRA Specs Module

Provides spec storage, schema validation, and service functions for the
Weaver → Spec Gate → Pipeline flow.

Usage:
    from app.specs import Spec, SpecSchema, create_spec, get_spec
    
    # Create a spec from Weaver output
    spec_schema = SpecSchema(
        title="My Feature",
        objective="Build a thing",
        ...
    )
    db_spec = create_spec(db, project_id, spec_schema)
    
    # Retrieve and validate
    spec = get_spec(db, spec_id)
    schema = get_spec_schema(spec)
    result = validate_spec(schema)
"""

# Schema and validation
from .schema import (
    SPEC_SCHEMA_VERSION,
    SpecStatus,
    Spec as SpecSchema,
    SpecInput,
    SpecOutput,
    SpecStep,
    SpecRequirements,
    SpecConstraints,
    SpecSafety,
    SpecMetadata,
    SpecProvenance,
    SpecValidationResult,
    validate_spec,
    spec_to_markdown,
)

# Database models
from .models import (
    Spec,
    SpecQuestion,
    SpecHistory,
)

# Service functions
from .service import (
    create_spec,
    get_spec,
    get_spec_by_db_id,
    get_latest_spec,
    get_latest_draft_spec,
    get_latest_validated_spec,
    list_specs,
    update_spec_status,
    add_spec_history,
    create_spec_question,
    answer_spec_question,
    get_pending_questions,
    get_spec_schema,
    check_duplicate_spec,
)

__all__ = [
    # Schema version
    "SPEC_SCHEMA_VERSION",
    "SpecStatus",
    
    # Schema classes
    "SpecSchema",
    "SpecInput",
    "SpecOutput",
    "SpecStep",
    "SpecRequirements",
    "SpecConstraints",
    "SpecSafety",
    "SpecMetadata",
    "SpecProvenance",
    "SpecValidationResult",
    
    # Schema functions
    "validate_spec",
    "spec_to_markdown",
    
    # Database models
    "Spec",
    "SpecQuestion",
    "SpecHistory",
    
    # Service functions
    "create_spec",
    "get_spec",
    "get_spec_by_db_id",
    "get_latest_spec",
    "get_latest_draft_spec",
    "get_latest_validated_spec",
    "list_specs",
    "update_spec_status",
    "add_spec_history",
    "create_spec_question",
    "answer_spec_question",
    "get_pending_questions",
    "get_spec_schema",
    "check_duplicate_spec",
]
