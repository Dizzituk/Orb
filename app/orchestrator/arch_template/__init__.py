# FILE: app/orchestrator/arch_template/__init__.py
"""
Architecture Template Engine — Job 4.

Generates partially-filled architecture documents from deterministic sources
(skeleton contracts, segment specs, design tokens, pattern references).
The LLM fills only the creative gaps marked with [LLM_FILL].

v1.0 (2026-03-01): Initial implementation.

Modules:
- engine.py: Main entry point, orchestrates template generation
- sections.py: Individual section generators (file inventory, imports, etc.)
- token_registry.py: Design token extraction from CSS files
"""

ARCH_TEMPLATE_BUILD_ID = "2026-03-01-v1.0-deterministic-architecture"
print(f"[ARCH_TEMPLATE_LOADED] BUILD_ID={ARCH_TEMPLATE_BUILD_ID}")
