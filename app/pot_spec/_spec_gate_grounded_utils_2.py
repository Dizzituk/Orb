from __future__ import annotations

def _version_check():
    """Quick diagnostic to verify module loading."""
    import sys
    grounded_modules = [k for k in sys.modules.keys() if 'grounded' in k]
    return {
        'version': 'v1.25 (modularized)',
        'loaded_modules': grounded_modules,
        'main_entry': 'run_spec_gate_grounded',
    }
