# FILE: app/translation/_tier0_filesystem.py
# Purpose: Tier 0 filesystem, multi-file, refactor, and chat pattern rules.
# Called-by: app.translation.tier0_rules
# Depends-on: app.translation._tier0_filesystem_query, app.translation._tier0_multifile_rules, app.translation._tier0_chat_patterns
# Last-renovated: 2026-06-21
"""
Tier 0 filesystem, multi-file, refactor, and chat pattern rules.

BATCH 4 split: the four independent classifiers moved to single-responsibility
modules; this module is now a thin re-export shim so tier0_rules.py imports the
same 10 symbols unchanged.
Contains: check_multi_file_trigger, check_refactor_codebase_trigger,
check_filesystem_query_trigger, is_user_chat_pattern, and supporting constants.
"""
from __future__ import annotations
from app.translation._tier0_filesystem_query import (
    _KNOWN_FOLDER_KEYWORDS,
    _ALLOWED_FS_ROOTS,
    _has_windows_path,
    _has_known_folder_keyword,
    _is_within_allowed_roots,
    check_filesystem_query_trigger,
)
from app.translation._tier0_multifile_rules import (
    MULTI_FILE_SEARCH_PATTERNS,
    MULTI_FILE_REFACTOR_PATTERNS,
    MULTI_FILE_SCOPE_KEYWORDS,
    check_multi_file_trigger,
    _REFACTOR_PATTERNS,
    _REFACTOR_EXACT_PHRASES,
    check_refactor_codebase_trigger,
)
from app.translation._tier0_chat_patterns import (
    USER_CHAT_PATTERNS,
    _COMPILED_USER_CHAT,
    is_user_chat_pattern,
    TAZISH_CHAT_PATTERNS,
    _COMPILED_TAZISH_CHAT,
    is_tazish_chat,
)
