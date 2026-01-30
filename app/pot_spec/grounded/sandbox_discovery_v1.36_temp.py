# This is just to test if I can write - checking the pattern issue
drive_patterns = [
    # v1.36 NEW: Reverse order - 'test2' on D: drive / 'test2' on D drive  
    # This is the ACTUAL Weaver format!
    (r"['\"](\w+(?:\.\w+)?)['\"]" + r"\s+(?:on|in|from)\s+(?:the\s+)?([A-Za-z]):\s*(?:drive)?", 'weaver_quoted_before_drive_v2'),
]
