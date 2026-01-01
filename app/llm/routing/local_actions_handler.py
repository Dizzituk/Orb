# FILE: Add to D:\Orb\app\llm\routing\local_actions.py
# 
# STEP 1: Add this import near line 20-30 (with other imports):
#
#     from app.llm.local_tools import arch_query
#
# STEP 2: Add this function after _maybe_handle_zobie_map (around line 300):


async def _maybe_handle_arch_query(task: "LLMTask", original_message: str) -> Optional["LLMResult"]:
    """Handle architecture query requests using arch_query_service."""
    msg_lower = original_message.lower()
    
    # Trigger patterns
    triggers = ["structure of", "signatures", "find function", "find class", 
                "what's in", "whats in", "search for"]
    has_py = ".py" in msg_lower
    has_struct = any(w in msg_lower for w in ["structure", "signature", "function", "class", "method"])
    
    if not (any(t in msg_lower for t in triggers) or (has_py and has_struct)):
        return None
    
    if not arch_query.is_service_available():
        _debug_log("arch_query_service not available")
        return None
    
    _debug_log(f"Handling arch query: {original_message[:80]}")
    
    try:
        result = arch_query.query_architecture(original_message)
        if result.startswith("Error:"):
            return None
        
        from app.llm.schemas import LLMResult
        return LLMResult(
            content=result,
            provider="local",
            model="arch_query",
            tokens_used=0,
            cost=0.0,
        )
    except Exception as e:
        _debug_log(f"Arch query exception: {e}")
        return None


# STEP 3: Find _maybe_handle_local_action function and add this call
#         (near where _maybe_handle_zobie_map is called):
#
#     # Architecture queries
#     arch_result = await _maybe_handle_arch_query(task, original_message)
#     if arch_result:
#         return arch_result
