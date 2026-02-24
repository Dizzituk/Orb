from app.translation._tier0_web_search import check_web_search_trigger

tests = [
    'search the web for brave search api pricing',
    'look up python 3.13 release date',
    'get me the latest on AI regulations UK',
    'google fastapi websocket tutorial',
    'what is the current price of bitcoin',
    'research electric van options for delivery',
    'whats the latest on openai',
    'find out about brave search api',
    # These should NOT match:
    'search codebase for memory leaks',
    'look at the pipeline module',
    'hello',
    'run critical pipeline',
]

for t in tests:
    r = check_web_search_trigger(t)
    if r.matched:
        print(f"  MATCH ({r.rule_name}, q={r.extracted_query}) <- \"{t}\"")
    else:
        print(f"  no match <- \"{t}\"")
