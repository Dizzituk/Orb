from app.orchestrator.codebase_scanner import scan_file

test_files = [
    ("app/translation/intents.py", "DATA-ONLY - should NOT flag structure"),
    ("app/translation/tier0_rules.py", "Recently decomposed"),
    ("app/pot_spec/grounded/spec_runner.py", "Single orchestrator - should NOT flag"),
    ("app/llm/weaver_stream.py", "Recently decomposed"),
    ("app/jobs/engine.py", "Recently decomposed"),
    ("app/overwatcher/architecture_executor/orchestrator.py", "95KB single orchestrator"),
]

STRUCTURAL_CATS = {
    "oversized_file", "oversized_function",
    "multi_responsibility", "monolithic_function",
    "extractable_block", "god_class", "tangled_dependencies",
}

for path, desc in test_files:
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        result = scan_file(path, src)
        
        structural = [i for i in result.health_issues if i.category.value in STRUCTURAL_CATS]
        
        print(f"\n{'='*70}")
        print(f"{path}")
        print(f"  ({desc})")
        print(f"  {result.line_count} lines | {len(result.symbols)} symbols | "
              f"{result.function_count} funcs | {result.class_count} classes")
        
        if structural:
            for issue in structural:
                print(f"  >> [{issue.category.value}] {issue.description}")
                if issue.suggestion:
                    print(f"     Suggestion: {issue.suggestion}")
        else:
            print(f"  >> No structural issues (correct!)")
            
    except Exception as e:
        print(f"\nERROR scanning {path}: {e}")
