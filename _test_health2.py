from app.orchestrator.codebase_scanner import scan_file

test_files = [
    ("app/rag/jobs/embedding_job.py", "Class-heavy file"),
    ("app/llm/preprocessor.py", "14 functions, well-distributed"),
    ("app/orchestrator/surgical_extractor.py", "16 functions, modular"),
    ("app/llm/fallbacks.py", "Class + functions mix"),
    ("app/overwatcher/overwatcher_command.py", "Single 704-line async fn"),
    ("app/llm/spec_gate_stream.py", "Single 640-line async generator"),
    ("app/pot_spec/grounded/simple_create.py", "413-line async fn"),
    ("app/llm/pipeline/high_stakes.py", "Single orchestrator"),
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
        
        print(f"\n{path}")
        print(f"  ({desc})")
        print(f"  {result.line_count} lines | {len(result.symbols)} syms | "
              f"{result.function_count} funcs | {result.class_count} classes")
        
        if structural:
            for issue in structural:
                print(f"  >> [{issue.category.value}] {issue.description}")
        else:
            print(f"  >> No structural issues")
            
    except Exception as e:
        print(f"\nERROR scanning {path}: {e}")
