from app.memory.domains.dependency_scanner import _iter_python_files_with_content, _extract_imports_from_source, _make_relative

root = r"D:\Orb"
skip = {"__pycache__", ".git", "node_modules", "data", "jobs", ".venv", ".architecture"}

count = 0
has_content = 0
has_imports = 0
for filepath, content in _iter_python_files_with_content(root, skip):
    count += 1
    if content:
        has_content += 1
        imports = _extract_imports_from_source(content, filepath)
        if imports:
            has_imports += 1
            if has_imports <= 3:
                rel = _make_relative(filepath, root)
                print(f"  {rel}: {imports[:3]}")
    if count >= 30:
        break

print(f"Checked {count} files, {has_content} had content, {has_imports} had imports")
