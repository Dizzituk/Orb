"""Debug the dependency scanner sandbox walk."""
from app.sandbox_walk import sandbox_walk

print("=== Walking D:\\Orb ===")
count = 0
py_count = 0
for dirpath, dirnames, filenames in sandbox_walk(r"D:\Orb"):
    py_files = [f for f in filenames if f.endswith(".py")]
    py_count += len(py_files)
    count += 1
    if count <= 3:
        print(f"  {dirpath}: {len(dirnames)} dirs, {len(filenames)} files ({len(py_files)} .py)")

print(f"  ... total dirs: {count}, total .py: {py_count}")

print()
print("=== Walking D:\\Orb\\app ===")
count2 = 0
py_count2 = 0
for dirpath, dirnames, filenames in sandbox_walk(r"D:\Orb\app"):
    py_files = [f for f in filenames if f.endswith(".py")]
    py_count2 += len(py_files)
    count2 += 1

print(f"  dirs: {count2}, .py files: {py_count2}")

print()
print("=== Testing _make_relative ===")
from app.memory.domains.dependency_scanner import _make_relative
tests = [
    (r"D:\Orb\app\db.py", r"D:\Orb"),
    (r"D:\Orb\app\memory\router.py", r"D:\Orb"),
    (r"D:\Orb\main.py", r"D:\Orb"),
]
for fp, root in tests:
    rel = _make_relative(fp, root)
    print(f"  {fp} -> {rel}")

print()
print("=== Testing _build_module_map ===")
from app.memory.domains.dependency_scanner import _build_module_map
skip = {"__pycache__", ".git", "node_modules", "data", "jobs", ".venv", ".architecture"}
mmap = _build_module_map(r"D:\Orb", skip)
print(f"  Module map entries: {len(mmap)}")
for k, v in sorted(mmap.items())[:10]:
    print(f"  {k} -> {v}")
