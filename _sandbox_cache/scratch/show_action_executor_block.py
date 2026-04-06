from pathlib import Path
text = Path("D:/Orb/app/debug/action_executor.py").read_text(encoding="utf-8")
start = text.index('_HOST_ONLY_PREFIXES = [')
end = text.index('def _is_host_only', start)
print(repr(text[start:end]))
