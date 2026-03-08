import json
import re
from html import unescape

with open('coursera_test.html', 'r', encoding='utf-8') as f:
    html = f.read()

def _walk_json(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)

def _extract_from_json_ld(html: str):
    modules = []
    for raw in re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', html, re.S | re.I):
        try:
            payload = json.loads(unescape(raw.strip()))
        except Exception:
            continue
        print("FOUND JSON-LD")
        for item in _walk_json(payload):
            if not isinstance(item, dict):
                continue
            title = item.get("name") or item.get("title")
            desc = item.get("description")
            if title and isinstance(title, str):
                modules.append((title, desc))
    return modules

print("JSON-LD modules:")
for m in _extract_from_json_ld(html):
    print(f"- {m[0]}")
