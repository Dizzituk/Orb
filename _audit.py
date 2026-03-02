import re, sys

pattern = re.compile(
    r'^##\s+\d+\)\s+`([^`]+)`\s*\((CREATE|MODIFY)\)',
    re.MULTILINE,
)

test = """## 1) `app/education/api.py` (CREATE)

Some description

```python
from fastapi import APIRouter
router = APIRouter()

async def create_course():
    pass
```

## 2) `app/education/models.py` (MODIFY)

Another description
"""

matches = pattern.findall(test)
print(f"Matches: {matches}")
if len(matches) != 2:
    print("BUG: Expected 2 matches")
    sys.exit(1)
print("PASS: Pattern matches architecture sections correctly")