# Purpose: bug check
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
from bug import add
assert add(2, 3) == 5, 'still broken'
print('CHECK_PASS')
