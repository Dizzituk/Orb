# Purpose: bug
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
def add(a, b):
    return a - b  # BUG: should add
