"""Run a detailed import test on the sandbox."""
import sys, json
sys.path.insert(0, r'D:\Orb')

from app.overwatcher.sandbox_client import get_sandbox_client

client = get_sandbox_client()

# First: write a test script to sandbox via file_write
test_code = """import sys, traceback
sys.path.insert(0, r'D:\\\\Orb')
try:
    from app.translation.intents import INTENT_DEFINITIONS
    print('INTENTS_PKG_OK:', type(INTENT_DEFINITIONS))
except:
    print('INTENTS_PKG_FAIL')
    traceback.print_exc()

try:
    from app.translation import CanonicalIntent
    print('CANONICAL_OK:', CanonicalIntent.RUN_PIPELINE)
except:
    print('CANONICAL_FAIL')
    traceback.print_exc()
"""

# Use file_write endpoint
client.write_file("test_import_check.py", test_code)

# Run it
result = client.shell_run("cd D:\\Orb && D:\\Orb\\.venv\\Scripts\\python.exe test_import_check.py")

print("=== STDOUT ===")
print(result.get('stdout', '(empty)'))
print("=== STDERR ===") 
print(result.get('stderr', '(empty)'))

# Cleanup
client.shell_run("del D:\\Orb\\test_import_check.py")
