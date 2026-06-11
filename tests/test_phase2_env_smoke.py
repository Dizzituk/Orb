import sys, asyncio
sys.path.insert(0, "D:/Orb")

# --- Job 5: test_env loader ---
from app.pipeline_v2.test_env import load_test_env

env = load_test_env("astra-bridge")
assert env is not None, "astra-bridge test_env missing"
assert env.logged_in_marker == "Logged in to ASTRA", env.logged_in_marker
assert env.password_field_marker == "ASTRA Password", env.password_field_marker
assert env.has_credentials() is False, "placeholder password should not count as credentials"
assert env.backend_url_from_device.startswith("http://10.0.2.2"), env.backend_url_from_device
red = env.redacted()
assert red["password"] == "***", red
assert load_test_env("no-such-target") is None
print("test_env OK:", {k: red[k] for k in ("target_id", "logged_in_marker", "avd_name")})

# --- Job 7: failure classifier (deterministic, no ADB paths) ---
from app.pipeline_v2.bvl.failure_classifier import classify_failure, ENVIRONMENT, CODE

class FakeCheck:
    def __init__(self, name, passed):
        self.name = name
        self.passed = passed

class FakeTier:
    def __init__(self, error_context="", checks=None):
        self.error_context = error_context
        self.checks = checks or []

class FakeProfile:
    package_name = "com.astra.astrabridge"
    project_id = "astra-bridge"

async def main():
    # gradle failure -> code
    c1 = await classify_failure(FakeTier("compile error", [FakeCheck("gradle_build", False)]), FakeProfile())
    assert c1.kind == CODE, c1

    # emulator_ready failed -> environment
    c2 = await classify_failure(FakeTier("no device", [FakeCheck("gradle_build", True), FakeCheck("emulator_ready", False)]), FakeProfile())
    assert c2.kind == ENVIRONMENT, c2
    assert "emulator" in c2.reason.lower(), c2.reason

    # error-pattern match -> environment
    c3 = await classify_failure(FakeTier("No emulator device found after boot wait"), FakeProfile())
    assert c3.kind == ENVIRONMENT, c3

    # app_launch crash -> code
    c4 = await classify_failure(FakeTier("crash", [FakeCheck("app_launch", False)]), FakeProfile())
    assert c4.kind == CODE, c4

    # apk_install -> environment
    c5 = await classify_failure(FakeTier("install fail", [FakeCheck("apk_install", False)]), FakeProfile())
    assert c5.kind == ENVIRONMENT, c5

    # default -> code
    c6 = await classify_failure(FakeTier("mystery", [FakeCheck("something_else", False)]), FakeProfile())
    assert c6.kind == CODE, c6

    print("failure_classifier OK (6 deterministic paths)")

asyncio.run(main())

# --- imports of new modules wire up cleanly ---
import importlib
for mod in ("app.pipeline_v2.bvl.preflight", "app.pipeline_v2.final_verdict"):
    importlib.import_module(mod)
print("module imports OK")
print("PHASE 2 SMOKE: ALL PASS")
