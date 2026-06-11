# FILE: app/pipeline_v2/test_env.py
"""
JOB 5 (2026-06-10) - Per-build-target test environment manifest.

The verification stack repeatedly failed for environmental reasons (emulator
not running, app not logged in, no credentials) and burnt builder fix cycles
on non-bugs. This module gives every build target a declared test
environment: a TEST-ONLY login credential, the backend URL as seen FROM the
device/emulator, the AVD to boot, and the UI markers that tell automation
"this is the login screen" / "we are logged in".

Storage: D:/Orb/config/test_env.json keyed by build-target id, with env-var
overrides (ASTRA_TESTENV_<TARGETID>_PASSWORD etc., target id uppercased,
dashes -> underscores). Credentials are TEST-ONLY, live in local config,
and must never appear in specs, chat logs, or pipeline narratives.

Single responsibility: load + represent the test environment. No ADB here.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join("D:\\Orb", "config", "test_env.json")
PLACEHOLDERS = {"", "CHANGE_ME", "TODO", "<set-me>"}


@dataclass
class TestEnv:
    target_id: str = ""
    username: str = ""
    password: str = ""
    backend_url_from_device: str = ""   # e.g. http://10.0.2.2:8000 (emulator -> host)
    avd_name: str = ""                  # AVD to auto-start if no emulator running
    main_activity: str = ".MainActivity"
    # UI markers (matched fuzzily against text / content-desc in the UI tree)
    url_field_marker: str = ""
    password_field_marker: str = ""
    login_button_text: str = "Login"
    save_button_text: str = ""
    logged_in_marker: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    def has_credentials(self) -> bool:
        return (self.password or "").strip() not in PLACEHOLDERS

    def redacted(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        if d.get("password"):
            d["password"] = "***"
        return d


def load_test_env(target_id: str) -> Optional[TestEnv]:
    """Load the test environment for a build target. Never raises.

    Env-var overrides win over the JSON file:
      ASTRA_TESTENV_ASTRA_BRIDGE_PASSWORD, ..._USERNAME, ..._BACKEND_URL,
      ..._AVD_NAME (target id uppercased, dashes -> underscores).
    """
    if not target_id:
        return None
    data: Dict[str, Any] = {}
    try:
        if os.path.isfile(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                all_cfg = json.load(f)
            data = all_cfg.get(target_id) or {}
    except Exception as exc:
        logger.warning("[test_env] Could not read %s: %s", CONFIG_PATH, exc)

    env_key = target_id.upper().replace("-", "_")
    overrides = {
        "password": os.getenv(f"ASTRA_TESTENV_{env_key}_PASSWORD"),
        "username": os.getenv(f"ASTRA_TESTENV_{env_key}_USERNAME"),
        "backend_url_from_device": os.getenv(f"ASTRA_TESTENV_{env_key}_BACKEND_URL"),
        "avd_name": os.getenv(f"ASTRA_TESTENV_{env_key}_AVD_NAME"),
    }
    for k, v in overrides.items():
        if v:
            data[k] = v

    if not data:
        return None

    known = {f.name for f in TestEnv.__dataclass_fields__.values()} - {"extras", "target_id"}
    fields = {k: v for k, v in data.items() if k in known}
    extras = {k: v for k, v in data.items() if k not in known}
    return TestEnv(target_id=target_id, extras=extras, **fields)


def get_test_env_for_profile(profile: Any) -> Optional[TestEnv]:
    """Convenience: resolve from a BuildTargetProfile."""
    target_id = str(getattr(profile, "project_id", "") or "")
    return load_test_env(target_id)
