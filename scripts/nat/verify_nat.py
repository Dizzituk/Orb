# verify_nat.py — Prove Nat is live once vLLM is serving on VLLM_BASE_URL.
# Run from D:\Orb with the backend venv:
#     .venv\Scripts\python.exe scripts\nat\verify_nat.py
#
# Loads .env (so it uses the same NAT_PROVIDER/NAT_MODEL/VLLM_BASE_URL as ASTRA),
# pings the server, then runs a real keyword-extraction job through run_nat_job
# and times the round-trip. Exit 0 = Nat answered; non-zero = not ready.
import asyncio
import json
import os
import sys
import time
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
except Exception:
    pass

BASE = os.getenv("VLLM_BASE_URL", "http://localhost:8003/v1")
MODEL = os.getenv("NAT_MODEL", "gemma4:e4b")


def _ping() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE}/models", timeout=5) as r:
            data = json.loads(r.read())
            ids = [m.get("id") for m in data.get("data", [])]
            print(f"[verify] /models OK — served: {ids}")
            if MODEL not in ids:
                print(f"[verify] WARNING: NAT_MODEL '{MODEL}' not in served list — "
                      f"set --served-model-name {MODEL} in 03_serve_nat.sh")
            return True
    except Exception as e:
        print(f"[verify] vLLM not reachable at {BASE}: {e}")
        return False


async def _job():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from app.llm.workers.nat_worker import run_nat_job, coerce_json
    t = time.time()
    out = await run_nat_job(
        "Extract 3-8 short topic keywords capturing what this message is about. "
        "Reply ONLY with a JSON array of strings, nothing else.",
        "I'm a bit worried about my calories today, felt like I overate at lunch with the pasta.",
        enable_thinking=False, max_tokens=80, timeout_seconds=30,
    )
    dt = (time.time() - t) * 1000
    print(f"[verify] run_nat_job round-trip: {dt:.0f}ms")
    print(f"[verify] raw reply: {out!r}")
    parsed = coerce_json(out)
    print(f"[verify] parsed JSON: {parsed!r}")
    if isinstance(parsed, list) and parsed:
        print("[verify] PASS — Nat is live and returning clean JSON.")
        return 0
    print("[verify] FAIL — Nat reachable but no usable JSON (check model/thinking config).")
    return 2


def main():
    if not _ping():
        return 1
    return asyncio.run(_job())


if __name__ == "__main__":
    raise SystemExit(main())
