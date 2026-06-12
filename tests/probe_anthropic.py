# Purpose: Minimal probe: what does the Anthropic API actually say? One tiny call
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""Minimal probe: what does the Anthropic API actually say? One tiny call
per variant, full exception text printed."""
import sys, os, asyncio
os.chdir("D:/Orb")
sys.path.insert(0, "D:/Orb")
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

def load_env(path="D:/Orb/.env"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
    except Exception as e:
        print("env load failed:", e)

load_env()
# The live key lives in the encrypted store (master key only in the running
# backend's memory - not extracting that). The pre-removal .env backup holds
# a plaintext key; use it for this probe only, never printed.
if not os.environ.get("ANTHROPIC_API_KEY"):
    load_env("D:/Orb/.env.backup-before-key-removal-20260414-221211")

key = os.environ.get("ANTHROPIC_API_KEY", "")
print("key present:", bool(key), "len:", len(key))

from anthropic import AsyncAnthropic

async def probe(model, thinking=None, tools=None):
    client = AsyncAnthropic(api_key=key, timeout=60.0)
    kwargs = dict(model=model, max_tokens=2048,
                  messages=[{"role": "user", "content": "Reply with exactly: OK"}])
    if thinking:
        kwargs["thinking"] = thinking
    if tools:
        kwargs["tools"] = tools
    label = f"{model} thinking={bool(thinking)} tools={bool(tools)}"
    try:
        resp = await client.messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        print(f"PASS  {label} -> '{text[:40]}' stop={resp.stop_reason} usage={resp.usage.input_tokens}/{resp.usage.output_tokens}")
        return True
    except Exception as e:
        print(f"FAIL  {label} -> {type(e).__name__}: {str(e)[:400]}")
        return False

TOOLS = [{"name": "ping", "description": "test tool", "input_schema": {"type": "object", "properties": {}, "required": []}}]

async def main():
    ok = await probe("claude-fable-5")
    await probe("claude-fable-5", thinking={"type": "enabled", "budget_tokens": 1024})
    await probe("claude-fable-5", tools=TOOLS)
    if not ok:
        await probe("claude-opus-4-8")
        await probe("claude-opus-4-6")

asyncio.run(main())
