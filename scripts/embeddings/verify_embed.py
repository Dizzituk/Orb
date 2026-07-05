# FILE: scripts/embeddings/verify_embed.py
# Purpose: Smoke-verify the local embedding server (:8004) — dims, instruction asymmetry, latency.
# Called-by: manual (Windows: .venv\Scripts\python.exe scripts\embeddings\verify_embed.py)
# Depends-on: httpx only (runs OUTSIDE the backend; no app imports)
# Last-renovated: 2026-07-02 (LANE E)
"""Confirms the Qwen3 embedding server is healthy and reports its ACTUAL
output dimensionality (stamp truth — set EMBEDDINGS_TEXT_DIMS to match) plus
warm p50 latency for single-query embeds."""
from __future__ import annotations

import os
import statistics
import sys
import time

import httpx

BASE = os.getenv("EMBEDDINGS_TEXT_BASE_URL", "http://127.0.0.1:8004/v1").rstrip("/")
MODEL = os.getenv("EMBEDDINGS_TEXT_MODEL", "qwen3-embedding-0.6b")

QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery: "
)


def embed(texts):
    r = httpx.post(
        f"{BASE}/embeddings",
        json={"model": MODEL, "input": texts},
        timeout=httpx.Timeout(30.0, connect=3.0),
    )
    r.raise_for_status()
    data = r.json()["data"]
    return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def main() -> int:
    try:
        models = httpx.get(f"{BASE}/models", timeout=5.0)
        models.raise_for_status()
    except Exception as e:
        print(f"[verify_embed] FAIL: server not reachable at {BASE}: {e}")
        return 1
    served = [m["id"] for m in models.json().get("data", [])]
    print(f"[verify_embed] server up; served models: {served}")
    if MODEL not in served:
        print(f"[verify_embed] FAIL: '{MODEL}' not in served models — "
              f"EMBED_SERVED_NAME and EMBEDDINGS_TEXT_MODEL must match")
        return 1

    doc = "The 4080 carries Nat, Chatterbox TTS and the text embedder."
    query = "what runs on the 4080 GPU"
    vecs = embed([doc, QUERY_PREFIX + query, query])
    dims = {len(v) for v in vecs}
    print(f"[verify_embed] output dims: {sorted(dims)} — set EMBEDDINGS_TEXT_DIMS accordingly")
    if len(dims) != 1:
        print("[verify_embed] FAIL: inconsistent dims across calls")
        return 1

    sim_instructed = cos(vecs[0], vecs[1])
    sim_plain = cos(vecs[0], vecs[2])
    print(f"[verify_embed] doc~query similarity: instructed={sim_instructed:.4f} "
          f"plain={sim_plain:.4f} (both should be clearly > 0)")

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        embed([QUERY_PREFIX + query])
        times.append((time.perf_counter() - t0) * 1000)
    print(f"[verify_embed] warm single-embed latency: p50={statistics.median(times):.0f}ms "
          f"min={min(times):.0f}ms max={max(times):.0f}ms")
    print("[verify_embed] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
