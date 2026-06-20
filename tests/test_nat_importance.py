# FILE: tests/test_nat_importance.py
# Purpose: Guard Nat Job 3 (importance extraction & save): fact cleaning, pref-key
#          normalization, extraction via Nat, and the Tier-1-arbiter vs Tier-2-
#          preference routing (no commit bypass).
# Last-renovated: 2026-06-19
import asyncio
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def test_clean_facts_and_pref_key():
    from app.memory.nat_jobs.importance import _clean_facts, _pref_key
    facts = _clean_facts([
        {"type": "preference", "key": "diet", "value": "low carb"},
        {"type": "biographical", "key": "location", "value": "Portugal"},
        {"key": "", "value": "x"},            # no key -> dropped
        {"type": "decision", "key": "k", "value": ""},  # empty value -> dropped
        "not a dict",                          # dropped
    ])
    assert [f["key"] for f in facts] == ["diet", "location"]
    # Canonical key collapses ALL separators so phrasing variants dedupe to one.
    assert _pref_key("preference", "Diet Plan") == "nat_importance:preference:dietplan"
    assert _pref_key("biographical", "place_of_residence") == \
           _pref_key("biographical", "placeofresidence")


def test_clean_facts_rejects_topics_and_dedupes():
    """Over-extraction guard: opinions/musings dropped; key variants collapse."""
    from app.memory.nat_jobs.importance import _clean_facts
    facts = _clean_facts([
        {"type": "biographical", "key": "place_of_residence", "value": "Cornwall"},
        {"type": "biographical", "key": "placeofresidence", "value": "Cornwall"},  # dup
        {"type": "decision", "key": "wealth", "value": "wealth should be redistributed"},  # opinion
        {"type": "preference", "key": "automation", "value": "automation will replace jobs"},  # prediction
        {"type": "preference", "key": "diet", "value": "vegetarian"},  # real, kept
    ])
    assert [f["key"] for f in facts] == ["place_of_residence", "diet"], facts


def test_save_routing_tier1_vs_tier2(monkeypatch):
    """Canonical bio key -> arbiter.propose; everything else -> create_preference."""
    import app.self_model.canonical_schema as cs
    import app.self_model.write_arbiter as wa
    import app.astra_memory.preference_service as ps
    from app.memory.nat_jobs import importance

    proposed, prefs = [], []
    # Only "current_location" is a canonical Tier-1 field in this test.
    monkeypatch.setattr(cs, "get_field_schema",
                        lambda k: {"x": 1} if k == "current_location" else None)
    monkeypatch.setattr(wa, "propose",
                        lambda **kw: proposed.append(kw) or type("R", (), {"status": "queued"})())
    monkeypatch.setattr(ps, "create_preference",
                        lambda db, **kw: prefs.append(kw))

    saved = importance._save_facts(db=None, project_id=7, facts=[
        {"type": "biographical", "key": "current_location", "value": "Portugal"},   # -> arbiter
        {"type": "biographical", "key": "favourite_colour", "value": "blue"},        # non-canonical -> pref
        {"type": "preference", "key": "diet", "value": "low carb"},                  # -> pref
        {"type": "commitment", "key": "gym", "value": "3x/week"},                     # -> pref
    ])
    assert saved == 4
    assert len(proposed) == 1 and proposed[0]["field_name"] == "current_location"
    assert proposed[0]["proposed_value"] == "Portugal"
    assert {p["preference_key"] for p in prefs} == {
        "nat_importance:biographical:favouritecolour",
        "nat_importance:preference:diet",
        "nat_importance:commitment:gym",
    }
    # source tagging proves traceability + no bypass of the existing path
    assert all(p["source"] == "nat_importance" for p in prefs)


# --- extraction via mock vLLM ---
def _pick_port():
    for p in [7655, 9098, 8183, 8285, 6125, 5598, 8091]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p)); s.close(); return p
        except OSError:
            s.close()
    raise RuntimeError("no port")


def _sse(o):
    return ("data: " + json.dumps(o) + "\n\n").encode()


def test_extract_durable_facts_via_nat():
    facts = [{"type": "preference", "key": "diet", "value": "vegetarian"}]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            b = json.dumps({"data": [{"id": "gemma4:e4b"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            self.wfile.write(_sse({"choices": [{"index": 0, "delta": {"content": json.dumps(facts)}, "finish_reason": None}]}))
            self.wfile.write(_sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
            self.wfile.write(b"data: [DONE]\n\n")

    port = _pick_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ.update(VLLM_BASE_URL=f"http://127.0.0.1:{port}/v1", NAT_PROVIDER="vllm_local",
                      NAT_MODEL="gemma4:e4b", VLLM_API_KEY="EMPTY")

    async def _run():
        from app.memory.nat_jobs.importance import extract_durable_facts
        out = await extract_durable_facts("I'm vegetarian", "Noted — vegetarian it is.")
        assert out == [{"type": "preference", "key": "diet", "value": "vegetarian"}], out

    try:
        asyncio.run(_run())
    finally:
        srv.shutdown()


if __name__ == "__main__":
    test_clean_facts_and_pref_key()
    test_clean_facts_rejects_topics_and_dedupes()
    test_extract_durable_facts_via_nat()
    print("nat importance basic tests passed (run routing test via pytest for monkeypatch)")
