# FILE: tests/test_nat_vllm_worker.py
# Purpose: Guard the Nat / vLLM worker contract end-to-end against a mock
#          OpenAI-compatible server (vLLM speaks the OpenAI API). No GPU/vLLM
#          needed — proves transport, token-key contract, thinking control,
#          and the agentic tool loop + sequentialthinking.
# Last-renovated: 2026-06-19
import asyncio
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _pick_port():
    for p in [7654, 9099, 8181, 8282, 8484, 6123, 5599, 8090, 8099, 7777]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p)); s.close(); return p
        except OSError:
            s.close()
    raise RuntimeError("no bindable port for mock vLLM")


def _sse(obj):
    return ("data: " + json.dumps(obj) + "\n\n").encode()


def _make_handler(captured):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.endswith("/models"):
                b = json.dumps({"object": "list", "data": [{"id": "gemma4:e4b"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            captured["bodies"].append(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            msgs = body.get("messages", [])
            if any(m.get("role") == "tool" for m in msgs):
                self.wfile.write(_sse({"choices": [{"index": 0, "delta": {"content": "FINAL: 42"}, "finish_reason": None}]}))
                self.wfile.write(_sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
            elif body.get("tools"):
                self.wfile.write(_sse({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "sequentialthinking", "arguments": ""}}]}}]}))
                self.wfile.write(_sse({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"thought\":\"reason\",\"thoughtNumber\":1,"}}]}}]}))
                self.wfile.write(_sse({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"nextThoughtNeeded\":false}"}}]}}]}))
                self.wfile.write(_sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}))
            else:
                for tok in ['["calorie ', 'tracking",', ' "pasta"]']:
                    self.wfile.write(_sse({"choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}]}))
                self.wfile.write(_sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
            self.wfile.write(_sse({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}}))
            self.wfile.write(b"data: [DONE]\n\n")
    return H


def test_nat_vllm_text_and_tool_loop():
    captured = {"bodies": []}
    port = _pick_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(captured))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ.update(
        VLLM_BASE_URL=f"http://127.0.0.1:{port}/v1",
        NAT_PROVIDER="vllm_local", NAT_MODEL="gemma4:e4b", VLLM_API_KEY="EMPTY",
    )

    async def _run():
        from app.llm.workers.nat_worker import run_nat_job, coerce_json
        # 1) Text job — token-key contract + thinking OFF by default.
        txt = await run_nat_job("Extract keywords, JSON only.", "calories pasta",
                                enable_thinking=False, max_tokens=80, timeout_seconds=10)
        assert txt == '["calorie tracking", "pasta"]', txt
        assert coerce_json(txt) == ["calorie tracking", "pasta"]
        assert captured["bodies"][-1].get("chat_template_kwargs") == {"enable_thinking": False}

        # 2) Agentic tool loop + sequentialthinking + thinking ON.
        res = await run_nat_job("Think then answer.", "6*7?",
                                tools=[{"name": "noop", "description": "x",
                                        "parameters": {"type": "object", "properties": {}}}],
                                enable_thinking=True, max_tokens=120, timeout_seconds=10)
        assert isinstance(res, dict)
        assert res["text"] == "FINAL: 42"
        assert res["rounds"] == 2
        assert any(c["name"] == "sequentialthinking" for c in res["tool_calls"])
        assert res["thoughts"][0]["thought"] == "reason"
        assert captured["bodies"][-1].get("chat_template_kwargs") == {"enable_thinking": True}

    try:
        asyncio.run(_run())
    finally:
        srv.shutdown()


def test_nat_worker_failure_proof_when_down():
    """Worker must return empty (never raise) when Nat is unreachable."""
    os.environ.update(VLLM_BASE_URL="http://127.0.0.1:1/v1", NAT_PROVIDER="vllm_local",
                      NAT_MODEL="gemma4:e4b")

    async def _run():
        from app.llm.workers.nat_worker import run_nat_job
        out = await run_nat_job("sys", "hi", timeout_seconds=2)
        assert out == ""  # graceful empty, not an exception
        out2 = await run_nat_job("sys", "hi", tools=[{"name": "noop", "parameters": {"type": "object", "properties": {}}}], timeout_seconds=2)
        assert isinstance(out2, dict)  # dict (possibly with stopped=error)

    asyncio.run(_run())


if __name__ == "__main__":
    test_nat_vllm_text_and_tool_loop()
    test_nat_worker_failure_proof_when_down()
    print("nat vllm worker tests passed")
