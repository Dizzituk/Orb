# nat_server_launcher.py — Start vLLM's OpenAI server with a small RUNTIME
# monkeypatch (no files modified) that makes prometheus_fastapi_instrumentator
# tolerate starlette 1.x.
#
# Why the patch: vLLM 0.22.1 pulls starlette 1.x, whose router objects
# (_IncludedRouter) have no `.path`. prometheus_fastapi_instrumentator's
# get_route_name assumes every route has `.path`, so it raises AttributeError on
# EVERY request (500s all endpoints, incl. /v1/chat/completions). We wrap it to
# return None on failure (metrics just lose that route name; requests work).
#
# Why force fork: vLLM forks its EngineCore subprocess. Python 3.14 changed the
# default multiprocessing start method away from "fork" to "spawn"/"forkserver",
# and spawn re-imports __main__ — which, because we launch via runpy, trips
# multiprocessing's freeze_support guard and kills engine init. Forcing "fork"
# restores vLLM's historical behaviour (the parent hasn't touched CUDA yet, so
# fork is safe) and makes this patch inherit into children for free.
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

import runpy  # noqa: E402
import sys  # noqa: E402

try:
    import multiprocessing as _mp
    _mp.set_start_method("fork", force=True)
except Exception as _e:  # noqa: BLE001
    print(f"[nat_launcher] could not force fork start method: {_e}")

try:
    import prometheus_fastapi_instrumentator.routing as _pr

    _orig_get_route_name = _pr.get_route_name

    def _safe_get_route_name(request):
        try:
            return _orig_get_route_name(request)
        except Exception:
            return None

    _pr.get_route_name = _safe_get_route_name
    print("[nat_launcher] prometheus get_route_name patched for starlette 1.x")
except Exception as _e:  # noqa: BLE001 — patch is best-effort
    print(f"[nat_launcher] prometheus patch skipped: {_e}")

# Forward our CLI args to vLLM's OpenAI API server and run it as __main__.
sys.argv = ["vllm.entrypoints.openai.api_server"] + sys.argv[1:]
runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
