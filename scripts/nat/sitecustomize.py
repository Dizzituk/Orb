# sitecustomize.py — auto-imported by Python at interpreter startup whenever this
# directory is on PYTHONPATH. Applies a runtime monkeypatch (no files modified)
# so prometheus_fastapi_instrumentator tolerates starlette 1.x.
#
# Why: vLLM 0.22.1 pulls starlette 1.x, whose router objects (_IncludedRouter)
# have no `.path`. prometheus_fastapi_instrumentator.get_route_name assumes every
# route has `.path`, so it raises AttributeError on EVERY request -> 500s all
# endpoints (incl. /v1/chat/completions). We wrap it to return None on failure.
#
# Why sitecustomize (not a launcher): vLLM spawns its EngineCore via Python 3.14
# multiprocessing "spawn", which re-imports the entry module. A runpy launcher
# trips spawn's freeze_support guard. sitecustomize runs in EVERY process at
# startup without touching the entry point, so spawn is unaffected and the patch
# is present in the HTTP-serving process.
try:
    import prometheus_fastapi_instrumentator.routing as _pr

    _orig_get_route_name = _pr.get_route_name

    def _safe_get_route_name(request):
        try:
            return _orig_get_route_name(request)
        except Exception:
            return None

    if getattr(_pr.get_route_name, "__name__", "") != "_safe_get_route_name":
        _pr.get_route_name = _safe_get_route_name
        print("[nat sitecustomize] prometheus get_route_name patched for starlette 1.x")
except Exception:
    # prometheus not importable in this process — nothing to patch.
    pass
