# FILE: app/self_model/capability_manifest.py
# Purpose: Generate + store ASTRA's self-description manifest from live registries.
# Called-by: main (boot), app.self_model.router (regenerate), astra_memory retrieval (fetcher)
# Depends-on: app.tools.registry, app.llm.frontier_models, app.llm.routing.handler_registry, app.astra_memory
# Last-renovated: 2026-06-12
"""
Capability manifest (memory architecture Job 4, 2026-06-12).

"What are you? What can you do? How are you different?" must retrieve a
coherent, current answer — not code fragments. This module walks the LIVE
registries (tool registry, stream-handler availability flags, frontier model
aliases, memory-tier state) and emits a structured self-description document.

Storage: data/self_model/capability_manifest.md (cold) + a high-priority hot
index row (record_type="manifest", tagged identity_capability) so identity/
capability queries surface it through the normal retrieval/injection path.

Regeneration: on boot when the registry hash changes (tools, handler flags,
model aliases). Nothing is hardcoded per-capability, so a future restricted
profile that registers fewer tools simply generates a smaller manifest.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/self_model/capability_manifest.md")
HASH_PATH = Path("data/self_model/capability_manifest.hash")

HOT_RECORD_TYPE = "manifest"
HOT_RECORD_ID = "capability_manifest"

# Where CREATE ARCHITECTURE MAP writes its scan outputs (INDEX/ARCHITECTURE_MAP).
_ARCH_DIR = Path(os.getenv("ORB_ROOT", r"D:\Orb")) / ".architecture"


# =========================================================================
# Live-source collectors (each degrades to a stub on failure)
# =========================================================================

def _collect_tools() -> List[Dict]:
    try:
        from app.tools.registry import list_tools
        return list_tools()
    except Exception as exc:
        logger.debug("[manifest] tool registry unavailable: %s", exc)
        return []


def _collect_handler_flags() -> Dict[str, bool]:
    """Stream-handler availability — the live truth of what can actually run."""
    flags: Dict[str, bool] = {}
    try:
        from app.llm.routing import handler_registry
        for name in dir(handler_registry):
            if name.endswith("_AVAILABLE"):
                value = getattr(handler_registry, name)
                if isinstance(value, bool):
                    flags[name.strip("_").replace("_AVAILABLE", "").lower()] = value
    except Exception as exc:
        logger.debug("[manifest] handler registry unavailable: %s", exc)
    return flags


def _collect_model_aliases() -> Dict[str, str]:
    try:
        from app.llm.frontier_models import all_aliases
        return dict(all_aliases())
    except Exception as exc:
        logger.debug("[manifest] frontier models unavailable: %s", exc)
        return {}


def _collect_memory_stats(db=None) -> Dict[str, object]:
    """Live memory-tier shape: hot index by type, vector count, arbiter mode."""
    stats: Dict[str, object] = {}
    close_db = False
    try:
        if db is None:
            from app.db import SessionLocal
            db = SessionLocal()
            close_db = True
        from sqlalchemy import text as _sql
        try:
            rows = db.execute(_sql(
                "SELECT record_type, COUNT(*) FROM astra_hot_index GROUP BY 1"
            )).fetchall()
            stats["hot_index"] = {r[0]: r[1] for r in rows}
        except Exception:
            stats["hot_index"] = {}
        try:
            stats["embeddings"] = db.execute(
                _sql("SELECT COUNT(*) FROM embeddings")
            ).scalar() or 0
        except Exception:
            stats["embeddings"] = 0
    except Exception as exc:
        logger.debug("[manifest] memory stats unavailable: %s", exc)
    finally:
        if close_db and db is not None:
            try:
                db.close()
            except Exception:
                pass

    try:
        from app.self_model.write_arbiter import get_arbiter_mode
        stats["arbiter_mode"] = str(get_arbiter_mode())
    except Exception:
        stats["arbiter_mode"] = "enforce (configured)"
    stats["decay_scheduler"] = os.getenv("ASTRA_DECAY_SCHEDULER_ENABLED", "1") != "0"
    return stats


def _latest_index_path() -> Optional[Path]:
    """Newest INDEX_<ts>.json, else legacy INDEX.json, from the last scan."""
    try:
        candidates = sorted(_ARCH_DIR.glob("INDEX_*.json"), reverse=True)
        if candidates:
            return candidates[0]
        legacy = _ARCH_DIR / "INDEX.json"
        return legacy if legacy.exists() else None
    except Exception:
        return None


def _collect_architecture() -> Dict[str, object]:
    """Subsystem breakdown derived from the last full CREATE ARCHITECTURE MAP
    scan, so self-description reflects the real, current codebase instead of a
    frozen list. Degrades to {} when no scan output is present."""
    info: Dict[str, object] = {}
    idx_path = _latest_index_path()
    if not idx_path:
        return info
    try:
        from collections import Counter

        with open(idx_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Both index formats: RAG ("scanned_files") and legacy ("files").
        files = data.get("scanned_files") or data.get("files") or []
        if not files:
            return info

        subsystems: "Counter[str]" = Counter()
        for entry in files:
            path = (entry.get("path") or "").replace("\\", "/")
            i = path.find("/app/")
            if i == -1:
                continue
            parts = path[i + len("/app/"):].split("/")
            if len(parts) >= 2:  # app/<subsystem>/<file...>
                subsystems[parts[0]] += 1

        scan_date = data.get("timestamp") or ""
        if not scan_date:
            try:
                scan_date = datetime.fromtimestamp(
                    idx_path.stat().st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%d")
            except Exception:
                scan_date = "unknown"

        info = {
            "scan_date": scan_date,
            "index_file": idx_path.name,
            "total_files": len(files),
            "subsystems": subsystems.most_common(40),
            "has_map": (_ARCH_DIR / "ARCHITECTURE_MAP.md").exists(),
        }
    except Exception as exc:
        logger.debug("[manifest] architecture collect failed: %s", exc)
    return info


# =========================================================================
# Hash — regeneration trigger
# =========================================================================

def registry_hash(db=None) -> str:
    """Stable hash over everything the manifest is generated from."""
    basis = {
        "tools": [(t.get("name"), t.get("version"), t.get("description"))
                  for t in _collect_tools()],
        "handlers": _collect_handler_flags(),
        "models": _collect_model_aliases(),
    }
    blob = json.dumps(basis, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# =========================================================================
# Generator
# =========================================================================

def generate_manifest(db=None) -> str:
    """Emit the markdown self-description from live registry data."""
    tools = _collect_tools()
    handlers = _collect_handler_flags()
    models = _collect_model_aliases()
    memory = _collect_memory_stats(db)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: List[str] = []
    add = lines.append

    add("# ASTRA — Capability Manifest")
    add(f"_Generated {now} from live registries. "
        "Regenerates automatically when capabilities change._")
    add("")

    # ── Identity ──────────────────────────────────────────────────
    add("## Identity")
    add("ASTRA is Taz's personal AI system: a local-first FastAPI backend on "
        "his own machine, with a desktop app (Electron/React), an Android "
        "bridge app, and a voice loop (Chatterbox TTS). It is one persistent "
        "system with memory, schedulers and tools — not a stateless chat "
        "window. A guarded Tier-1 identity store holds durable user facts "
        "(every write passes a write arbiter in "
        f"{memory.get('arbiter_mode', 'enforce')} mode).")
    add("")

    # ── Memory architecture ───────────────────────────────────────
    add("## Memory architecture")
    hot = memory.get("hot_index") or {}
    hot_total = sum(hot.values()) if isinstance(hot, dict) else 0
    add("Tiered retrieval, not a context window:")
    add("- **Tier 1 identity store** — durable structured facts, always "
        "injected, write-arbitered.")
    add(f"- **Hot index** — {hot_total} records "
        f"({', '.join(f'{k}: {v}' for k, v in sorted(hot.items())) or 'building'}) "
        "with tags/entities from an 11-domain topic tagger; two-stage "
        "retrieval with depth gating (D0–D4).")
    add(f"- **Semantic channel** — {memory.get('embeddings', 0)} Gemini "
        "Embedding 2 vectors (1536-d, single multimodal vector space) merged "
        "into candidate selection.")
    add("- **Summary pyramids + conversation summaries** — long histories "
        "compressed, injected provider-agnostically.")
    add("- **Enrichment layer** — images get vision captions, documents get "
        "abstracts, music gets metadata tags, so media is conversationally "
        "retrievable with file paths.")
    add("- **Repo-scan memory** — ASTRA's own code is embedded and "
        "retrievable in chat with honest staleness labels (host vs sandbox "
        "snapshot dates).")
    add("- **Decay scheduler** — "
        + ("active" if memory.get("decay_scheduler") else "disabled")
        + "; reinforcement on retrieval keeps used records alive.")
    add("")

    # ── Autonomous pipeline ───────────────────────────────────────
    add("## Autonomous build pipeline")
    add("ASTRA can modify itself: a staged pipeline runs Weaver (intent → "
        "build plan) → SpecGate (spec validation) → Scaffold → Agentic "
        "Builder → BVL/Verifier (three-tier verification cascade), executing "
        "inside an isolated sandbox VM (ZOBIE-ORB) so the running system is "
        "never edited live.")
    if handlers:
        live = sorted(k for k, v in handlers.items() if v)
        dark = sorted(k for k, v in handlers.items() if not v)
        add(f"Live stream handlers right now ({len(live)}): "
            + ", ".join(live) + ".")
        if dark:
            add(f"Currently unavailable: {', '.join(dark)}.")
    add("")

    # ── Tools ─────────────────────────────────────────────────────
    add("## Tools")
    if tools:
        add(f"{len(tools)} registered tools (live registry):")
        for t in tools:
            desc = (t.get("description") or "").strip().split("\n")[0][:110]
            add(f"- `{t.get('name')}` — {desc}")
    else:
        add("Tool registry empty or unavailable at generation time.")
    add("")

    # ── Providers & models ────────────────────────────────────────
    add("## Providers & models")
    if models:
        add("Multi-model orchestration — frontier aliases in use:")
        for alias, concrete in sorted(models.items()):
            add(f"- `{alias}` → {concrete}")
    add("A deterministic tier router picks the cheapest adequate model per "
        "turn and escalates for architectural depth; high-stakes jobs get "
        "multi-model critique.")
    add("")

    # ── Architecture (from last full scan) ─────────────────────────
    arch = _collect_architecture()
    subsystems = arch.get("subsystems") if arch else None
    if subsystems:
        add("## Architecture map (last full scan)")
        add(f"From the most recent CREATE ARCHITECTURE MAP scan "
            f"({arch.get('scan_date', 'unknown')}; "
            f"{arch.get('total_files', 0)} files indexed). Backend subsystems "
            "under `app/`, by file count — this is ASTRA's actual code surface, "
            "not a curated list:")
        for name, count in subsystems[:30]:
            add(f"- **{name}** — {count} files")
        if arch.get("has_map"):
            add("A full prose architecture map lives at "
                "`.architecture/ARCHITECTURE_MAP.md` and is retrievable in chat.")
        add("")

    # ── Differentiators ───────────────────────────────────────────
    add("## How ASTRA differs from a stock assistant")
    add("- **Tiered RAG memory** that persists and decays like memory should, "
        "instead of a single context window.")
    add("- **Personalisation over time** — preferences with confidence "
        "scoring and evidence, identity facts behind a write arbiter, "
        "pattern learning from experience.")
    add("- **Multi-model orchestration** — GPT, Claude, Gemini and local "
        "models share one conversation context; switching mid-conversation "
        "keeps continuity.")
    add("- **Self-building** — it ships changes to itself through its own "
        "sandboxed pipeline with verification gates.")
    add("- **Embodied at home** — voice, phone bridge, vehicle telemetry, "
        "network sentinel, finance/lifestyle domains, all feeding one memory.")

    return "\n".join(lines)


# =========================================================================
# Store + regenerate
# =========================================================================

def _hot_row_exists(db) -> bool:
    close_db = False
    try:
        if db is None:
            from app.db import SessionLocal
            db = SessionLocal()
            close_db = True
        from app.astra_memory.preference_models import HotIndex
        return db.query(HotIndex).filter(
            HotIndex.record_type == HOT_RECORD_TYPE,
            HotIndex.record_id == HOT_RECORD_ID,
        ).first() is not None
    except Exception:
        return False
    finally:
        if close_db and db is not None:
            try:
                db.close()
            except Exception:
                pass


def _upsert_hot_row(db, manifest_md: str) -> None:
    from app.astra_memory.retrieval import upsert_hot_index
    from app.astra_memory.preference_models import RetrievalCost

    tool_count = len(_collect_tools())
    one_liner = (
        "ASTRA self-description: local-first personal AI with tiered RAG "
        f"memory, autonomous sandboxed build pipeline, {tool_count} tools, "
        "multi-model orchestration. Full manifest available."
    )
    bullets = [
        "Tiered memory: Tier-1 identity (arbitered), hot index, semantic vectors, summaries, decay",
        "Autonomous pipeline: Weaver → SpecGate → Scaffold → Agentic Builder → BVL/Verifier (sandboxed)",
        f"{tool_count} live registered tools; multi-model orchestration with shared context",
        "Media enrichment: image captions, document abstracts, music metadata",
        "Repo-scan self-knowledge with staleness labels",
    ]
    upsert_hot_index(
        db=db,
        record_type=HOT_RECORD_TYPE,
        record_id=HOT_RECORD_ID,
        title="ASTRA Capability Manifest",
        one_liner=one_liner,
        bullets_5=bullets,
        tags=["identity_capability", "astra_dev", "architecture"],
        entities=["ASTRA"],
        cold_storage_path=str(MANIFEST_PATH),
        retrieval_priority=0.9,
        retrieval_cost=RetrievalCost.TINY,
    )


def regenerate_if_stale(db=None, force: bool = False) -> bool:
    """Regenerate manifest when the registry hash changed. Returns True if
    regenerated. Never raises (boot-safe)."""
    try:
        current = registry_hash(db)
        stored = None
        try:
            if HASH_PATH.exists():
                stored = HASH_PATH.read_text(encoding="utf-8").strip()
        except Exception:
            stored = None

        if not force and stored == current and MANIFEST_PATH.exists():
            # Hash + file are filesystem-global, but the hot row lives in
            # THIS db — an ephemeral boot may have written the hash while a
            # different (live) DB still lacks its row. Ensure it exists
            # before skipping.
            if _hot_row_exists(db):
                logger.debug("[manifest] up to date (hash %s…)", current[:12])
                return False
            manifest_md = MANIFEST_PATH.read_text(encoding="utf-8")
            close_db = False
            if db is None:
                from app.db import SessionLocal
                db = SessionLocal()
                close_db = True
            try:
                _upsert_hot_row(db, manifest_md)
            finally:
                if close_db:
                    db.close()
            logger.info("[manifest] hot row restored for this DB (content unchanged)")
            return False

        manifest_md = generate_manifest(db)
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(manifest_md, encoding="utf-8")
        HASH_PATH.write_text(current, encoding="utf-8")

        close_db = False
        if db is None:
            from app.db import SessionLocal
            db = SessionLocal()
            close_db = True
        try:
            _upsert_hot_row(db, manifest_md)
        finally:
            if close_db:
                db.close()

        logger.info("[manifest] regenerated (%d chars, hash %s…)",
                    len(manifest_md), current[:12])
        return True
    except Exception as exc:
        logger.warning("[manifest] regeneration failed (non-fatal): %s", exc)
        return False


def read_manifest() -> Optional[str]:
    """Cold-storage read used by the retrieval fetcher."""
    try:
        if MANIFEST_PATH.exists():
            return MANIFEST_PATH.read_text(encoding="utf-8")
    except Exception as exc:
        logger.debug("[manifest] read failed: %s", exc)
    return None


__all__ = [
    "generate_manifest",
    "registry_hash",
    "regenerate_if_stale",
    "read_manifest",
    "MANIFEST_PATH",
    "HASH_PATH",
    "HOT_RECORD_TYPE",
    "HOT_RECORD_ID",
]
