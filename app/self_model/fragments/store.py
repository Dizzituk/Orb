# FILE: app/self_model/fragments/store.py
# Purpose: Fragment store — append-only JSON-backed persistence.
# Called-by: app.self_model.fragments.capture, app.self_model.fragments.cluster, app.self_model.fragments.injection, app.self_model.fragments.router (+1 more)
# Depends-on: app.self_model.fragments.models
# Last-renovated: 2026-06-11
"""
Fragment store — append-only JSON-backed persistence.

Responsibilities:
  - Persist fragments to data/self_model/fragments.json
  - Persist themes (emergent clusters) to themes.json
  - Provide read access (all fragments, by theme, by id)
  - Provide atomic append of new fragments

Never deletes. Dismissal and decay are metadata changes, not removal.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from app.self_model.fragments.models import Fragment

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "self_model"
_FRAGMENTS_FILE = _DATA_DIR / "fragments.json"
_THEMES_FILE    = _DATA_DIR / "themes.json"

_lock = Lock()


class FragmentStore:
    def __init__(self) -> None:
        self._fragments: Dict[str, Fragment] = {}
        self._themes: Dict[str, Dict[str, Any]] = {}
        self._ensure_storage()
        self._load()

    def _ensure_storage(self) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not _FRAGMENTS_FILE.exists():
            _FRAGMENTS_FILE.write_text(
                json.dumps({"fragments": []}, indent=2), encoding="utf-8",
            )
        if not _THEMES_FILE.exists():
            _THEMES_FILE.write_text(
                json.dumps({"themes": {}}, indent=2), encoding="utf-8",
            )

    def _load(self) -> None:
        try:
            raw = json.loads(_FRAGMENTS_FILE.read_text(encoding="utf-8"))
            for item in raw.get("fragments", []):
                f = Fragment.from_dict(item)
                self._fragments[f.fragment_id] = f
        except Exception as exc:
            logger.error("[fragment_store] load fragments failed: %s", exc)
        try:
            traw = json.loads(_THEMES_FILE.read_text(encoding="utf-8"))
            self._themes = traw.get("themes", {})
        except Exception as exc:
            logger.error("[fragment_store] load themes failed: %s", exc)

    def _persist_fragments(self) -> None:
        payload = {
            "fragments": [f.to_dict() for f in self._fragments.values()],
        }
        _FRAGMENTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _persist_themes(self) -> None:
        payload = {"themes": self._themes}
        _THEMES_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ─── write path ───────────────────────────────────────────

    def append(self, fragment: Fragment) -> Fragment:
        """Atomic append. Never overwrites an existing fragment_id."""
        with _lock:
            if fragment.fragment_id in self._fragments:
                logger.warning("[fragment_store] duplicate id: %s", fragment.fragment_id)
                return self._fragments[fragment.fragment_id]
            self._fragments[fragment.fragment_id] = fragment
            self._persist_fragments()
            logger.info(
                "[fragment_store] appended fragment %s type=%s signal=%s text=%r",
                fragment.fragment_id, fragment.claim_type, fragment.signal_type,
                fragment.text[:60],
            )
            return fragment

    def mark_surfaced(self, fragment_id: str, when: str) -> bool:
        with _lock:
            f = self._fragments.get(fragment_id)
            if not f:
                return False
            f.last_surfaced_at = when
            self._persist_fragments()
            return True

    def mark_dismissed(self, fragment_id: str) -> bool:
        with _lock:
            f = self._fragments.get(fragment_id)
            if not f:
                return False
            f.dismissed = True
            self._persist_fragments()
            return True

    def assign_theme(self, fragment_id: str, theme_id: Optional[str]) -> bool:
        with _lock:
            f = self._fragments.get(fragment_id)
            if not f:
                return False
            f.theme_id = theme_id
            self._persist_fragments()
            return True

    def update_embedding(self, fragment_id: str, embedding, embedding_model: str) -> bool:
        """LANE E: restamp a fragment's vector (reembed_batch migration)."""
        with _lock:
            f = self._fragments.get(fragment_id)
            if not f:
                return False
            f.embedding = list(embedding) if embedding else None
            f.embedding_model = embedding_model if embedding else ""
            self._persist_fragments()
            return True

    # ─── read path ────────────────────────────────────────────

    def get(self, fragment_id: str) -> Optional[Fragment]:
        return self._fragments.get(fragment_id)

    def all(self) -> List[Fragment]:
        return list(self._fragments.values())

    def by_theme(self, theme_id: str) -> List[Fragment]:
        return [f for f in self._fragments.values() if f.theme_id == theme_id]

    def with_embeddings(self) -> List[Fragment]:
        return [f for f in self._fragments.values() if f.embedding]

    # ─── theme persistence (theme metadata lives separately) ──

    def save_theme(self, theme_id: str, data: Dict[str, Any]) -> None:
        with _lock:
            self._themes[theme_id] = data
            self._persist_themes()

    def get_theme(self, theme_id: str) -> Optional[Dict[str, Any]]:
        return self._themes.get(theme_id)

    def all_themes(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._themes)

    def delete_theme(self, theme_id: str) -> bool:
        with _lock:
            if theme_id not in self._themes:
                return False
            del self._themes[theme_id]
            # Unlink fragments from this theme (fragments themselves are kept)
            for f in self._fragments.values():
                if f.theme_id == theme_id:
                    f.theme_id = None
            self._persist_themes()
            self._persist_fragments()
            return True

    def summary(self) -> Dict[str, Any]:
        with_emb = sum(1 for f in self._fragments.values() if f.embedding)
        orphans  = sum(1 for f in self._fragments.values() if f.theme_id is None)
        return {
            "total_fragments": len(self._fragments),
            "with_embeddings": with_emb,
            "orphan_fragments": orphans,
            "total_themes": len(self._themes),
            "fragments_path": str(_FRAGMENTS_FILE),
            "themes_path": str(_THEMES_FILE),
        }


_store: Optional[FragmentStore] = None


def get_fragment_store() -> FragmentStore:
    global _store
    if _store is None:
        _store = FragmentStore()
    return _store