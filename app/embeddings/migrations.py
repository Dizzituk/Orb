# FILE: app/embeddings/migrations.py
# Purpose: Startup schema migration — model/dims stamp columns on embeddings + rag_entries vectors (corruption-tolerant).
# Called-by: app.db.init_db
# Depends-on: sqlalchemy engine only
# Last-renovated: 2026-07-02 (LANE E; hotfix same day: chunked backfill, never kills boot)
"""
LANE E (2026-07-02): every stored vector must carry its producing model and
dimensionality so mixed-model cosine is structurally impossible — search
filters rows to the active model-space BEFORE scoring
(app/embeddings/service.py).

HOTFIX (2026-07-02 evening, live incident): the embeddings table carries
localised btree damage (diagnosed 2026-06-10 — see semantic_candidates.py's
chunked loader). A single full-table backfill UPDATE walks the damaged pages
and dies with "database disk image is malformed", which killed init_db and
with it the whole backend boot. The backfill is now:
  - chunked into id windows, each its own transaction;
  - per-window failures are rolled back, skipped and counted;
  - the whole migration is exception-proof — a stamp backfill must NEVER
    cost a boot. Rows left unstamped (NULL) are treated as legacy gemini
    rows by every read path and picked up per-row by reembed_batch.

Legacy rows are backfilled as gemini-embedding-2-preview / 1536 — the only
model ever used in production. Idempotent: re-runs are no-ops.
"""
from __future__ import annotations

import logging

from sqlalchemy import text as _sql

logger = logging.getLogger(__name__)

LEGACY_MODEL = "gemini-embedding-2-preview"
LEGACY_DIMS = 1536

_WINDOW = 500  # id-window size, mirrors the semantic loader's damage stride


def _columns(conn, table: str) -> set:
    try:
        rows = conn.execute(_sql(f"PRAGMA table_info({table})")).fetchall()
        return {r[1] for r in rows}
    except Exception:
        return set()


def _add_columns(engine, table: str, ddl: list) -> bool:
    """ALTERs commit implicitly on pysqlite; run them individually and treat
    'already exists' as done."""
    added = False
    for statement in ddl:
        try:
            with engine.begin() as conn:
                conn.execute(_sql(statement))
            added = True
        except Exception as exc:
            if "duplicate column" not in str(exc).lower():
                logger.warning("[embeddings.migrations] %s: %s", statement[:60], exc)
    return added


def _chunked_backfill(engine, table: str, set_clause: str, where_null: str) -> tuple:
    """UPDATE in id windows; skip windows that hit damaged pages. Returns
    (updated, skipped_windows)."""
    try:
        with engine.connect() as conn:
            max_id = conn.execute(_sql(f"SELECT MAX(id) FROM {table}")).scalar() or 0
    except Exception as exc:
        logger.error("[embeddings.migrations] %s MAX(id) failed: %s", table, exc)
        return 0, -1

    updated = 0
    skipped = 0
    lo = 0
    while lo <= max_id:
        try:
            with engine.begin() as conn:
                result = conn.execute(_sql(
                    f"UPDATE {table} SET {set_clause} "
                    f"WHERE id > :lo AND id <= :hi AND {where_null}"
                ), {"m": LEGACY_MODEL, "d": LEGACY_DIMS, "lo": lo, "hi": lo + _WINDOW})
                updated += result.rowcount or 0
        except Exception:
            # Damaged page range — leave these rows NULL-stamped; read paths
            # treat NULL as legacy gemini and reembed_batch retries per-row.
            skipped += 1
        lo += _WINDOW
    return updated, skipped


def migrate_embeddings_schema(engine) -> None:
    """Add embedding_model/dims columns and backfill legacy rows. Runs at
    every boot (init_db); cheap once applied; NEVER raises."""
    try:
        # ── embeddings table ──────────────────────────────────────
        with engine.connect() as conn:
            cols = _columns(conn, "embeddings")
        if cols:
            ddl = []
            if "embedding_model" not in cols:
                ddl.append("ALTER TABLE embeddings ADD COLUMN embedding_model VARCHAR(100)")
            if "dims" not in cols:
                ddl.append("ALTER TABLE embeddings ADD COLUMN dims INTEGER")
            added = _add_columns(engine, "embeddings", ddl) if ddl else False

            updated, skipped = _chunked_backfill(
                engine, "embeddings",
                "embedding_model = :m, dims = :d",
                "embedding_model IS NULL",
            )
            try:
                with engine.begin() as conn:
                    conn.execute(_sql(
                        "CREATE INDEX IF NOT EXISTS ix_embeddings_model "
                        "ON embeddings (embedding_model)"
                    ))
            except Exception as exc:
                logger.warning("[embeddings.migrations] index create failed: %s", exc)
            if added or updated or skipped:
                logger.warning(
                    "[embeddings.migrations] embeddings stamped: columns_added=%s "
                    "backfilled=%d skipped_windows=%d (%s/%d) — skipped windows "
                    "stay NULL and read as legacy gemini",
                    added, updated, skipped, LEGACY_MODEL, LEGACY_DIMS,
                )

        # ── rag_entries table (video-asset vectors) ───────────────
        with engine.connect() as conn:
            cols = _columns(conn, "rag_entries")
        if cols:
            ddl = []
            if "embedding_model" not in cols:
                ddl.append("ALTER TABLE rag_entries ADD COLUMN embedding_model VARCHAR(100)")
            if "embedding_dims" not in cols:
                ddl.append("ALTER TABLE rag_entries ADD COLUMN embedding_dims INTEGER")
            added = _add_columns(engine, "rag_entries", ddl) if ddl else False

            updated, skipped = _chunked_backfill(
                engine, "rag_entries",
                "embedding_model = :m, embedding_dims = :d",
                "embedding IS NOT NULL AND embedding_model IS NULL",
            )
            if added or updated or skipped:
                logger.warning(
                    "[embeddings.migrations] rag_entries stamped: columns_added=%s "
                    "backfilled=%d skipped_windows=%d", added, updated, skipped,
                )
    except Exception as exc:
        # A stamp backfill must never cost a boot — NULL rows are handled by
        # every read path. Log CRITICAL so it can't hide.
        logger.critical(
            "[embeddings.migrations] migration failed (boot continues; "
            "unstamped rows read as legacy gemini): %s", exc,
        )
