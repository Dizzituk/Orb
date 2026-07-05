# FILE: tests/test_web_automation_seed.py
# Purpose: Tests for app.web_automation.seed + migrations (shared-partition seeding).
# Called-by: pytest
# Depends-on: app.db, app.web_automation
# Last-renovated: 2026-07-01
"""
Regression tests for the 2026-07-01 shared-partition seed fix.

Five DEFAULT_SESSIONS share partition "persist:media" by design. The old
model declared WebSession.partition unique=True, so on any schema-fresh DB
the second persist:media insert raised IntegrityError and seed_sessions()
aborted at 8 of 12 rows (live DB included). Covers:
  - fresh seed creates every definition (shared partitions included)
  - idempotency (second run creates nothing)
  - per-row guard (one poisoned row cannot strand the rows after it)
  - legacy-DB rebuild migration (constraint dropped, rows + actions kept)
  - migration no-op on already-fixed schemas
"""
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.web_automation.migrations import migrate_web_sessions_schema
from app.web_automation.models import WebAction, WebSession
from app.web_automation.seed import DEFAULT_SESSIONS, seed_sessions
from app.web_automation import session_registry

_TABLES = [WebSession.__table__, WebAction.__table__]

# The exact DDL SQLAlchemy rendered for the pre-fix model (partition
# unique=True) — what the live DB carried until the 2026-07-01 rebuild.
_LEGACY_DDL = """
CREATE TABLE web_sessions (
    id VARCHAR NOT NULL,
    platform VARCHAR NOT NULL,
    label VARCHAR NOT NULL,
    partition VARCHAR NOT NULL,
    landing_url VARCHAR NOT NULL,
    purpose TEXT,
    status VARCHAR(7) NOT NULL,
    current_url VARCHAR,
    current_title VARCHAR,
    last_error TEXT,
    last_used_at DATETIME,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (partition)
)
"""


def _fresh_engine(tmp_path, name="fresh.db"):
    engine = create_engine(f"sqlite:///{tmp_path.as_posix()}/{name}")
    Base.metadata.create_all(bind=engine, tables=_TABLES)
    return engine


def _legacy_engine(tmp_path):
    """DB shaped like the live one pre-fix: UNIQUE(partition) + 8 rows."""
    engine = create_engine(f"sqlite:///{tmp_path.as_posix()}/legacy.db")
    with engine.connect() as conn:
        conn.exec_driver_sql(_LEGACY_DDL)
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX ix_web_sessions_platform ON web_sessions (platform)"
        )
        conn.commit()
    # web_actions (FK -> web_sessions.id) from the current model.
    Base.metadata.create_all(bind=engine, tables=[WebAction.__table__])
    with engine.connect() as conn:
        for i, (platform, label, partition, url, _purpose) in enumerate(
            DEFAULT_SESSIONS[:8]
        ):
            conn.exec_driver_sql(
                "INSERT INTO web_sessions (id, platform, label, partition, "
                "landing_url, status, created_at, updated_at) VALUES "
                f"('legacy-{i}', '{platform}', '{label.replace(chr(39), '')}', "
                f"'{partition}', '{url}', 'idle', "
                "'2026-06-12 00:00:00', '2026-06-12 00:00:00')"
            )
        conn.exec_driver_sql(
            "INSERT INTO web_actions (id, session_id, action_type, payload, "
            "status, created_at) VALUES ('act-1', 'legacy-0', 'navigate', "
            "'{}', 'completed', '2026-06-12 00:00:00')"
        )
        conn.commit()
    return engine


def _session(engine):
    return sessionmaker(bind=engine)()


def _table_ddl(engine, table="web_sessions"):
    with engine.connect() as conn:
        row = conn.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?",
            (table,),
        ).fetchone()
    return " ".join((row[0] or "").split()).lower() if row else ""


class TestDefaults:
    def test_media_sessions_share_a_partition(self):
        """The regression is only meaningful while partitions are shared."""
        partitions = [p for (_, _, p, _, _) in DEFAULT_SESSIONS]
        assert partitions.count("persist:media") >= 2


class TestFreshSeed:
    def test_seed_creates_every_definition(self, tmp_path):
        db = _session(_fresh_engine(tmp_path))
        try:
            result = seed_sessions(db)
            assert result["created"] == len(DEFAULT_SESSIONS)
            assert result["failed"] == 0
            rows = db.query(WebSession).all()
            assert len(rows) == len(DEFAULT_SESSIONS)
            media = [r for r in rows if r.partition == "persist:media"]
            assert len(media) == 5  # displays + 4 media platforms
        finally:
            db.close()

    def test_seed_is_idempotent(self, tmp_path):
        db = _session(_fresh_engine(tmp_path))
        try:
            seed_sessions(db)
            result = seed_sessions(db)
            assert result["created"] == 0
            assert result["failed"] == 0
            assert db.query(WebSession).count() == len(DEFAULT_SESSIONS)
        finally:
            db.close()

    def test_one_poisoned_row_does_not_strand_the_rest(self, tmp_path, monkeypatch):
        """Rows after a failing row must still seed (the original bug's shape)."""
        real_create = session_registry.create_session

        def poisoned(db, *, platform, **kwargs):
            if platform == "mixcloud":
                raise RuntimeError("boom")
            return real_create(db, platform=platform, **kwargs)

        monkeypatch.setattr(session_registry, "create_session", poisoned)
        db = _session(_fresh_engine(tmp_path))
        try:
            result = seed_sessions(db)
            assert result["failed"] == 1
            assert result["created"] == len(DEFAULT_SESSIONS) - 1
            platforms = {r.platform for r in db.query(WebSession).all()}
            assert "mixcloud" not in platforms
            # The rows AFTER mixcloud in DEFAULT_SESSIONS still made it.
            assert {"youtube_watch", "streaming"} <= platforms
        finally:
            db.close()


class TestLegacyMigration:
    def test_rebuild_drops_constraint_and_keeps_data(self, tmp_path):
        engine = _legacy_engine(tmp_path)
        assert "unique (partition)" in _table_ddl(engine)

        assert migrate_web_sessions_schema(engine) is True

        ddl = _table_ddl(engine)
        assert "unique (partition)" not in ddl
        # Pre-existing rows, ids and the FK'd action row all survive.
        db = _session(engine)
        try:
            assert db.query(WebSession).count() == 8
            assert db.get(WebSession, "legacy-0") is not None
            assert db.get(WebAction, "act-1") is not None
            # Platform uniqueness (the real identity) is preserved.
            idx = {i["name"] for i in inspect(engine).get_indexes("web_sessions")}
            assert "ix_web_sessions_platform" in idx
        finally:
            db.close()

    def test_seed_completes_after_rebuild(self, tmp_path):
        engine = _legacy_engine(tmp_path)
        migrate_web_sessions_schema(engine)
        db = _session(engine)
        try:
            result = seed_sessions(db)
            assert result["failed"] == 0
            assert db.query(WebSession).count() == len(DEFAULT_SESSIONS)
            media = db.query(WebSession).filter(
                WebSession.partition == "persist:media"
            ).count()
            assert media == 5
        finally:
            db.close()

    def test_noop_on_fixed_schema(self, tmp_path):
        engine = _fresh_engine(tmp_path, name="fixed.db")
        db = _session(engine)
        try:
            seed_sessions(db)
        finally:
            db.close()
        assert migrate_web_sessions_schema(engine) is False
        db = _session(engine)
        try:
            assert db.query(WebSession).count() == len(DEFAULT_SESSIONS)
        finally:
            db.close()

    def test_noop_when_table_missing(self, tmp_path):
        engine = create_engine(f"sqlite:///{tmp_path.as_posix()}/empty.db")
        assert migrate_web_sessions_schema(engine) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
