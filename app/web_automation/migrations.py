# FILE: app/web_automation/migrations.py
# Purpose: Schema migrations for web_automation tables (legacy UNIQUE(partition) rebuild).
# Called-by: app.db.init_db
# Depends-on: sqlalchemy (schema of app.web_automation.models frozen in DDL below)
# Last-renovated: 2026-07-01
"""
Schema migrations for the web_automation tables.

migrate_web_sessions_schema — drops the legacy UNIQUE constraint on
web_sessions.partition. The media sessions (displays / soundcloud /
mixcloud / youtube_watch / streaming) share ONE Electron partition
("persist:media") BY DESIGN so a single login per site survives restarts.
The original model declared partition unique=True, which SQLite bakes into
the table DDL as an inline UNIQUE constraint; on any DB built from that
model the second persist:media insert raised IntegrityError and
seed_sessions() aborted at 8 of 12 defaults (the live DB hit this too —
verified 2026-07-01).

SQLite cannot drop an inline constraint, so this rebuilds the table
(create-new -> copy -> drop-old -> rename -> recreate platform index)
inside one immediate transaction: all-or-nothing, and web_actions rows
(FK -> web_sessions.id) are untouched because ids are preserved.
Idempotent — a no-op unless the legacy constraint is present in the DDL.
Failure is logged, never raised: startup must not die on a migration,
and a failed rebuild just leaves today's behaviour in place.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Target schema for web_sessions: app.web_automation.models.WebSession with
# NO unique constraint on partition. Frozen copy on purpose — a migration
# describes the target state at the time it was written.
_NEW_DDL = """
CREATE TABLE web_sessions_new (
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
    PRIMARY KEY (id)
)
"""

_COLUMNS = (
    "id, platform, label, partition, landing_url, purpose, status, "
    "current_url, current_title, last_error, last_used_at, created_at, updated_at"
)


def migrate_web_sessions_schema(engine) -> bool:
    """Rebuild web_sessions without the legacy UNIQUE(partition).

    Returns True only when a rebuild actually happened.
    """
    if engine.dialect.name != "sqlite":
        return False

    from sqlalchemy import inspect

    try:
        if "web_sessions" not in inspect(engine).get_table_names():
            return False  # fresh DB — create_all builds the fixed schema

        with engine.connect() as probe:
            row = probe.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='web_sessions'"
            ).fetchone()
        ddl = " ".join((row[0] or "").split()).lower() if row else ""
        if "unique (partition)" not in ddl:
            return False  # already migrated / created from the fixed model

        # Standard SQLite table rebuild. pysqlite: AUTOCOMMIT + explicit
        # BEGIN IMMEDIATE so the whole rebuild is one transaction and we
        # hold the write lock from the start.
        conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        try:
            # web_actions FK (ON DELETE CASCADE) must not fire mid-rebuild.
            conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
            conn.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                conn.exec_driver_sql(_NEW_DDL)
                conn.exec_driver_sql(
                    f"INSERT INTO web_sessions_new ({_COLUMNS}) "
                    f"SELECT {_COLUMNS} FROM web_sessions"
                )
                conn.exec_driver_sql("DROP TABLE web_sessions")
                conn.exec_driver_sql(
                    "ALTER TABLE web_sessions_new RENAME TO web_sessions"
                )
                conn.exec_driver_sql(
                    "CREATE UNIQUE INDEX ix_web_sessions_platform "
                    "ON web_sessions (platform)"
                )
                bad = conn.exec_driver_sql(
                    "PRAGMA foreign_key_check(web_actions)"
                ).fetchall()
                if bad:
                    raise RuntimeError(
                        f"foreign_key_check failed post-rebuild: {bad[:5]}"
                    )
                conn.exec_driver_sql("COMMIT")
            except Exception:
                conn.exec_driver_sql("ROLLBACK")
                raise
            finally:
                conn.exec_driver_sql("PRAGMA foreign_keys=ON")
        finally:
            conn.close()
    except Exception as exc:
        logger.warning(
            "[db_migrate] web_sessions UNIQUE(partition) rebuild failed "
            "(seed of shared-partition sessions will keep failing): %s", exc
        )
        return False

    logger.info(
        "[db_migrate] web_sessions rebuilt without UNIQUE(partition) "
        "(media sessions share persist:media)"
    )
    print("[db_migrate] web_sessions: dropped legacy UNIQUE(partition) constraint")
    return True
