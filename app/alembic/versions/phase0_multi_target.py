# Purpose: Phase 0: multi-target awareness on build_projects.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""Phase 0: multi-target awareness on build_projects.

Adds target_ids (JSON, list of target_id strings) and target_group_id
(nullable string) to build_projects. Backfills existing rows with
target_ids = [build_target_id] when build_target_id is set.

v1.0 (2026-04-11): Initial.
"""
from alembic import op
import sqlalchemy as sa


revision = "phase0_multi_target"
down_revision = "edu_add_json_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("build_projects", sa.Column("target_ids", sa.JSON(), nullable=True))
    op.add_column("build_projects", sa.Column("target_group_id", sa.String(), nullable=True))
    # Backfill: existing single-target rows get target_ids = [build_target_id]
    bind = op.get_bind()
    bind.execute(sa.text(
        "UPDATE build_projects "
        "SET target_ids = json_array(build_target_id) "
        "WHERE build_target_id IS NOT NULL AND target_ids IS NULL"
    ))


def downgrade() -> None:
    op.drop_column("build_projects", "target_group_id")
    op.drop_column("build_projects", "target_ids")
