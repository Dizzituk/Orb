from alembic import op
import sqlalchemy as sa


revision = "edu_add_json_fields"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("education_courses", sa.Column("skills_gained", sa.JSON(), nullable=True))
    op.add_column("education_courses", sa.Column("tools_learned", sa.JSON(), nullable=True))
    op.add_column("education_courses", sa.Column("course_details", sa.JSON(), nullable=True))
    op.add_column("education_course_modules", sa.Column("sub_modules", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("education_course_modules", "sub_modules")
    op.drop_column("education_courses", "course_details")
    op.drop_column("education_courses", "tools_learned")
    op.drop_column("education_courses", "skills_gained")
