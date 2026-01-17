"""RAG configuration."""
from .scan_roots import (
    INDEX_ROOTS,
    get_scan_targets,
    get_root_alias,
    is_system_path,
    is_allowed_user_subdir,
    should_skip_directory,
    should_skip_file,
    is_scannable_path,
    ROOT_ALIASES,
    ZOBIE_OUTPUT_DIR,
    get_zobie_output_dir,
    get_latest_zobie_file,
)
