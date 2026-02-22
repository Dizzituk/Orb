import hashlib
import logging
import os
import re
import uuid
from .multi_file_detection import _build_multi_file_operation, _detect_multi_file_intent
from .weaver_parser import _is_placeholder_goal, parse_weaver_intent
from app.pot_spec.grounded.__spec_runner_utils_5_utils import _CREATE_BUILDER_AVAILABLE, _DIRECT_BUILDER_AVAILABLE
from app.pot_spec.grounded._spec_runner_utils import _build_simple_spec
from app.pot_spec.grounded._spec_runner_utils import _build_single_segment_manifest, _dedup_evidence_requests, _extract_project_paths, _write_segmentation_output
from app.pot_spec.grounded._spec_runner_utils import _extract_acceptance_from_spec, _extract_file_scope_from_spec
from app.pot_spec.grounded._spec_runner_utils import _extract_requirements_from_spec, _get_job_dir_for_segmentation
from app.pot_spec.grounded._spec_runner_utils import _reconcile_ac_names_against_source
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from app.pot_spec.grounded.___spec_runner_utils_5_utils_1_utils import run_spec_gate_grounded
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
build_direct_spec = None
build_grounded_create_spec = None
