import json
import logging
import os
from app.orchestrator._cohesion_check_utils import CohesionIssue
from app.orchestrator._cohesion_check_utils import _extract_arch_file_paths, _extract_segment_references
from typing import Any, Dict, List, Optional
from app.orchestrator.__cohesion_check_utils_4_utils import run_skeleton_compliance
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
