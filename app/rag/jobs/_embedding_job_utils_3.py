import logging
import threading
import time
from app.rag.jobs._embedding_job_utils import EMBEDDING_BATCH_SIZE, EMBEDDING_RATE_LIMIT_DELAY, SQLITE_LOCK_INITIAL_BACKOFF, SQLITE_LOCK_MAX_BACKOFF, SQLITE_LOCK_MAX_RETRIES, _is_sqlite_lock_error
from app.rag.jobs._embedding_job_utils import EMBEDDING_MODEL, STATUS_FILE, classify_chunk_priority, compute_content_hash
from app.rag.jobs._embedding_job_utils import EmbeddingPriority
from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import Callable, Dict, List, Optional
from app.rag.jobs.__embedding_job_utils_3_utils import EmbeddingJob
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
_current_status = EmbeddingJobStatus()
_job_lock = threading.Lock()
