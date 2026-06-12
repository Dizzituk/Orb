# FILE: app/optimize/profiler.py
# Purpose: Phase B: Profiler.
# Called-by: app.optimize.executor, app.optimize.orchestrator
# Depends-on: app.optimize.config, app.optimize.models, app.sandbox_walk
# Last-renovated: 2026-06-11
"""
Phase B: Profiler.

Measures each chunk against performance metrics to establish a
quantitative baseline before proposing changes.

For Python/ASTRA core:
  - Line count, file size, cyclomatic complexity estimate
  - Import coupling (in/out)
  - Duplicate code ratio (token overlap with neighbours)
  - Days since last modification

Produces: ProfileResult with per-chunk metrics and ranked bottlenecks.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-OPT-001.
"""
from __future__ import annotations

import ast
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from app.optimize.config import (
    HIGH_COMPLEXITY_LINES,
    HIGH_COUPLING_THRESHOLD,
    MAX_CHUNKS_TO_PROFILE,
)
from app.optimize.models import (
    Bottleneck,
    ChunkManifest,
    ChunkMetrics,
    CodeChunk,
    ProfileResult,
)

logger = logging.getLogger(__name__)


async def profile(
    manifest: ChunkManifest,
    target_root: str,
    emit: Optional[Callable[[str], None]] = None,
) -> ProfileResult:
    """Run Phase B: Profile all chunks in the manifest.

    Args:
        manifest: ChunkManifest from the Decompose phase.
        target_root: Root directory of the target.
        emit: Progress callback.

    Returns:
        ProfileResult with metrics and bottlenecks.
    """
    emit = emit or (lambda msg: None)
    t_start = time.time()

    emit(f"📊 Profile: Measuring {len(manifest.chunks)} chunks...")

    # Profile each chunk
    metrics_list = []
    chunks_to_profile = manifest.chunks[:MAX_CHUNKS_TO_PROFILE]

    for i, chunk in enumerate(chunks_to_profile):
        if (i + 1) % 50 == 0:
            emit(f"   Profiled {i + 1}/{len(chunks_to_profile)}...")

        metrics = _profile_chunk(chunk, target_root)
        metrics_list.append(metrics)

    emit(f"   Profiled {len(metrics_list)} chunks")

    # Identify bottlenecks
    bottlenecks = _identify_bottlenecks(metrics_list)
    emit(f"   Found {len(bottlenecks)} bottlenecks")

    total_complexity = sum(m.cyclomatic_complexity for m in metrics_list)

    result = ProfileResult(
        target=manifest.target,
        chunk_metrics=metrics_list,
        bottlenecks=bottlenecks,
        total_complexity=total_complexity,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    duration = time.time() - t_start
    emit(f"✅ Profile complete ({duration:.1f}s): "
         f"{len(bottlenecks)} bottlenecks, complexity={total_complexity}")

    return result


# ═══════════════════════════════════════════════════════════════════
# Per-chunk profiling
# ═══════════════════════════════════════════════════════════════════

def _profile_chunk(chunk: CodeChunk, target_root: str) -> ChunkMetrics:
    """Profile a single code chunk. Reads source from sandbox."""
    fpath = Path(target_root) / chunk.path

    metrics = ChunkMetrics(
        path=chunk.path,
        line_count=chunk.lines,
        size_bytes=chunk.size_bytes,
        coupling_in=chunk.dependents,
        coupling_out=chunk.dependencies,
    )

    # Cyclomatic complexity (Python only — read from sandbox)
    if chunk.path.endswith('.py'):
        source = _read_source_from_sandbox(chunk.path)
        if source:
            metrics.cyclomatic_complexity = _estimate_cyclomatic_from_source(source)

    # Last modified (host stat is fine — sandbox doesn't expose mtime)
    try:
        mtime = fpath.stat().st_mtime
        modified = datetime.fromtimestamp(mtime, tz=timezone.utc)
        now = datetime.now(timezone.utc)
        metrics.last_modified_days = (now - modified).days
    except OSError:
        pass

    # Anomaly detection
    metrics.anomaly_flags = _detect_anomalies(metrics, chunk)

    return metrics


def _read_source_from_sandbox(chunk_path: str) -> Optional[str]:
    """Read Python source from the sandbox filesystem."""
    try:
        from app.sandbox_walk import sandbox_read_python
        return sandbox_read_python(chunk_path)
    except Exception as e:
        logger.debug("[profiler] Sandbox read failed for %s: %s", chunk_path, e)
        return None


def _estimate_cyclomatic_from_source(source: str) -> int:
    """Estimate cyclomatic complexity from Python source text.

    Counts decision points: if, elif, for, while, except, with,
    and, or, assert, comprehensions.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return 0

    complexity = 1  # Base path
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            complexity += 1
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            complexity += 1
        elif isinstance(node, (ast.While,)):
            complexity += 1
        elif isinstance(node, (ast.ExceptHandler,)):
            complexity += 1
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            complexity += 1
        elif isinstance(node, (ast.Assert,)):
            complexity += 1
        elif isinstance(node, (ast.BoolOp,)):
            # Each 'and'/'or' adds a path
            complexity += len(node.values) - 1
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            complexity += 1

    return complexity


def _detect_anomalies(
    metrics: ChunkMetrics,
    chunk: CodeChunk,
) -> List[str]:
    """Flag statistical anomalies in a chunk's metrics."""
    flags = []

    if metrics.line_count > HIGH_COMPLEXITY_LINES:
        flags.append(f"high_line_count:{metrics.line_count}")

    if metrics.size_bytes > 30 * 1024:
        flags.append(f"oversized:{metrics.size_bytes // 1024}KB")

    total_coupling = metrics.coupling_in + metrics.coupling_out
    if total_coupling > HIGH_COUPLING_THRESHOLD:
        flags.append(f"high_coupling:{total_coupling}")

    if metrics.cyclomatic_complexity > 50:
        flags.append(f"high_complexity:{metrics.cyclomatic_complexity}")

    if metrics.last_modified_days > 90 and metrics.coupling_in == 0:
        flags.append("stale_uncoupled")

    return flags


# ═══════════════════════════════════════════════════════════════════
# Bottleneck identification
# ═══════════════════════════════════════════════════════════════════

def _identify_bottlenecks(
    metrics_list: List[ChunkMetrics],
) -> List[Bottleneck]:
    """Identify and rank bottlenecks from profiling data."""
    bottlenecks = []

    if not metrics_list:
        return bottlenecks

    # Size bottlenecks — largest files
    by_size = sorted(metrics_list, key=lambda m: m.size_bytes, reverse=True)
    for m in by_size[:5]:
        if m.size_bytes > 20 * 1024:
            bottlenecks.append(Bottleneck(
                path=m.path,
                metric="size_bytes",
                value=m.size_bytes,
                impact_score=min(1.0, m.size_bytes / (100 * 1024)),
                description=f"Large file: {m.size_bytes // 1024}KB",
            ))

    # Complexity bottlenecks
    by_complexity = sorted(
        metrics_list, key=lambda m: m.cyclomatic_complexity, reverse=True,
    )
    for m in by_complexity[:5]:
        if m.cyclomatic_complexity > 30:
            bottlenecks.append(Bottleneck(
                path=m.path,
                metric="cyclomatic_complexity",
                value=m.cyclomatic_complexity,
                impact_score=min(1.0, m.cyclomatic_complexity / 100),
                description=f"High complexity: {m.cyclomatic_complexity}",
            ))

    # Coupling bottlenecks — highly coupled modules
    by_coupling = sorted(
        metrics_list,
        key=lambda m: m.coupling_in + m.coupling_out,
        reverse=True,
    )
    for m in by_coupling[:5]:
        total = m.coupling_in + m.coupling_out
        if total > HIGH_COUPLING_THRESHOLD:
            bottlenecks.append(Bottleneck(
                path=m.path,
                metric="coupling",
                value=total,
                impact_score=min(1.0, total / 50),
                description=f"High coupling: {m.coupling_in} in, {m.coupling_out} out",
            ))

    # Sort by impact
    bottlenecks.sort(key=lambda b: b.impact_score, reverse=True)

    return bottlenecks[:15]
