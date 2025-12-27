# FILE: app/overwatcher/replay.py
"""Block 12: Deterministic Replay Pack.

Generates a single folder containing everything needed to replay decisions.

Contents:
- spec/arch/critique/plan artifacts
- Ledger events
- Model identifiers used per stage
- Verification outputs
- Exact tool commands run
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.overwatcher.schemas import ReplayPack

logger = logging.getLogger(__name__)


# =============================================================================
# Replay Pack Generation
# =============================================================================

def generate_replay_pack(
    job_id: str,
    job_artifact_root: str,
    output_dir: Optional[str] = None,
) -> ReplayPack:
    """Generate a replay pack for a job.
    
    Collects all artifacts into a single portable folder.
    
    Args:
        job_id: Job UUID
        job_artifact_root: Root path for job artifacts
        output_dir: Where to write the pack (defaults to job_artifact_root/replay_packs)
    
    Returns:
        ReplayPack with paths to collected artifacts
    """
    pack_id = str(uuid4())
    job_dir = Path(job_artifact_root) / "jobs" / job_id
    
    if output_dir:
        pack_dir = Path(output_dir) / f"replay_{job_id}_{pack_id[:8]}"
    else:
        pack_dir = Path(job_artifact_root) / "replay_packs" / f"replay_{job_id}_{pack_id[:8]}"
    
    pack_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[replay] Generating pack for job {job_id}")
    
    pack = ReplayPack(
        pack_id=pack_id,
        job_id=job_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    
    # 1. Copy spec artifacts
    spec_dir = job_dir / "spec"
    if spec_dir.exists():
        dest_spec = pack_dir / "spec"
        shutil.copytree(spec_dir, dest_spec, dirs_exist_ok=True)
        
        # Find the latest spec
        spec_files = sorted(dest_spec.glob("spec_v*.json"), reverse=True)
        if spec_files:
            pack.spec_path = str(spec_files[0].relative_to(pack_dir))
    
    # 2. Copy arch artifacts
    arch_dir = job_dir / "arch"
    if arch_dir.exists():
        dest_arch = pack_dir / "arch"
        shutil.copytree(arch_dir, dest_arch, dirs_exist_ok=True)
        
        # Find the latest arch
        arch_files = sorted(dest_arch.glob("arch_v*.md"), reverse=True)
        if arch_files:
            pack.arch_path = str(arch_files[0].relative_to(pack_dir))
    
    # 3. Copy critique artifacts
    critique_dir = job_dir / "critique"
    if critique_dir.exists():
        dest_critique = pack_dir / "critique"
        shutil.copytree(critique_dir, dest_critique, dirs_exist_ok=True)
        
        # List all critique files
        critique_files = sorted(dest_critique.glob("critique_v*.json"))
        pack.critique_paths = [str(f.relative_to(pack_dir)) for f in critique_files]
    
    # 4. Copy plan artifacts
    plan_dir = job_dir / "plan"
    if plan_dir.exists():
        dest_plan = pack_dir / "plan"
        shutil.copytree(plan_dir, dest_plan, dirs_exist_ok=True)
        
        # Find the latest plan
        plan_files = sorted(dest_plan.glob("chunks_v*.json"), reverse=True)
        if plan_files:
            pack.plan_path = str(plan_files[0].relative_to(pack_dir))
    
    # 5. Copy ledger
    ledger_dir = job_dir / "ledger"
    if ledger_dir.exists():
        dest_ledger = pack_dir / "ledger"
        shutil.copytree(ledger_dir, dest_ledger, dirs_exist_ok=True)
        pack.ledger_path = "ledger/events.ndjson"
    
    # 6. Copy verification outputs
    verification_dir = job_dir / "verification"
    if verification_dir.exists():
        dest_verification = pack_dir / "verification"
        shutil.copytree(verification_dir, dest_verification, dirs_exist_ok=True)
        
        verification_files = list(dest_verification.glob("**/*.json"))
        pack.verification_paths = [str(f.relative_to(pack_dir)) for f in verification_files]
    
    # 7. Extract model versions from ledger
    ledger_path = job_dir / "ledger" / "events.ndjson"
    if ledger_path.exists():
        pack.model_versions = extract_model_versions(ledger_path)
    
    # 8. Create commands log from ledger
    if ledger_path.exists():
        commands_log = extract_commands_log(ledger_path)
        commands_path = pack_dir / "commands.log"
        commands_path.write_text("\n".join(commands_log), encoding="utf-8")
        pack.commands_log_path = "commands.log"
    
    # 9. Write pack manifest
    manifest_path = pack_dir / "manifest.json"
    manifest_path.write_text(pack.to_json(), encoding="utf-8")
    
    logger.info(f"[replay] Pack generated: {pack_dir}")
    
    return pack


def extract_model_versions(ledger_path: Path) -> Dict[str, str]:
    """Extract model identifiers from ledger events."""
    models = {}
    
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    event = json.loads(line)
                    stage = event.get("stage_name") or event.get("stage") or event.get("event", "")
                    model = event.get("model")
                    
                    if model and stage:
                        models[stage] = model
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        logger.warning(f"[replay] Failed to extract model versions: {e}")
    
    return models


def extract_commands_log(ledger_path: Path) -> List[str]:
    """Extract commands that were run from ledger events."""
    commands = []
    
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    event = json.loads(line)
                    
                    # Look for verification commands
                    if "command" in event:
                        ts = event.get("ts", "")
                        cmd = event.get("command", "")
                        exit_code = event.get("exit_code", "?")
                        commands.append(f"[{ts}] {cmd} -> exit {exit_code}")
                    
                    # Look for command_results
                    if "command_results" in event:
                        for result in event["command_results"]:
                            if isinstance(result, dict):
                                ts = event.get("ts", "")
                                cmd = result.get("command", "")
                                exit_code = result.get("exit_code", "?")
                                commands.append(f"[{ts}] {cmd} -> exit {exit_code}")
                                
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        logger.warning(f"[replay] Failed to extract commands: {e}")
    
    return commands


def load_replay_pack(pack_dir: str) -> Optional[ReplayPack]:
    """Load a replay pack from disk."""
    manifest_path = Path(pack_dir) / "manifest.json"
    
    if not manifest_path.exists():
        logger.warning(f"[replay] Manifest not found: {manifest_path}")
        return None
    
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return ReplayPack(
            pack_id=data.get("pack_id", ""),
            job_id=data.get("job_id", ""),
            created_at=data.get("created_at", ""),
            spec_path=data.get("spec_path", ""),
            arch_path=data.get("arch_path", ""),
            critique_paths=data.get("critique_paths", []),
            plan_path=data.get("plan_path", ""),
            ledger_path=data.get("ledger_path", ""),
            model_versions=data.get("model_versions", {}),
            verification_paths=data.get("verification_paths", []),
            commands_log_path=data.get("commands_log_path", ""),
        )
    except Exception as e:
        logger.error(f"[replay] Failed to load pack: {e}")
        return None


def compare_replay_packs(
    pack_a: ReplayPack,
    pack_b: ReplayPack,
    pack_a_dir: str,
    pack_b_dir: str,
) -> Dict[str, Any]:
    """Compare two replay packs for drift detection.
    
    Returns dict with differences found.
    """
    differences = {
        "model_changes": {},
        "spec_changes": False,
        "arch_changes": False,
        "critique_count_diff": 0,
    }
    
    # Compare model versions
    all_stages = set(pack_a.model_versions.keys()) | set(pack_b.model_versions.keys())
    for stage in all_stages:
        model_a = pack_a.model_versions.get(stage)
        model_b = pack_b.model_versions.get(stage)
        if model_a != model_b:
            differences["model_changes"][stage] = {"old": model_a, "new": model_b}
    
    # Compare spec content
    if pack_a.spec_path and pack_b.spec_path:
        spec_a = Path(pack_a_dir) / pack_a.spec_path
        spec_b = Path(pack_b_dir) / pack_b.spec_path
        if spec_a.exists() and spec_b.exists():
            differences["spec_changes"] = spec_a.read_text() != spec_b.read_text()
    
    # Compare arch content
    if pack_a.arch_path and pack_b.arch_path:
        arch_a = Path(pack_a_dir) / pack_a.arch_path
        arch_b = Path(pack_b_dir) / pack_b.arch_path
        if arch_a.exists() and arch_b.exists():
            differences["arch_changes"] = arch_a.read_text() != arch_b.read_text()
    
    # Compare critique counts
    differences["critique_count_diff"] = len(pack_b.critique_paths) - len(pack_a.critique_paths)
    
    return differences


__all__ = [
    "generate_replay_pack",
    "extract_model_versions",
    "extract_commands_log",
    "load_replay_pack",
    "compare_replay_packs",
]
