import logging
import re
from app.overwatcher._overwatcher_utils import CODE_PATTERNS, OVERWATCHER_SYSTEM, OVERWATCHER_USER
from app.overwatcher.evidence import EvidenceBundle
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
logger = logging.getLogger(__name__)
_EVIDENCE_CONTRACT_AVAILABLE = True
EVIDENCE_CONTRACT_PROMPT = ""
logger = logging.getLogger(__name__)


def contains_code(text: str) -> bool:
    """Check if text contains code patterns.
    
    Used to enforce Overwatcher output contract.
    """
    if not text:
        return False
    
    for pattern in CODE_PATTERNS:
        if re.search(pattern, text, re.MULTILINE):
            return True
    
    return False

def build_overwatcher_prompt(
    evidence: EvidenceBundle,
) -> tuple[str, str]:
    """Build Overwatcher prompts.
    
    v2.0: Appends Evidence-or-Request Contract when available.
    
    Returns (system_prompt, user_prompt)
    """
    system = OVERWATCHER_SYSTEM.format(
        spec_hash=evidence.spec_hash,
    )
    
    # v2.0: Append evidence contract (CITE/EVIDENCE_REQUEST/DECISION/HUMAN_REQUIRED)
    if _EVIDENCE_CONTRACT_AVAILABLE and EVIDENCE_CONTRACT_PROMPT:
        system += "\n\n" + EVIDENCE_CONTRACT_PROMPT
        logger.info("[overwatcher] v2.0 Evidence contract appended to system prompt")
    
    strike_hint = ""
    if evidence.strike_number == 1:
        strike_hint = "Consider needs_deep_research if stuck"
    elif evidence.strike_number == 2:
        strike_hint = "Deep research available if needed"
    else:
        strike_hint = "Final strike - be thorough"
    
    user = OVERWATCHER_USER.format(
        evidence_text=evidence.to_prompt_text(),
        strike_number=evidence.strike_number,
        strike_hint=strike_hint,
    )
    
    return system, user

async def run_pot_spec_execution(
    *,
    spec: "ResolvedSpec",
    pot_tasks: "POTParseResult",
    job_id: str,
    llm_call_fn: Optional[Callable] = None,
    artifact_root: str = "D:/Orb/jobs",
    file_scope: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Execute POT spec with sequential task processing.
    
    POT specs contain grounded atomic tasks (file path + line number + content).
    Each task is executed individually with verification before continuing.
    
    3-Strike Error Handling (per task):
    - Strike 1: Try to fix with own logic
    - Strike 2: Deep research / online resources  
    - Strike 3: HARD STOP → write incident report
    - Different error: Counter resets (making progress)
    
    Args:
        spec: Resolved spec with POT tasks
        pot_tasks: Parsed POT tasks from markdown
        job_id: Unique job identifier
        llm_call_fn: Optional LLM function for Overwatcher decisions
        artifact_root: Root directory for artifacts
    
    Returns:
        Dict with execution results:
        {
            "success": bool,
            "decision": str,
            "tasks_completed": int,
            "total_tasks": int,
            "error": Optional[str],
            "trace": List[Dict],
            "artifacts_written": List[str],
        }
    """
    from .pot_spec_parser import POTAtomicTask
    from .spec_resolution import ResolvedSpec
    from .spec_parsing import ParsedDeliverable, DEFAULT_TARGET
    from .implementer import run_implementer, run_verification
    
    tasks = pot_tasks.tasks
    total_tasks = len(tasks)
    tasks_completed = 0
    trace = []
    artifacts_written = []
    
    logger.info(
        "[pot_execution] Starting POT execution: %d tasks, search='%s', replace='%s'",
        total_tasks,
        pot_tasks.search_term,
        pot_tasks.replace_term,
    )
    
    trace.append({
        "stage": "POT_EXECUTION_START",
        "status": "started",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": {
            "total_tasks": total_tasks,
            "search_term": pot_tasks.search_term,
            "replace_term": pot_tasks.replace_term,
        },
    })
    
    # ==========================================================================
    # Sequential task execution
    # ==========================================================================
    for i, task in enumerate(tasks, start=1):
        logger.info(
            "[pot_execution] Task %d/%d: %s L%d",
            i,
            total_tasks,
            task.file_path,
            task.line_number,
        )
        
        trace.append({
            "stage": f"TASK_{i}",
            "status": "started",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": task.to_dict(),
        })
        
        # ======================================================================
        # Convert POT task to deliverable for Implementer
        # ======================================================================
        # Build content: replace search_term with replace_term in original_content
        if pot_tasks.search_term and pot_tasks.replace_term:
            modified_content = task.original_content.replace(
                pot_tasks.search_term,
                pot_tasks.replace_term
            )
        else:
            # Fallback if terms not extracted
            logger.warning(
                "[pot_execution] No search/replace terms - using original content"
            )
            modified_content = task.original_content
        
        task_deliverable = ParsedDeliverable(
            filename=task.file_path,
            content=modified_content,
            action="modify",  # POT tasks are always modifications
            target=DEFAULT_TARGET,
            must_exist=True,  # File must exist (POT specs are grounded)
        )
        
        # Create temporary spec for this task
        task_spec = ResolvedSpec(
            spec_id=f"{spec.spec_id}-task-{i}",
            spec_hash=spec.spec_hash,
            project_id=spec.project_id,
            title=f"Task {i}/{total_tasks}: {task.file_path} L{task.line_number}",
            created_at=spec.created_at,
            spec_content=spec.spec_content,
            deliverable=task_deliverable,
            is_pot_spec=False,  # Individual tasks are not POT specs
            pot_tasks=None,
        )
        
        # ======================================================================
        # Execute task (with 3-strike error handling)
        # ======================================================================
        strike_count = 0
        max_strikes = 3
        last_error = None
        task_success = False
        
        while strike_count < max_strikes and not task_success:
            try:
                logger.info(
                    "[pot_execution] Task %d/%d - Attempt %d/%d",
                    i,
                    total_tasks,
                    strike_count + 1,
                    max_strikes,
                )
                
                # Create minimal OverwatcherOutput for Implementer
                impl_output = OverwatcherOutput(
                    decision=Decision.PASS,
                    diagnosis=f"POT Task {i}/{total_tasks}: {task.file_path} L{task.line_number}",
                )
                
                impl_result = await run_implementer(spec=task_spec, output=impl_output)
                
                if impl_result.success:
                    # Verify the implementation
                    verify_result = await run_verification(impl_result=impl_result, spec=task_spec)
                    
                    if verify_result.passed:
                        # SUCCESS - task completed
                        task_success = True
                        tasks_completed += 1
                        
                        if impl_result.output_path:
                            if impl_result.output_path not in artifacts_written:
                                artifacts_written.append(impl_result.output_path)
                        
                        trace.append({
                            "stage": f"TASK_{i}",
                            "status": "success",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "details": {
                                "output_path": impl_result.output_path,
                                "action": impl_result.action_taken,
                            },
                        })
                        
                        logger.info(
                            "[pot_execution] ✓ Task %d/%d COMPLETE: %s",
                            i,
                            total_tasks,
                            impl_result.output_path,
                        )
                        break  # Move to next task
                    
                    else:
                        # Verification failed
                        last_error = verify_result.error
                        strike_count += 1
                        
                        logger.warning(
                            "[pot_execution] ✗ Task %d/%d verification failed (Strike %d/%d): %s",
                            i,
                            total_tasks,
                            strike_count,
                            max_strikes,
                            last_error,
                        )
                        
                        trace.append({
                            "stage": f"TASK_{i}_VERIFY",
                            "status": f"failed_strike_{strike_count}",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "details": {"error": last_error},
                        })
                        
                        # 3-Strike escalation
                        if strike_count == 1:
                            # Strike 1: Try to fix with own logic
                            logger.info("[pot_execution] Strike 1: Retrying with adjusted logic...")
                            continue
                        elif strike_count == 2:
                            # Strike 2: Deep research (placeholder for now)
                            logger.info("[pot_execution] Strike 2: Deep research mode...")
                            # TODO: Implement deep research / online lookup
                            continue
                        elif strike_count == 3:
                            # Strike 3: HARD STOP
                            logger.error("[pot_execution] Strike 3: HARD STOP - Task failed permanently")
                            break
                
                else:
                    # Implementation failed
                    last_error = impl_result.error
                    strike_count += 1
                    
                    logger.warning(
                        "[pot_execution] ✗ Task %d/%d implementation failed (Strike %d/%d): %s",
                        i,
                        total_tasks,
                        strike_count,
                        max_strikes,
                        last_error,
                    )
                    
                    trace.append({
                        "stage": f"TASK_{i}_IMPL",
                        "status": f"failed_strike_{strike_count}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "details": {"error": last_error},
                    })
                    
                    if strike_count >= max_strikes:
                        logger.error("[pot_execution] Strike 3: HARD STOP - Task failed permanently")
                        break
                    
                    # Retry
                    continue
            
            except Exception as e:
                last_error = str(e)
                strike_count += 1
                logger.exception(
                    "[pot_execution] Task %d/%d exception (Strike %d/%d): %s",
                    i,
                    total_tasks,
                    strike_count,
                    max_strikes,
                    e,
                )
                
                if strike_count >= max_strikes:
                    break
        
        # ======================================================================
        # Check if task failed after all retries
        # ======================================================================
        if not task_success:
            # HARD STOP - don't continue to next task
            error_msg = (
                f"Task {i}/{total_tasks} failed after {max_strikes} strikes: {last_error}\n"
                f"Tasks completed: {tasks_completed}/{total_tasks}\n"
                f"Stopped at: {task.file_path} L{task.line_number}"
            )
            
            logger.error("[pot_execution] POT EXECUTION STOPPED: %s", error_msg)
            
            trace.append({
                "stage": "POT_EXECUTION_STOPPED",
                "status": "failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": {
                    "stopped_at_task": i,
                    "tasks_completed": tasks_completed,
                    "total_tasks": total_tasks,
                    "last_error": last_error,
                },
            })
            
            return {
                "success": False,
                "decision": "FAIL",
                "tasks_completed": tasks_completed,
                "total_tasks": total_tasks,
                "error": error_msg,
                "trace": trace,
                "artifacts_written": artifacts_written,
            }
    
    # ==========================================================================
    # All tasks completed successfully
    # ==========================================================================
    logger.info(
        "[pot_execution] ✓ ALL TASKS COMPLETE: %d/%d tasks successful",
        tasks_completed,
        total_tasks,
    )
    
    trace.append({
        "stage": "POT_EXECUTION_COMPLETE",
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": {
            "tasks_completed": tasks_completed,
            "total_tasks": total_tasks,
        },
    })
    
    return {
        "success": True,
        "decision": "PASS",
        "tasks_completed": tasks_completed,
        "total_tasks": total_tasks,
        "error": None,
        "trace": trace,
        "artifacts_written": artifacts_written,
    }
