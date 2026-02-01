# FILE: app/pot_spec/grounded/step_derivation.py
"""
Step/Test Derivation for SpecGate

This module derives implementation steps and acceptance tests from
detected domains and resolved decisions/assumptions.

Responsibilities:
- Derive implementation steps based on domain detection
- Derive acceptance tests based on domain detection
- Apply conditional logic based on resolved decisions/assumptions
- Handle sandbox_file, game, mobile_app, and greenfield domains

Key Features:
- v1.5: Conditional step/test derivation
- v1.6: Sandbox file domain steps
- v1.12: Game domain steps and tests
- v1.31: REWRITE_IN_PLACE branch support
- v1.39: Multi-target read handling

Used by:
- question_generator.py for step/test derivation
- spec_runner.py for final derivation

Version: v2.0 (2026-02-01) - Extracted from spec_generation.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .spec_models import GroundedPOTSpec
from .domain_detection import detect_domains
from .sandbox_discovery import OutputMode

logger = logging.getLogger(__name__)


__all__ = [
    "_derive_steps_from_domain",
    "_derive_tests_from_domain",
]


def _derive_steps_from_domain(intent: Dict[str, Any], spec: GroundedPOTSpec) -> List[str]:
    """
    v1.5: Derive implementation steps from domain + resolved decisions + assumptions.
    
    This is SpecGate's job, NOT the user's. Once product decisions are made,
    the steps can be derived automatically.
    
    Args:
        intent: Parsed Weaver intent
        spec: GroundedPOTSpec with assumptions and decisions
        
    Returns:
        List of implementation step strings
    """
    weaver_text = intent.get("raw_text", "") or ""
    detected_domains = detect_domains(weaver_text)
    
    logger.info(
        "[step_derivation] _derive_steps_from_domain: raw_text_len=%d, detected_domains=%s",
        len(weaver_text), detected_domains
    )
    
    # Build lookup of resolved values: decisions override assumptions
    resolved = {}
    for assumption in spec.assumptions:
        resolved[assumption.topic] = assumption.assumed_value
    for key, value in spec.decisions.items():
        resolved[key] = value
    
    steps = []
    
    # v1.6: Sandbox file domain - specific steps for file discovery/read/reply tasks
    if "sandbox_file" in detected_domains:
        # v1.39: Multi-target read has different steps
        if spec.is_multi_target_read and spec.multi_target_files:
            valid_count = sum(1 for ft in spec.multi_target_files if ft.found)
            file_names = [ft.name for ft in spec.multi_target_files if ft.found][:4]
            files_display = ", ".join(file_names) + ("..." if len(file_names) > 4 else "")
            steps = [
                f"Read {valid_count} files: {files_display}",
                "Parse and understand content from each file",
                "Synthesize combined understanding of all file contents",
            ]
            output_mode = spec.sandbox_output_mode
            if output_mode == OutputMode.SEPARATE_REPLY_FILE.value:
                steps.append(f"Write synthesized reply to: `{spec.sandbox_output_path}`")
                steps.append("Verify reply.txt file exists")
            elif output_mode == OutputMode.REWRITE_IN_PLACE.value:
                steps.append("Insert answers under each question in source files")
            else:  # CHAT_ONLY or None
                steps.append("[Present synthesized reply in chat - no file modification]")
            return steps
        elif spec.sandbox_discovery_used and spec.sandbox_input_path:
            steps = [
                f"Read input file from sandbox: `{spec.sandbox_input_path}`",
                "Parse and understand the question/content in the file",
                "Generate reply based on file content (included in SPoT output)",
            ]
            output_mode = spec.sandbox_output_mode
            # v1.31: Added REWRITE_IN_PLACE branch
            if output_mode == OutputMode.REWRITE_IN_PLACE.value:
                steps.append("Insert answers under each question in-place")
                steps.append(f"Verify file updated: `{spec.sandbox_input_path}`")
            elif output_mode == OutputMode.APPEND_IN_PLACE.value:
                steps.append("Append reply beneath question in same file")
                steps.append(f"Verify file updated: `{spec.sandbox_input_path}`")
            elif output_mode == OutputMode.SEPARATE_REPLY_FILE.value:
                steps.append(f"Write reply to: `{spec.sandbox_output_path}`")
                steps.append("Verify reply.txt file exists")
            else:  # CHAT_ONLY or None
                steps.append("[Chat only - no file modification]")
        else:
            steps = [
                "Discover target file in sandbox (Desktop or Documents)",
                "Read and parse the file content",
                "Generate reply based on file content (included in SPoT output)",
                "[For later stages] Output based on detected mode",
            ]
        return steps
    
    # v1.12: GAME DOMAIN - Takes priority over mobile_app
    if "game" in detected_domains:
        logger.info("[step_derivation] v1.12 GAME domain detected - using game steps")
        steps = [
            "Analyze game requirements and create technical design",
            "Set up project structure (HTML/CSS/JS or chosen framework)",
            "Implement game board/playfield rendering",
            "Implement game piece/entity logic",
            "Implement player input handling (keyboard/touch controls)",
            "Implement core game mechanics (movement, collision, scoring)",
            "Add game state management (start, pause, game over)",
            "Implement scoring and level progression",
            "Add visual polish (animations, transitions)",
            "Testing and bug fixes",
        ]
        return steps
    
    if "mobile_app" in detected_domains:
        # Mobile app domain - conditional implementation steps
        platform = resolved.get("platform_v1", "")
        if "android" in platform.lower():
            steps.append("Set up Android project (Android Studio, Gradle)")
        elif "ios" in platform.lower() or "both" in platform.lower():
            steps.append("Set up mobile project structure (Android + iOS)")
        else:
            steps.append("Set up mobile project structure")
        
        steps.append("Implement local encrypted data storage layer")
        steps.append("Build core UI screens (shift start/stop, daily summary)")
        
        input_mode = resolved.get("input_mode_v1", "")
        if "voice" in input_mode.lower() and "manual" in input_mode.lower():
            steps.append("Implement push-to-talk voice input with manual fallback")
        elif "voice" in input_mode.lower():
            steps.append("Implement voice input handler")
        elif "manual" in input_mode.lower() or "screenshot" in input_mode.lower():
            steps.append("Implement manual input forms (screenshot import + manual entry)")
        
        ocr_scope = resolved.get("ocr_scope_v1", "")
        if ocr_scope:
            if "finish tour" in ocr_scope.lower() or "completed parcels" in ocr_scope.lower():
                steps.append("Implement screenshot OCR parser for Finish Tour screen (Successfully Completed Parcels)")
            elif "multiple" in ocr_scope.lower():
                steps.append("Implement screenshot OCR parser for multiple screen formats")
        
        steps.append("Implement pay/cost/net calculations (parcel rate, fuel, wear & tear)")
        steps.append("Implement end-of-week summary calculations")
        
        sync_behaviour = resolved.get("sync_behaviour", "")
        sync_target = resolved.get("sync_target", "")
        if "local-only" not in sync_behaviour.lower() and "local only" not in sync_behaviour.lower():
            if "export" in sync_target.lower() or "file" in sync_target.lower():
                steps.append("Add export functionality for ASTRA integration (file-based)")
            elif "endpoint" in sync_target.lower() or "live" in sync_target.lower():
                steps.append("Build sync mechanism and ASTRA integration endpoint")
        
        steps.append("Integration testing")
        steps.append("Security audit (encryption, data handling)")
        
    else:
        # Generic steps for unknown domains
        steps = [
            "Analyze requirements and create technical design",
            "Set up project structure and dependencies",
            "Implement core functionality",
            "Add error handling and edge cases",
            "Write tests and documentation",
            "Integration testing",
            "Security review",
        ]
    
    return steps


def _derive_tests_from_domain(intent: Dict[str, Any], spec: GroundedPOTSpec) -> List[str]:
    """
    v1.5: Derive acceptance tests from domain + resolved decisions + assumptions.
    
    Args:
        intent: Parsed Weaver intent
        spec: GroundedPOTSpec with assumptions and decisions
        
    Returns:
        List of acceptance test strings
    """
    weaver_text = intent.get("raw_text", "") or ""
    detected_domains = detect_domains(weaver_text)
    
    # Build lookup of resolved values: decisions override assumptions
    resolved = {}
    for assumption in spec.assumptions:
        resolved[assumption.topic] = assumption.assumed_value
    for key, value in spec.decisions.items():
        resolved[key] = value
    
    tests = []
    
    # v1.6: Sandbox file domain
    if "sandbox_file" in detected_domains:
        # v1.39: Multi-target read has different tests
        if spec.is_multi_target_read and spec.multi_target_files:
            valid_count = sum(1 for ft in spec.multi_target_files if ft.found)
            total_count = len(spec.multi_target_files)
            tests = [
                f"{valid_count}/{total_count} target files were found and read successfully",
                "Content from all files was correctly parsed",
                "Combined/synthesized understanding was generated",
            ]
            output_mode = spec.sandbox_output_mode
            if output_mode == OutputMode.SEPARATE_REPLY_FILE.value:
                tests.append(f"Synthesized reply written to `{spec.sandbox_output_path}`")
                tests.append("reply.txt file exists and contains coherent synthesis")
            elif output_mode == OutputMode.REWRITE_IN_PLACE.value:
                tests.append("Answers inserted in each source file")
            else:  # CHAT_ONLY
                tests.append("Synthesized reply presented in chat")
            return tests
        elif spec.sandbox_discovery_used and spec.sandbox_input_path:
            tests = [
                f"Input file `{spec.sandbox_input_path}` was found and read successfully",
                "File content was correctly parsed and understood",
                "Reply was generated based on file content",
            ]
            if spec.sandbox_input_excerpt and spec.sandbox_selected_type and spec.sandbox_selected_type.lower() != "unknown":
                tests.insert(1, f"Input content type identified: {spec.sandbox_selected_type}")
            output_mode = spec.sandbox_output_mode
            # v1.31: Added REWRITE_IN_PLACE branch
            if output_mode == OutputMode.REWRITE_IN_PLACE.value:
                tests.append(f"Answers inserted under each question in `{spec.sandbox_input_path}`")
                tests.append("File contains both questions and answers in correct positions")
            elif output_mode == OutputMode.APPEND_IN_PLACE.value:
                tests.append(f"Reply appended to `{spec.sandbox_input_path}` beneath original question")
                tests.append("File contains both question and answer")
            elif output_mode == OutputMode.SEPARATE_REPLY_FILE.value:
                tests.append(f"Reply written to `{spec.sandbox_output_path}`")
                tests.append("reply.txt file exists and contains expected content")
            else:  # CHAT_ONLY or None
                tests.append("Reply presented in chat (no file modification)")
        else:
            tests = [
                "Target file was discovered in sandbox",
                "File content was read and parsed correctly",
                "Reply was generated based on file content",
                "Output delivered per detected mode (chat/file)",
            ]
        return tests
    
    # v1.12: GAME DOMAIN
    if "game" in detected_domains:
        logger.info("[step_derivation] v1.12 GAME domain detected - using game tests")
        tests = [
            "Game starts and displays initial state correctly",
            "Game board/playfield renders with correct dimensions",
            "Player input controls respond correctly (keyboard/touch)",
            "Game pieces/entities move and behave as expected",
            "Collision detection works correctly",
            "Scoring updates correctly on valid actions",
            "Game over condition triggers at correct time",
            "Level progression works (if applicable)",
            "Pause/resume functionality works",
            "Game is playable and fun to use",
        ]
        return tests
    
    if "mobile_app" in detected_domains:
        tests.append("App starts and displays main screen within 2 seconds")
        tests.append("Shift start/stop logs timestamp correctly to local storage")
        
        input_mode = resolved.get("input_mode_v1", "")
        if "voice" in input_mode.lower():
            tests.append("Voice input correctly transcribes test phrases")
        if "manual" in input_mode.lower() or "screenshot" in input_mode.lower():
            tests.append("Manual entry form accepts and validates input correctly")
        
        ocr_scope = resolved.get("ocr_scope_v1", "")
        if ocr_scope:
            if "finish tour" in ocr_scope.lower() or "completed parcels" in ocr_scope.lower():
                tests.append("Screenshot OCR extracts 'Successfully Completed Parcels' from test Finish Tour screenshot")
            elif "multiple" in ocr_scope.lower():
                tests.append("Screenshot OCR extracts parcel counts from multiple screen format test images")
        
        tests.append("Data persists across app restart (encrypted storage verified)")
        
        sync_behaviour = resolved.get("sync_behaviour", "")
        sync_target = resolved.get("sync_target", "")
        if "local-only" not in sync_behaviour.lower() and "local only" not in sync_behaviour.lower():
            if "export" in sync_target.lower() or "file" in sync_target.lower():
                tests.append("Export functionality produces valid file for ASTRA import")
            elif "endpoint" in sync_target.lower() or "live" in sync_target.lower():
                tests.append("Sync successfully transfers data to ASTRA endpoint")
        
        tests.append("Pay calculation correctly computes gross from parcel count (rate × parcels)")
        tests.append("Net profit calculation correctly subtracts fuel and wear & tear")
        tests.append("End-of-week summary shows correct totals for parcels and pay")
        tests.append("App functions fully offline (no network required for core features)")
        tests.append("No sensitive data exposed in logs or debug output")
        
    else:
        # Generic tests for unknown domains
        tests = [
            "Core functionality works as specified",
            "Error handling covers expected failure modes",
            "Performance meets requirements",
            "Security review passes",
            "Documentation is complete and accurate",
        ]
    
    return tests
