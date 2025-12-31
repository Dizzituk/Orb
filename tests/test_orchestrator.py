# FILE: tests/test_orchestrator.py
"""
Orb Test Orchestrator
=====================
Modular test runner that reads from test_registry.json.

Usage:
    python tests/test_orchestrator.py                    # Run ALL subsystems
    python tests/test_orchestrator.py overwatcher        # Run one subsystem
    python tests/test_orchestrator.py overwatcher sandbox # Run multiple
    python tests/test_orchestrator.py --list             # List available subsystems
    python tests/test_orchestrator.py --json             # Output JSON report
    python tests/test_orchestrator.py --dry-run          # Show what would run
    python tests/test_orchestrator.py --brief            # Compact summary only
"""

import subprocess
import sys
import json
import time
import argparse
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class IndividualTestResult:
    """Result for a single test function."""
    name: str
    status: str  # PASSED, FAILED, SKIPPED, ERROR
    duration_ms: float = 0.0
    error_detail: str = ""


@dataclass
class TestFileResult:
    """Result from running a single test file."""
    file: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_ms: float = 0.0
    exit_code: int = 0
    output: str = ""
    error_output: str = ""
    individual_tests: List[IndividualTestResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and self.failed == 0 and self.errors == 0
    
    @property
    def total_tests(self) -> int:
        return self.passed + self.failed + self.skipped + self.errors


@dataclass
class SubsystemResult:
    """Aggregated result for a subsystem."""
    name: str
    description: str
    test_results: List[TestFileResult] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def total_passed(self) -> int:
        return sum(t.passed for t in self.test_results)

    @property
    def total_failed(self) -> int:
        return sum(t.failed for t in self.test_results)

    @property
    def total_skipped(self) -> int:
        return sum(t.skipped for t in self.test_results)

    @property
    def total_errors(self) -> int:
        return sum(t.errors for t in self.test_results)

    @property
    def total_tests(self) -> int:
        return sum(t.total_tests for t in self.test_results)

    @property
    def success(self) -> bool:
        return all(t.success for t in self.test_results)

    @property
    def status_icon(self) -> str:
        return "✓" if self.success else "✗"


@dataclass
class OrchestratorReport:
    """Full test run report."""
    timestamp: str
    subsystems_run: List[str]
    results: List[SubsystemResult] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def all_passed(self) -> bool:
        return all(s.success for s in self.results)

    @property
    def summary(self) -> dict:
        return {
            "total_subsystems": len(self.results),
            "passed_subsystems": sum(1 for s in self.results if s.success),
            "failed_subsystems": sum(1 for s in self.results if not s.success),
            "total_tests": sum(s.total_tests for s in self.results),
            "total_passed": sum(s.total_passed for s in self.results),
            "total_failed": sum(s.total_failed for s in self.results),
            "total_skipped": sum(s.total_skipped for s in self.results),
            "duration_seconds": round(self.total_duration_ms / 1000, 2),
        }


# ============================================================================
# Registry Loader
# ============================================================================

def load_registry(registry_path: Path) -> dict:
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    with open(registry_path, "r") as f:
        return json.load(f)


def get_subsystem_tests(registry: dict, subsystem: str) -> tuple[str, list[str]]:
    if subsystem not in registry["subsystems"]:
        raise ValueError(f"Unknown subsystem: {subsystem}")
    info = registry["subsystems"][subsystem]
    return info.get("description", ""), info.get("test_files", [])


def list_subsystems(registry: dict) -> list[tuple[str, str, int]]:
    result = []
    for name, info in registry["subsystems"].items():
        desc = info.get("description", "")
        count = len(info.get("test_files", []))
        result.append((name, desc, count))
    return sorted(result)


# ============================================================================
# Test Runner
# ============================================================================

def parse_pytest_output(output: str) -> tuple[int, int, int, int]:
    passed = failed = skipped = errors = 0
    for line in output.split("\n"):
        line_lower = line.lower()
        if "passed" in line_lower or "failed" in line_lower:
            match = re.search(r"(\d+)\s+passed", line_lower)
            if match:
                passed = int(match.group(1))
            match = re.search(r"(\d+)\s+failed", line_lower)
            if match:
                failed = int(match.group(1))
            match = re.search(r"(\d+)\s+skipped", line_lower)
            if match:
                skipped = int(match.group(1))
            match = re.search(r"(\d+)\s+error", line_lower)
            if match:
                errors = int(match.group(1))
    return passed, failed, skipped, errors


def parse_individual_tests(output: str) -> List[IndividualTestResult]:
    tests = []
    pattern = r'^(.*?::\S+)\s+(PASSED|FAILED|SKIPPED|ERROR)\s*(?:\[.*?\])?\s*$'
    for line in output.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            name = match.group(1)
            status = match.group(2)
            short_name = name.split("::")[-2:] if "::" in name else [name]
            short_name = "::".join(short_name)
            tests.append(IndividualTestResult(name=short_name, status=status))
    return tests


def extract_failures_section(output: str) -> str:
    lines = output.split("\n")
    in_failures = False
    failures = []
    for line in lines:
        if "FAILURES" in line and "=" in line:
            in_failures = True
            failures.append(line)
        elif in_failures:
            if line.startswith("=") and ("short test summary" in line.lower() or "passed" in line.lower() or "warnings" in line.lower()):
                break
            failures.append(line)
    return "\n".join(failures) if failures else ""


def run_test_file(test_file: Path, project_root: Path, verbose: bool = False) -> TestFileResult:
    import os
    start = time.time()
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=project_root,
            env=env,
        )
        duration_ms = (time.time() - start) * 1000
        passed, failed, skipped, errors = parse_pytest_output(result.stdout)
        individual_tests = parse_individual_tests(result.stdout)
        return TestFileResult(
            file=test_file.name,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration_ms=duration_ms,
            exit_code=result.returncode,
            output=result.stdout,
            error_output=result.stderr,
            individual_tests=individual_tests,
        )
    except subprocess.TimeoutExpired:
        return TestFileResult(
            file=test_file.name,
            errors=1,
            duration_ms=(time.time() - start) * 1000,
            exit_code=-1,
            error_output="TIMEOUT: Test exceeded 180 second limit",
        )
    except Exception as e:
        return TestFileResult(
            file=test_file.name,
            errors=1,
            duration_ms=(time.time() - start) * 1000,
            exit_code=-1,
            error_output=f"ERROR: {str(e)}",
        )


def run_subsystem(subsystem: str, registry: dict, tests_dir: Path, project_root: Path, verbose: bool = False) -> SubsystemResult:
    description, test_files = get_subsystem_tests(registry, subsystem)
    result = SubsystemResult(name=subsystem, description=description)
    start = time.time()
    for test_file in test_files:
        test_path = tests_dir / test_file
        if not test_path.exists():
            result.test_results.append(TestFileResult(
                file=test_file,
                errors=1,
                error_output=f"Test file not found: {test_path}",
            ))
            continue
        file_result = run_test_file(test_path, project_root, verbose)
        result.test_results.append(file_result)
    result.duration_ms = (time.time() - start) * 1000
    return result


# ============================================================================
# Output Formatters
# ============================================================================

def print_subsystem_result(result: SubsystemResult, brief: bool = False):
    icon = result.status_icon
    status = "PASS" if result.success else "FAIL"
    
    print()
    print(f"╔{'═'*68}╗")
    print(f"║ {icon} [{status}] {result.name}: {result.description[:50]:<50} ║")
    print(f"╚{'═'*68}╝")
    
    if brief:
        for tr in result.test_results:
            file_icon = "✓" if tr.success else "✗"
            print(f"  {file_icon} {tr.file}: {tr.passed} passed, {tr.failed} failed, {tr.skipped} skipped")
    else:
        test_num = 0
        total_tests = result.total_tests
        
        for tr in result.test_results:
            print(f"\n  ┌─ {tr.file} {'─'*(55-len(tr.file))}")
            
            if tr.individual_tests:
                for test in tr.individual_tests:
                    test_num += 1
                    pct = int((test_num / total_tests) * 100) if total_tests > 0 else 0
                    
                    if test.status == "PASSED":
                        icon = "✓"
                    elif test.status == "FAILED":
                        icon = "✗"
                    elif test.status == "SKIPPED":
                        icon = "⊘"
                    else:
                        icon = "!"
                    
                    name = test.name[:45] + "..." if len(test.name) > 48 else test.name
                    print(f"  │ {icon} {name:<48} {test.status:>7} [{pct:3d}%]")
            else:
                file_icon = "✓" if tr.success else "✗"
                print(f"  │ {file_icon} {tr.passed} passed, {tr.failed} failed, {tr.skipped} skipped")
            
            print(f"  └{'─'*60} ({tr.duration_ms:.0f}ms)")
            
            if not tr.success:
                failures = extract_failures_section(tr.output)
                if failures:
                    print()
                    print(f"  ┌─ FAILURE DETAILS {'─'*42}")
                    for line in failures.split("\n"):
                        if line.strip():
                            print(f"  │ {line}")
                    print(f"  └{'─'*60}")
    
    print()
    total_icon = "✓" if result.success else "✗"
    print(f"  {total_icon} Total: {result.total_passed} passed, {result.total_failed} failed, {result.total_skipped} skipped ({result.duration_ms/1000:.1f}s)")


def print_summary(report: OrchestratorReport):
    s = report.summary
    
    print()
    print(f"╔{'═'*68}╗")
    print(f"║{'FINAL SUMMARY':^68}║")
    print(f"╠{'═'*68}╣")
    print(f"║  Timestamp:  {report.timestamp:<53} ║")
    print(f"║  Duration:   {s['duration_seconds']}s{' '*(55-len(str(s['duration_seconds'])))} ║")
    print(f"╠{'═'*68}╣")
    print(f"║  Subsystems: {s['passed_subsystems']}/{s['total_subsystems']} passed{' '*50} ║")
    print(f"║  Tests:      {s['total_passed']}/{s['total_tests']} passed{' '*50} ║")
    
    if s['total_failed'] > 0:
        print(f"╠{'═'*68}╣")
        print(f"║  ⚠ FAILURES: {s['total_failed']} test(s) failed{' '*43} ║")
        for sub in report.results:
            if not sub.success:
                msg = f"     - {sub.name}: {sub.total_failed} failed"
                print(f"║{msg:<68}║")
    
    if s['total_skipped'] > 0:
        print(f"║  ⏭ SKIPPED: {s['total_skipped']} test(s){' '*47} ║")
    
    print(f"╠{'═'*68}╣")
    if report.all_passed:
        print(f"║{'✅ ALL TESTS PASSED':^68}║")
    else:
        print(f"║{'❌ SOME TESTS FAILED':^68}║")
    print(f"╚{'═'*68}╝")


def to_json_report(report: OrchestratorReport) -> str:
    data = {
        "timestamp": report.timestamp,
        "total_duration_ms": report.total_duration_ms,
        "all_passed": report.all_passed,
        "summary": report.summary,
        "subsystems": [],
    }
    for sub in report.results:
        sub_data = {
            "name": sub.name,
            "description": sub.description,
            "success": sub.success,
            "duration_ms": sub.duration_ms,
            "totals": {
                "passed": sub.total_passed,
                "failed": sub.total_failed,
                "skipped": sub.total_skipped,
                "errors": sub.total_errors,
            },
            "test_files": [
                {
                    "file": tr.file,
                    "success": tr.success,
                    "passed": tr.passed,
                    "failed": tr.failed,
                    "skipped": tr.skipped,
                    "duration_ms": tr.duration_ms,
                    "tests": [{"name": t.name, "status": t.status} for t in tr.individual_tests],
                    "failures": extract_failures_section(tr.output) if not tr.success else None,
                }
                for tr in sub.test_results
            ],
        }
        data["subsystems"].append(sub_data)
    return json.dumps(data, indent=2)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Orb Test Orchestrator - Run modular test suites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_orchestrator.py                    # Run all tests
  python test_orchestrator.py overwatcher        # Run overwatcher tests only
  python test_orchestrator.py sandbox pot_spec   # Run multiple subsystems
  python test_orchestrator.py --list             # List available subsystems
  python test_orchestrator.py --brief            # Compact output
  python test_orchestrator.py --json > report.json  # JSON output
        """,
    )
    
    parser.add_argument("subsystems", nargs="*", help="Subsystem(s) to test. If empty, runs all.")
    parser.add_argument("--list", "-l", action="store_true", help="List available subsystems and exit")
    parser.add_argument("--json", "-j", action="store_true", help="Output JSON report instead of text")
    parser.add_argument("--brief", "-b", action="store_true", help="Brief output - just file summaries")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    parser.add_argument("--registry", type=Path, default=None, help="Path to test_registry.json")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    tests_dir = script_dir
    
    if args.registry:
        registry_path = args.registry
    else:
        candidates = [
            tests_dir / "test_registry.json",
            project_root / "test_registry.json",
            project_root / "tests" / "test_registry.json",
        ]
        registry_path = None
        for c in candidates:
            if c.exists():
                registry_path = c
                break
        if not registry_path:
            print("ERROR: Could not find test_registry.json")
            sys.exit(1)
    
    registry = load_registry(registry_path)
    
    if args.list:
        print()
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║                     AVAILABLE SUBSYSTEMS                             ║")
        print("╠══════════════════════════════════════════════════════════════════════╣")
        for name, desc, count in list_subsystems(registry):
            print(f"║  {name:18} {count:3} tests   {desc[:40]:<40} ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        if registry.get("future_subsystems"):
            print("\nPlanned (not yet implemented):")
            for name in registry["future_subsystems"]:
                if name != "_comment":
                    print(f"  ○ {name}")
        sys.exit(0)
    
    all_subsystems = list(registry["subsystems"].keys())
    
    if args.subsystems:
        for s in args.subsystems:
            if s not in all_subsystems:
                print(f"ERROR: Unknown subsystem '{s}'")
                print(f"Available: {', '.join(all_subsystems)}")
                sys.exit(1)
        subsystems_to_run = args.subsystems
    else:
        subsystems_to_run = all_subsystems
    
    if args.dry_run:
        print("Would run the following tests:")
        for sub in subsystems_to_run:
            desc, files = get_subsystem_tests(registry, sub)
            print(f"  {sub}: {desc}")
            for f in files:
                print(f"    - {f}")
        sys.exit(0)
    
    if not args.json:
        print()
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║                      ORB TEST ORCHESTRATOR                           ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
    
    report = OrchestratorReport(
        timestamp=datetime.now().isoformat(),
        subsystems_run=subsystems_to_run,
    )
    
    total_start = time.time()
    
    for subsystem in subsystems_to_run:
        if not args.json:
            print(f"\n▶ Running {subsystem}...")
        result = run_subsystem(
            subsystem=subsystem,
            registry=registry,
            tests_dir=tests_dir,
            project_root=project_root,
            verbose=not args.brief,
        )
        report.results.append(result)
        if not args.json:
            print_subsystem_result(result, brief=args.brief)
    
    report.total_duration_ms = (time.time() - total_start) * 1000
    
    if args.json:
        print(to_json_report(report))
    else:
        print_summary(report)
    
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
