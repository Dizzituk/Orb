# FILE: tests/register_test.py
"""
Test Registry Helper
====================
Programmatically add/remove tests from the registry.

Used by Overwatcher when creating new subsystem tests.

Usage:
    python register_test.py add --subsystem memory --file test_memory.py --desc "Memory service"
    python register_test.py remove --subsystem memory --file test_memory.py
    python register_test.py promote --subsystem llm_routing  # Move from future to active
    python register_test.py show --subsystem memory
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def load_registry(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_registry(path: Path, registry: dict):
    registry["_meta"]["updated"] = datetime.now().strftime("%Y-%m-%d")
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def add_test(registry: dict, subsystem: str, test_file: str, description: str = None):
    """Add a test file to a subsystem (creates subsystem if needed)."""
    
    if subsystem not in registry["subsystems"]:
        # Create new subsystem entry
        registry["subsystems"][subsystem] = {
            "description": description or f"{subsystem} tests",
            "test_files": [],
            "dependencies": [],
        }
        print(f"Created new subsystem: {subsystem}")
    
    sub = registry["subsystems"][subsystem]
    
    if test_file in sub["test_files"]:
        print(f"Test file already registered: {test_file}")
        return False
    
    sub["test_files"].append(test_file)
    
    if description and not sub.get("description"):
        sub["description"] = description
    
    print(f"Added {test_file} to {subsystem}")
    return True


def remove_test(registry: dict, subsystem: str, test_file: str):
    """Remove a test file from a subsystem."""
    
    if subsystem not in registry["subsystems"]:
        print(f"Subsystem not found: {subsystem}")
        return False
    
    sub = registry["subsystems"][subsystem]
    
    if test_file not in sub["test_files"]:
        print(f"Test file not registered: {test_file}")
        return False
    
    sub["test_files"].remove(test_file)
    print(f"Removed {test_file} from {subsystem}")
    
    # Remove subsystem if empty
    if not sub["test_files"]:
        del registry["subsystems"][subsystem]
        print(f"Removed empty subsystem: {subsystem}")
    
    return True


def promote_subsystem(registry: dict, subsystem: str):
    """Move a subsystem from future_subsystems to active subsystems."""
    
    future = registry.get("future_subsystems", {})
    
    if subsystem not in future:
        print(f"Subsystem not in future_subsystems: {subsystem}")
        return False
    
    if subsystem in registry["subsystems"]:
        print(f"Subsystem already active: {subsystem}")
        return False
    
    # Move to active
    info = future.pop(subsystem)
    registry["subsystems"][subsystem] = {
        "description": info.get("description", ""),
        "test_files": info.get("test_files", []),
        "dependencies": [],
    }
    
    print(f"Promoted {subsystem} to active subsystems")
    return True


def show_subsystem(registry: dict, subsystem: str):
    """Show details of a subsystem."""
    
    if subsystem in registry["subsystems"]:
        print(f"\n[ACTIVE] {subsystem}")
        info = registry["subsystems"][subsystem]
    elif subsystem in registry.get("future_subsystems", {}):
        print(f"\n[FUTURE] {subsystem}")
        info = registry["future_subsystems"][subsystem]
    else:
        print(f"Subsystem not found: {subsystem}")
        return
    
    print(f"  Description: {info.get('description', 'N/A')}")
    print(f"  Test files:")
    for f in info.get("test_files", []):
        print(f"    - {f}")
    
    if info.get("source_modules"):
        print(f"  Source modules:")
        for m in info["source_modules"]:
            print(f"    - {m}")
    
    if info.get("dependencies"):
        print(f"  Dependencies: {', '.join(info['dependencies'])}")


def main():
    parser = argparse.ArgumentParser(description="Test Registry Helper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add test to subsystem")
    add_parser.add_argument("--subsystem", "-s", required=True)
    add_parser.add_argument("--file", "-f", required=True)
    add_parser.add_argument("--desc", "-d", default=None)
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove test from subsystem")
    remove_parser.add_argument("--subsystem", "-s", required=True)
    remove_parser.add_argument("--file", "-f", required=True)
    
    # Promote command
    promote_parser = subparsers.add_parser("promote", help="Promote from future to active")
    promote_parser.add_argument("--subsystem", "-s", required=True)
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show subsystem details")
    show_parser.add_argument("--subsystem", "-s", required=True)
    
    # List command
    subparsers.add_parser("list", help="List all subsystems")
    
    # Common args
    parser.add_argument("--registry", type=Path, default=None)
    
    args = parser.parse_args()
    
    # Find registry
    script_dir = Path(__file__).parent
    if args.registry:
        registry_path = args.registry
    else:
        registry_path = script_dir / "test_registry.json"
    
    if not registry_path.exists():
        print(f"Registry not found: {registry_path}")
        return 1
    
    registry = load_registry(registry_path)
    modified = False
    
    if args.command == "add":
        modified = add_test(registry, args.subsystem, args.file, args.desc)
    
    elif args.command == "remove":
        modified = remove_test(registry, args.subsystem, args.file)
    
    elif args.command == "promote":
        modified = promote_subsystem(registry, args.subsystem)
    
    elif args.command == "show":
        show_subsystem(registry, args.subsystem)
    
    elif args.command == "list":
        print("\nActive subsystems:")
        for name, info in sorted(registry["subsystems"].items()):
            count = len(info.get("test_files", []))
            print(f"  {name}: {count} test(s)")
        
        if registry.get("future_subsystems"):
            print("\nFuture subsystems (not yet implemented):")
            for name in sorted(registry["future_subsystems"].keys()):
                if name != "_comment":
                    print(f"  {name}")
    
    if modified:
        save_registry(registry_path, registry)
        print(f"\nRegistry updated: {registry_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
