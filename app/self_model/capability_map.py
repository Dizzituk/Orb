from __future__ import annotations

import ast
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.self_model.models import CapabilityEntry

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MAIN_FILE = _PROJECT_ROOT / "main.py"

_CAPABILITY_BLUEPRINTS: List[CapabilityEntry] = [
    CapabilityEntry(domain="chat", name="Conversational AI", description="Multi-model chat with routing, memory injection, and context awareness", module_path="app/llm", external_services=["OpenAI", "Google Gemini", "Ollama"]),
    CapabilityEntry(domain="builds", name="Code Generation Pipeline", description="Spec-gated code generation with Weaver, SpecGate, scaffold engine, and agentic builder", module_path="app/builds", external_services=["OpenAI GPT-5.4"]),
    CapabilityEntry(domain="finance", name="Finance & Accounting", description="Transaction tracking, bank imports, tax calculations, HMRC compliance, van costs, credit cards", module_path="app/finance", external_services=["Google Drive"]),
    CapabilityEntry(domain="content", name="Content Production", description="Video pipeline, script writing, thumbnail generation, shorts creation, and publishing", module_path="app/content", external_services=["YouTube API", "D-ID", "HeyGen", "Pexels", "Pixabay", "fal.ai"]),
    CapabilityEntry(domain="investments", name="Investment Tracking", description="Portfolio snapshots, market data, crypto tracking, and investment chat", module_path="app/investments", external_services=["Trading 212", "CoinMarketCap"]),
    CapabilityEntry(domain="lifestyle", name="Lifestyle & Fitness", description="Workout tracking, nutrition logging, and fitness planning", module_path="app/lifestyle"),
    CapabilityEntry(domain="memory", name="Multi-Tier Memory", description="Tiered memory with write path, read path, extraction, and retrieval", module_path="app/memory"),
    CapabilityEntry(domain="bridge", name="Astra Bridge", description="Bridge services that connect ASTRA to external and companion surfaces", module_path="app/bridge", external_services=["Tailscale"]),
    CapabilityEntry(domain="email", name="Email", description="IMAP reading and SMTP sending via Proton Bridge", module_path="app/email_service", external_services=["Proton Bridge"]),
    CapabilityEntry(domain="cloud", name="Cloud & Drive", description="Drive sync, cloud file management, and remote storage workflows", module_path="app/cloud", external_services=["Google Drive", "rclone"]),
    CapabilityEntry(domain="optimize", name="Optimizer", description="Closed-loop optimisation with evidence-based passes and scope control", module_path="app/optimize"),
    CapabilityEntry(domain="self_model", name="Self-Model & Evolution", description="Capabilities, user understanding, pattern observation, suggestions, and journal", module_path="app/self_model"),
    CapabilityEntry(domain="education", name="Education", description="Course scraping and learning resource management", module_path="app/education"),
    CapabilityEntry(domain="briefing", name="Daily Briefing", description="Scheduled briefings with personalised content compilation", module_path="app/briefing"),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _RouterIncludeVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self._aliases: Dict[str, str] = {}
        self.included_modules: set[str] = set()

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not node.module:
            return
        for alias in node.names:
            local_name = alias.asname or alias.name
            self._aliases[local_name] = f"{node.module}.{alias.name}"

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "include_router" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Name):
                import_path = self._aliases.get(arg.id, "")
                module = import_path.rsplit(".", 1)[0] if import_path else ""
                if module:
                    self.included_modules.add(module.replace(".", "/"))
        self.generic_visit(node)


class CapabilityMap:
    def __init__(self) -> None:
        self._capabilities: Dict[str, CapabilityEntry] = {}
        self._last_refresh: Optional[str] = None
        self.refresh()

    def refresh(self) -> Dict[str, Any]:
        previous = {key: entry.status for key, entry in self._capabilities.items()}
        included_modules = self._discover_included_router_modules()
        refreshed: Dict[str, CapabilityEntry] = {}
        now = _utc_now()
        for blueprint in _CAPABILITY_BLUEPRINTS:
            entry = replace(blueprint)
            entry.status = self._infer_status(entry.module_path, included_modules)
            entry.last_verified = now
            refreshed[self._make_key(entry.domain, entry.name)] = entry
        self._capabilities = refreshed
        self._last_refresh = now
        self._log_status_changes(previous)
        return self.summary()

    def _discover_included_router_modules(self) -> set[str]:
        if not _MAIN_FILE.exists():
            return set()
        try:
            tree = ast.parse(_MAIN_FILE.read_text(encoding="utf-8"))
            visitor = _RouterIncludeVisitor()
            visitor.visit(tree)
            return visitor.included_modules
        except Exception as exc:
            logger.warning("[self_model] Failed to parse main.py for router includes: %s", exc)
            return set()

    def _infer_status(self, module_path: str, included_modules: set[str]) -> str:
        module_fs_path = _PROJECT_ROOT / module_path
        if not module_fs_path.exists():
            return "missing"
        router_file = module_fs_path / "router.py"
        mounted = module_path in included_modules or f"{module_path}/router" in included_modules
        if router_file.exists() and mounted:
            return "active"
        if router_file.exists() and not mounted:
            return "in_development"
        if any(module_fs_path.rglob("*.py")):
            return "partial"
        return "stubbed"

    def _log_status_changes(self, previous: Dict[str, str]) -> None:
        try:
            from app.self_model.journal import get_journal
        except Exception:
            get_journal = None
        for key, entry in self._capabilities.items():
            old_status = previous.get(key)
            if old_status and old_status != entry.status:
                logger.info("[self_model] Capability status changed: %s %s -> %s", key, old_status, entry.status)
                if get_journal:
                    get_journal().record_capability_change(f"Capability status changed: {entry.name}", f"{old_status} -> {entry.status}")

    @staticmethod
    def _make_key(domain: str, name: str) -> str:
        return f"{domain}:{name}"

    def get_all(self) -> List[CapabilityEntry]:
        return sorted(self._capabilities.values(), key=lambda item: (item.domain, item.name))

    def get_by_domain(self, domain: str) -> List[CapabilityEntry]:
        return [c for c in self.get_all() if c.domain == domain]

    def get_domains(self) -> List[str]:
        return sorted({c.domain for c in self._capabilities.values()})

    def get_active(self) -> List[CapabilityEntry]:
        return [c for c in self._capabilities.values() if c.status == "active"]

    def get_in_development(self) -> List[CapabilityEntry]:
        return [c for c in self._capabilities.values() if c.status == "in_development"]

    def can_do(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower().strip()
        matches: List[tuple[int, CapabilityEntry]] = []
        for cap in self._capabilities.values():
            haystacks = [cap.name.lower(), cap.domain.lower(), cap.description.lower(), " ".join(cap.external_services).lower()]
            score = sum(3 for hay in haystacks if query_lower and query_lower in hay)
            score += sum(1 for word in query_lower.split() if any(word in hay for hay in haystacks))
            if score > 0:
                matches.append((score, cap))
        matches.sort(key=lambda item: item[0], reverse=True)
        if not matches:
            return {"found": False, "answer": f"I do not currently have a verified capability matching '{query}'. That may be something worth building."}
        best = matches[0][1]
        if best.status == "active":
            answer = f"Yes — {best.description}. This capability is currently active."
        elif best.status == "in_development":
            answer = f"Partly — {best.description}. The code exists but it is not fully mounted yet."
        elif best.status == "partial":
            answer = f"Partly — I have code for {best.name}, but it is only partially wired."
        else:
            answer = f"Not reliably — I have a trace of {best.name}, but its current status is {best.status}."
        return {"found": True, "answer": answer, "capability": best.to_dict()}

    def summary(self) -> Dict[str, Any]:
        capabilities = self.get_all()
        return {
            "total_capabilities": len(capabilities),
            "active": len([c for c in capabilities if c.status == "active"]),
            "in_development": len([c for c in capabilities if c.status == "in_development"]),
            "partial": len([c for c in capabilities if c.status == "partial"]),
            "domains": self.get_domains(),
            "last_refresh": self._last_refresh,
        }

    def to_plain_language(self) -> str:
        lines = ["Here is what I can currently verify about myself:\n"]
        for domain in self.get_domains():
            lines.append(f"**{domain.title()}**")
            for cap in self.get_by_domain(domain):
                services = f" External services: {', '.join(cap.external_services)}." if cap.external_services else ""
                lines.append(f"- **{cap.name}** [{cap.status.replace('_', ' ')}]: {cap.description}.{services}")
            lines.append("")
        return "\n".join(lines)


_capability_map: Optional[CapabilityMap] = None


def get_capability_map() -> CapabilityMap:
    global _capability_map
    if _capability_map is None:
        _capability_map = CapabilityMap()
    return _capability_map


def refresh_capabilities() -> Dict[str, Any]:
    cap_map = get_capability_map()
    summary = cap_map.refresh()
    logger.info("[self_model] Capability map refreshed: %s", summary)
    return summary
