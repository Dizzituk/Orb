# FILE: app/self_model/capability_map.py
"""
Pillar 1: Capability Map

Maintains a living map of what ASTRA can do. Built from architecture
scans and updated after build promotions. Provides honest answers
about what is active, stubbed, in development, or broken.

The capability map is not a static document — it refreshes from the
actual codebase and should always reflect the real state of the system.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.self_model.models import CapabilityEntry

logger = logging.getLogger(__name__)


class CapabilityMap:
    """Living capability map for ASTRA."""

    def __init__(self) -> None:
        self._capabilities: Dict[str, CapabilityEntry] = {}
        self._last_refresh: Optional[str] = None
        self._seed_known_capabilities()

    def _seed_known_capabilities(self) -> None:
        """Seed the map with known ASTRA capabilities from the architecture."""
        known = [
            CapabilityEntry(
                domain="chat",
                name="Conversational AI",
                description="Multi-model chat with routing, memory injection, and context awareness",
                status="active",
                module_path="app/llm/routing",
                external_services=["OpenAI", "Google Gemini", "Ollama"],
            ),
            CapabilityEntry(
                domain="builds",
                name="Code Generation Pipeline",
                description="Spec-gated code generation with Weaver, SpecGate, scaffold engine, and agentic builder",
                status="active",
                module_path="app/pipeline_v2",
                external_services=["OpenAI GPT-5.4"],
            ),
            CapabilityEntry(
                domain="finance",
                name="Finance & Accounting",
                description="Transaction tracking, bank imports, tax calculations, HMRC compliance, van costs, credit cards",
                status="active",
                module_path="app/finance",
                external_services=["Google Drive"],
            ),
            CapabilityEntry(
                domain="content",
                name="Content Production",
                description="Video pipeline, script writing, thumbnail generation, shorts creation, YouTube publishing",
                status="active",
                module_path="app/content",
                external_services=["YouTube API", "D-ID", "HeyGen", "Pexels", "Pixabay", "fal.ai"],
            ),
            CapabilityEntry(
                domain="content",
                name="Social Media Management",
                description="Post scheduling, engagement scanning, algorithm strategy, multi-platform distribution",
                status="active",
                module_path="app/content/distribution",
                external_services=["Facebook", "Instagram", "TikTok", "YouTube"],
            ),
            CapabilityEntry(
                domain="investments",
                name="Investment Tracking",
                description="Portfolio snapshots, market data, crypto tracking, investment chat",
                status="active",
                module_path="app/investments",
                external_services=["Trading 212", "CoinMarketCap"],
            ),
            CapabilityEntry(
                domain="lifestyle",
                name="Lifestyle & Fitness",
                description="Workout tracking, nutrition logging, fitness planning",
                status="active",
                module_path="app/lifestyle",
            ),
            CapabilityEntry(
                domain="memory",
                name="Multi-Tier Memory",
                description="Tiered memory with write path, read path, intelligent memory, experience extraction, and RAG",
                status="active",
                module_path="app/memory",
            ),
            CapabilityEntry(
                domain="bridge",
                name="AstraBridge (Android)",
                description="Android companion app with voice chat, connection management, and desktop sync",
                status="active",
                module_path="app/bridge",
                external_services=["Tailscale"],
            ),
            CapabilityEntry(
                domain="voice",
                name="Voice Interface",
                description="Speech-to-text via Faster Whisper, text-to-speech with 63+ British voices, wake word detection",
                status="active",
                module_path="app/voice",
                external_services=["Google Cloud TTS"],
            ),
            CapabilityEntry(
                domain="email",
                name="Email",
                description="IMAP reading and SMTP sending via Proton Bridge",
                status="active",
                module_path="app/email_service",
                external_services=["Proton Bridge"],
            ),
            CapabilityEntry(
                domain="cloud",
                name="Cloud & Drive",
                description="Google Drive sync, rclone integration, file management",
                status="active",
                module_path="app/cloud",
                external_services=["Google Drive", "rclone"],
            ),
            CapabilityEntry(
                domain="optimize",
                name="Optimizer",
                description="Recursive closed-loop optimisation with evidence-based passes, scope management, and trust levels",
                status="in_development",
                module_path="app/optimize",
            ),
            CapabilityEntry(
                domain="self_model",
                name="Self-Model & Evolution",
                description="Self-awareness of capabilities, user understanding, pattern observation, and suggestion engine",
                status="in_development",
                module_path="app/self_model",
            ),
            CapabilityEntry(
                domain="education",
                name="Education",
                description="Course scraping, learning resource management",
                status="active",
                module_path="app/education",
            ),
            CapabilityEntry(
                domain="briefing",
                name="Daily Briefing",
                description="Scheduled audio briefings with personalised content compilation",
                status="active",
                module_path="app/briefing",
            ),
        ]
        now = datetime.now(timezone.utc).isoformat()
        for cap in known:
            cap.last_verified = now
            key = f"{cap.domain}:{cap.name}"
            self._capabilities[key] = cap

    def get_all(self) -> List[CapabilityEntry]:
        return list(self._capabilities.values())

    def get_by_domain(self, domain: str) -> List[CapabilityEntry]:
        return [c for c in self._capabilities.values() if c.domain == domain]

    def get_domains(self) -> List[str]:
        return sorted(set(c.domain for c in self._capabilities.values()))

    def get_active(self) -> List[CapabilityEntry]:
        return [c for c in self._capabilities.values() if c.status == "active"]

    def get_in_development(self) -> List[CapabilityEntry]:
        return [c for c in self._capabilities.values() if c.status == "in_development"]

    def can_do(self, query: str) -> Dict[str, Any]:
        """Check if ASTRA can do something based on a natural language query."""
        query_lower = query.lower()
        matches = []
        for cap in self._capabilities.values():
            score = 0
            if query_lower in cap.description.lower():
                score += 3
            if query_lower in cap.domain.lower():
                score += 2
            if query_lower in cap.name.lower():
                score += 2
            for word in query_lower.split():
                if word in cap.description.lower():
                    score += 1
                if any(word in svc.lower() for svc in cap.external_services):
                    score += 1
            if score > 0:
                matches.append((score, cap))

        matches.sort(key=lambda x: x[0], reverse=True)
        if not matches:
            return {"found": False, "answer": f"I don't currently have a capability that matches '{query}'. This could be something worth building."}

        best = matches[0][1]
        if best.status == "active":
            return {"found": True, "answer": f"Yes — {best.description}. This is fully active.", "capability": best.to_dict()}
        elif best.status == "in_development":
            return {"found": True, "answer": f"I have this in development: {best.description}. It's not fully wired up yet.", "capability": best.to_dict()}
        else:
            return {"found": True, "answer": f"I have a module for this ({best.name}) but its status is: {best.status}.", "capability": best.to_dict()}

    def update_status(self, domain: str, name: str, status: str) -> Optional[CapabilityEntry]:
        key = f"{domain}:{name}"
        cap = self._capabilities.get(key)
        if cap:
            cap.status = status
            cap.last_verified = datetime.now(timezone.utc).isoformat()
        return cap

    def add_capability(self, entry: CapabilityEntry) -> None:
        key = f"{entry.domain}:{entry.name}"
        entry.last_verified = datetime.now(timezone.utc).isoformat()
        self._capabilities[key] = entry

    def summary(self) -> Dict[str, Any]:
        all_caps = self.get_all()
        return {
            "total_capabilities": len(all_caps),
            "active": len([c for c in all_caps if c.status == "active"]),
            "in_development": len([c for c in all_caps if c.status == "in_development"]),
            "domains": self.get_domains(),
            "last_refresh": self._last_refresh,
        }

    def to_plain_language(self) -> str:
        lines = ["Here is what I can currently do:\n"]
        for domain in self.get_domains():
            caps = self.get_by_domain(domain)
            for cap in caps:
                status_label = "ready" if cap.status == "active" else cap.status.replace("_", " ")
                services = f" (uses {', '.join(cap.external_services)})" if cap.external_services else ""
                lines.append(f"- **{cap.name}** [{status_label}]: {cap.description}{services}")
        return "\n".join(lines)


# ── Singleton ─────────────────────────────────────────────

_capability_map: Optional[CapabilityMap] = None


def get_capability_map() -> CapabilityMap:
    global _capability_map
    if _capability_map is None:
        _capability_map = CapabilityMap()
    return _capability_map


def refresh_capabilities() -> Dict[str, Any]:
    """Trigger a capability refresh from the current architecture state."""
    cap_map = get_capability_map()
    cap_map._last_refresh = datetime.now(timezone.utc).isoformat()
    logger.info("[self_model] Capability map refreshed: %s", cap_map.summary())
    return cap_map.summary()
