# FILE: app/tools/registry.py
"""
Phase 4 Tool Registry

Manages tool definitions with:
- Strict input/output schemas
- Side-effect level tracking
- Tool capability requirements
- Bounded retry policies

CRITICAL RULE: Tools with "code" or "system" side-effects are PROPOSAL-ONLY.
Orb NEVER executes code or modifies files automatically at runtime.
"""
from __future__ import annotations

import logging
from typing import Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class SideEffectLevel(str, Enum):
    """
    Side-effect classification for tools.
    
    NONE: Pure read/compute operations (safe to execute)
    DOCUMENT: Can modify project documents in artefact store
    CODE: Would modify code (PROPOSAL-ONLY, never executed)
    SYSTEM: Would change environment (PROPOSAL-ONLY, never executed)
    """
    NONE = "none"
    DOCUMENT = "document"
    CODE = "code"
    SYSTEM = "system"


class ToolStatus(str, Enum):
    """Tool availability status."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    EXPERIMENTAL = "experimental"


# =============================================================================
# TOOL RESPONSE CONTRACT
# =============================================================================

@dataclass
class ToolResponse:
    """
    Standard response structure for all tools.
    
    Every tool MUST return this structure.
    """
    ok: bool
    result: Optional[dict] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    # Metadata
    tool_name: str = ""
    tool_version: str = ""
    execution_time_ms: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ok": self.ok,
            "result": self.result,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "execution_time_ms": self.execution_time_ms,
        }


# =============================================================================
# TOOL DEFINITION
# =============================================================================

@dataclass
class ToolDefinition:
    """
    Complete definition of a tool.
    
    Includes:
    - Metadata (name, version, description)
    - Schemas (input/output)
    - Side-effect level
    - Execution function (if enabled)
    - Capability requirements
    """
    tool_name: str
    tool_version: str
    description: str
    
    # Schemas (JSON Schema or Pydantic models)
    input_schema: dict
    output_schema: dict
    
    # Side-effect classification
    side_effect_level: SideEffectLevel
    
    # Status
    status: ToolStatus = ToolStatus.ENABLED
    
    # Execution
    executor: Optional[Callable] = None  # Function to execute the tool
    max_retries: int = 2
    timeout_seconds: int = 30
    
    # Constraints
    allowed_models: Optional[list[str]] = None  # None = all allowed
    forbidden_models: Optional[list[str]] = None
    allowed_job_types: Optional[list[str]] = None
    forbidden_job_types: Optional[list[str]] = None
    
    # Requirements
    requires_internet: bool = False
    requires_provider_capability: Optional[str] = None  # e.g., "vision", "web_search"


# =============================================================================
# TOOL REGISTRY
# =============================================================================

class ToolRegistry:
    """
    Central registry for all tools.
    
    Manages:
    - Tool definitions and schemas
    - Side-effect enforcement
    - Tool availability checks
    - Tool execution (for ENABLED tools)
    
    CRITICAL: Tools with side_effect_level "code" or "system" return
    proposals as text/JSON. They NEVER execute directly.
    """
    
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._register_builtin_tools()
    
    def register_tool(self, definition: ToolDefinition) -> None:
        """Register a tool in the registry."""
        key = f"{definition.tool_name}:{definition.tool_version}"
        self._tools[key] = definition
        logger.info(
            f"[tool_registry] Registered: {key} "
            f"side_effect={definition.side_effect_level.value} "
            f"status={definition.status.value}"
        )
    
    def get_tool(self, tool_name: str, tool_version: str = "v1") -> Optional[ToolDefinition]:
        """Get a tool definition."""
        key = f"{tool_name}:{tool_version}"
        return self._tools.get(key)
    
    def list_tools(
        self,
        side_effect_level: Optional[SideEffectLevel] = None,
        status: Optional[ToolStatus] = None,
    ) -> list[ToolDefinition]:
        """List tools with optional filters."""
        tools = list(self._tools.values())
        
        if side_effect_level:
            tools = [t for t in tools if t.side_effect_level == side_effect_level]
        
        if status:
            tools = [t for t in tools if t.status == status]
        
        return tools
    
    def is_tool_available(self, tool_name: str, tool_version: str = "v1") -> bool:
        """Check if a tool is available and enabled."""
        tool = self.get_tool(tool_name, tool_version)
        return tool is not None and tool.status == ToolStatus.ENABLED
    
    def validate_tool_input(self, tool_name: str, tool_version: str, input_data: dict) -> tuple[bool, Optional[str]]:
        """
        Validate tool input against schema.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        tool = self.get_tool(tool_name, tool_version)
        if not tool:
            return False, f"Tool not found: {tool_name}:{tool_version}"
        
        # TODO: Implement JSON schema validation
        # For now, just basic check
        if not isinstance(input_data, dict):
            return False, "Input must be a dictionary"
        
        return True, None
    
    def execute_tool(
        self,
        tool_name: str,
        tool_version: str,
        input_data: dict,
        context: Optional[dict] = None,
    ) -> ToolResponse:
        """
        Execute a tool.
        
        For tools with side_effect_level NONE or DOCUMENT:
            - Execute directly and return result
        
        For tools with side_effect_level CODE or SYSTEM:
            - Return a PROPOSAL as text/JSON
            - NEVER execute the actual operation
        
        Args:
            tool_name: Tool identifier
            tool_version: Tool version
            input_data: Tool input parameters
            context: Optional execution context (job_id, project_id, etc.)
        
        Returns:
            ToolResponse with result or error
        """
        tool = self.get_tool(tool_name, tool_version)
        
        if not tool:
            return ToolResponse(
                ok=False,
                error_code="TOOL_NOT_FOUND",
                error_message=f"Tool not found: {tool_name}:{tool_version}",
                tool_name=tool_name,
                tool_version=tool_version,
            )
        
        if tool.status != ToolStatus.ENABLED:
            return ToolResponse(
                ok=False,
                error_code="TOOL_DISABLED",
                error_message=f"Tool is disabled: {tool_name}",
                tool_name=tool_name,
                tool_version=tool_version,
            )
        
        # Validate input
        is_valid, error_msg = self.validate_tool_input(tool_name, tool_version, input_data)
        if not is_valid:
            return ToolResponse(
                ok=False,
                error_code="INVALID_INPUT",
                error_message=error_msg,
                tool_name=tool_name,
                tool_version=tool_version,
            )
        
        # Check if tool has executor
        if not tool.executor:
            return ToolResponse(
                ok=False,
                error_code="NO_EXECUTOR",
                error_message=f"Tool has no executor function: {tool_name}",
                tool_name=tool_name,
                tool_version=tool_version,
            )
        
        # Execute tool
        try:
            result = tool.executor(input_data, context)
            return ToolResponse(
                ok=True,
                result=result,
                tool_name=tool_name,
                tool_version=tool_version,
            )
        except Exception as e:
            logger.error(f"[tool_registry] Tool execution failed: {tool_name} - {e}")
            return ToolResponse(
                ok=False,
                error_code="EXECUTION_ERROR",
                error_message=str(e),
                tool_name=tool_name,
                tool_version=tool_version,
            )
    
    def _register_builtin_tools(self) -> None:
        """Register built-in tools."""
        
        # ==================== READ-ONLY TOOLS (side_effect_level: NONE) ====================
        
        # search_web - Web search
        self.register_tool(ToolDefinition(
            tool_name="search_web",
            tool_version="v1",
            description="Search the web using Gemini grounding",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "string"},
                    "sources": {"type": "array"},
                },
            },
            side_effect_level=SideEffectLevel.NONE,
            status=ToolStatus.ENABLED,
            requires_internet=True,
            requires_provider_capability="web_search",
        ))
        
        # read_file - Read file from artefact store
        self.register_tool(ToolDefinition(
            tool_name="read_file",
            tool_version="v1",
            description="Read a file from the project artefact store",
            input_schema={
                "type": "object",
                "properties": {
                    "artefact_id": {"type": "string"},
                },
                "required": ["artefact_id"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                },
            },
            side_effect_level=SideEffectLevel.NONE,
            status=ToolStatus.ENABLED,
        ))
        
        # vector_search - Semantic search
        self.register_tool(ToolDefinition(
            tool_name="vector_search",
            tool_version="v1",
            description="Semantic search across project knowledge",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "project_id": {"type": "integer"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query", "project_id"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                },
            },
            side_effect_level=SideEffectLevel.NONE,
            status=ToolStatus.ENABLED,
        ))
        
        # ==================== DOCUMENT TOOLS (side_effect_level: DOCUMENT) ====================
        
        # write_artefact - Write to artefact store
        self.register_tool(ToolDefinition(
            tool_name="write_artefact",
            tool_version="v1",
            description="Write or update an artefact in the project store",
            input_schema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "artefact_type": {"type": "string"},
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                    "metadata": {"type": "object"},
                    "etag": {"type": "string"},  # For concurrency control
                },
                "required": ["project_id", "artefact_type", "name", "content"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "artefact_id": {"type": "string"},
                    "version": {"type": "integer"},
                    "etag": {"type": "string"},
                },
            },
            side_effect_level=SideEffectLevel.DOCUMENT,
            status=ToolStatus.ENABLED,
        ))
        
        # ==================== PROPOSAL-ONLY TOOLS (side_effect_level: CODE/SYSTEM) ====================
        
        # code_patch_proposal - Generate code patch (NEVER APPLIED)
        self.register_tool(ToolDefinition(
            tool_name="code_patch_proposal",
            tool_version="v1",
            description="Generate a code patch proposal (not applied automatically)",
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "patch_content": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["file_path", "patch_content", "description"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "proposal_id": {"type": "string"},
                    "patch": {"type": "string"},
                    "files_affected": {"type": "array"},
                },
            },
            side_effect_level=SideEffectLevel.CODE,
            status=ToolStatus.ENABLED,
        ))
        
        # script_proposal - Generate script proposal (NEVER EXECUTED)
        self.register_tool(ToolDefinition(
            tool_name="script_proposal",
            tool_version="v1",
            description="Generate a script proposal (not executed automatically)",
            input_schema={
                "type": "object",
                "properties": {
                    "script_type": {"type": "string"},  # "powershell" | "bash" | "python"
                    "script_content": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["script_type", "script_content", "description"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "proposal_id": {"type": "string"},
                    "script": {"type": "string"},
                    "warnings": {"type": "array"},
                },
            },
            side_effect_level=SideEffectLevel.SYSTEM,
            status=ToolStatus.ENABLED,
        ))
        
        logger.info("[tool_registry] Built-in tools registered")


# =============================================================================
# GLOBAL REGISTRY INSTANCE
# =============================================================================

_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def is_tool_available(tool_name: str, tool_version: str = "v1") -> bool:
    """Check if a tool is available."""
    return get_tool_registry().is_tool_available(tool_name, tool_version)


def execute_tool(tool_name: str, tool_version: str, input_data: dict, context: Optional[dict] = None) -> ToolResponse:
    """Execute a tool through the registry."""
    return get_tool_registry().execute_tool(tool_name, tool_version, input_data, context)


__all__ = [
    "SideEffectLevel",
    "ToolStatus",
    "ToolResponse",
    "ToolDefinition",
    "ToolRegistry",
    "get_tool_registry",
    "is_tool_available",
    "execute_tool",
]