# FILE: app/llm/critical_pipeline/foundation_templates.py
"""
Foundation Templates — Pre-Validated Architectural Patterns.

Phase 4B of Pipeline Evolution.

Provides pre-validated architectural patterns for common foundations that
greenfield projects almost always need: authentication, data persistence,
state management, API structure, error handling, and configuration.

When SpecGate detects a greenfield CREATE job, the template matcher identifies
which foundations are relevant based on tech stack and spec concepts. Matching
templates are injected into the architecture prompt as reference patterns.

This means the architecture LLM doesn't have to reinvent standard patterns —
it gets a tested skeleton to build from. The templates are:
    - Opinionated but not rigid (the LLM can deviate with good reason)
    - Framework-specific (FastAPI vs Express, React vs Vue, etc.)
    - Pre-validated (the patterns have been tested to work together)

Templates are stored as structured Python dicts, not as files on disk. This
keeps them versioned with the code and avoids filesystem dependencies.

v1.0 (2026-02-10): Initial implementation — Phase 4B.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

FOUNDATION_TEMPLATES_BUILD_ID = "2026-02-10-v1.0-foundation-templates"
print(f"[FOUNDATION_TEMPLATES_LOADED] BUILD_ID={FOUNDATION_TEMPLATES_BUILD_ID}")


# =============================================================================
# TEMPLATE SCHEMA
# =============================================================================

@dataclass
class FoundationTemplate:
    """A single foundation pattern."""
    id: str                          # Unique ID (e.g. "fastapi-auth-jwt")
    name: str                        # Display name
    category: str                    # auth | persistence | state | api | error | config
    tech_tags: Set[str]              # Match triggers (e.g. {"fastapi", "python"})
    concept_tags: Set[str]           # Concept triggers (e.g. {"auth", "login", "jwt"})
    description: str                 # What this pattern provides
    pattern_markdown: str            # The actual architecture pattern
    file_patterns: List[str] = field(default_factory=list)  # Example file layout
    dependencies: List[str] = field(default_factory=list)    # Template IDs this depends on

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "file_patterns": self.file_patterns,
        }


@dataclass
class MatchedTemplates:
    """Result of template matching against a spec."""
    templates: List[FoundationTemplate] = field(default_factory=list)
    match_reasons: Dict[str, str] = field(default_factory=dict)  # template_id → reason

    @property
    def count(self) -> int:
        return len(self.templates)

    def format_for_prompt(self) -> str:
        """Format matched templates as markdown for architecture prompt injection."""
        if not self.templates:
            return ""

        sections = []
        sections.append("=" * 60)
        sections.append("FOUNDATION PATTERNS — Pre-Validated Reference Architecture")
        sections.append("=" * 60)
        sections.append("")
        sections.append(
            "The following architectural patterns are RECOMMENDED starting points "
            "for this project. They have been pre-validated to work together. "
            "You SHOULD follow these patterns unless the spec explicitly requires "
            "a different approach. If you deviate, document WHY in the architecture."
        )
        sections.append("")

        for tmpl in self.templates:
            reason = self.match_reasons.get(tmpl.id, "")
            sections.append(f"### {tmpl.name} ({tmpl.category})")
            if reason:
                sections.append(f"_Matched because: {reason}_")
            sections.append("")
            sections.append(tmpl.description)
            sections.append("")
            if tmpl.file_patterns:
                sections.append("**Suggested file layout:**")
                for fp in tmpl.file_patterns:
                    sections.append(f"  - `{fp}`")
                sections.append("")
            sections.append(tmpl.pattern_markdown)
            sections.append("")
            sections.append("---")
            sections.append("")

        sections.append("=" * 60)
        sections.append("END FOUNDATION PATTERNS")
        sections.append("=" * 60)

        return "\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "templates": [t.to_dict() for t in self.templates],
            "match_reasons": self.match_reasons,
        }


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

_REGISTRY: List[FoundationTemplate] = []


def _register(tmpl: FoundationTemplate):
    """Register a template in the global registry."""
    _REGISTRY.append(tmpl)


# -----------------------------------------------------------------------------
# AUTH PATTERNS
# -----------------------------------------------------------------------------

_register(FoundationTemplate(
    id="fastapi-auth-jwt",
    name="FastAPI JWT Authentication",
    category="auth",
    tech_tags={"fastapi", "python"},
    concept_tags={"auth", "authentication", "login", "jwt", "token", "user"},
    description="JWT-based authentication for FastAPI with password hashing and token refresh.",
    file_patterns=[
        "app/auth/router.py",
        "app/auth/service.py",
        "app/auth/models.py",
        "app/auth/schemas.py",
        "app/auth/dependencies.py",
    ],
    pattern_markdown="""\
**Service layer** (`auth/service.py`):
- `AuthService` class with `register(email, password)`, `login(email, password)`, `verify_token(token)`
- Password hashing via `passlib` with bcrypt
- JWT creation via `python-jose` with configurable expiry
- Token refresh with separate refresh token (longer expiry)

**Router** (`auth/router.py`):
- `POST /auth/register` → `AuthService.register()`
- `POST /auth/login` → returns `{access_token, refresh_token, token_type}`
- `POST /auth/refresh` → new access token from refresh token
- `GET /auth/me` → current user from token

**Dependency** (`auth/dependencies.py`):
- `get_current_user(token: str = Depends(oauth2_scheme))` — extracts + verifies JWT
- Raises `HTTPException(401)` on invalid/expired token
- Use as `Depends(get_current_user)` on protected routes

**Models** (`auth/models.py`):
- `User` SQLAlchemy model: id, email, hashed_password, created_at, is_active
- Email has unique constraint

**Schemas** (`auth/schemas.py`):
- `UserCreate(email, password)`, `UserResponse(id, email, created_at)`, `TokenResponse(access_token, refresh_token, token_type)`
""",
))

_register(FoundationTemplate(
    id="express-auth-jwt",
    name="Express JWT Authentication",
    category="auth",
    tech_tags={"express", "node", "javascript", "typescript"},
    concept_tags={"auth", "authentication", "login", "jwt", "token", "user"},
    description="JWT-based authentication for Express/Node with bcrypt password hashing.",
    file_patterns=[
        "src/auth/auth.router.ts",
        "src/auth/auth.service.ts",
        "src/auth/auth.middleware.ts",
        "src/auth/auth.types.ts",
    ],
    pattern_markdown="""\
**Service** (`auth/auth.service.ts`):
- `register(email, password)` → hash with bcrypt, store user, return tokens
- `login(email, password)` → verify bcrypt hash, sign JWT pair
- `verifyToken(token)` → decode and validate, return user payload

**Router** (`auth/auth.router.ts`):
- `POST /auth/register`, `POST /auth/login`, `POST /auth/refresh`
- `GET /auth/me` (protected)

**Middleware** (`auth/auth.middleware.ts`):
- `authMiddleware(req, res, next)` → extract Bearer token, verify, set `req.user`
- `optionalAuth(req, res, next)` → same but doesn't reject unauthenticated

**Types** (`auth/auth.types.ts`):
- `User { id, email, createdAt }`, `TokenPair { accessToken, refreshToken }`
""",
))


# -----------------------------------------------------------------------------
# PERSISTENCE PATTERNS
# -----------------------------------------------------------------------------

_register(FoundationTemplate(
    id="sqlalchemy-persistence",
    name="SQLAlchemy Database Layer",
    category="persistence",
    tech_tags={"fastapi", "python", "sqlalchemy", "postgresql", "sqlite"},
    concept_tags={"database", "db", "persistence", "sql", "model", "migration", "crud"},
    description="SQLAlchemy async database layer with session management and base model.",
    file_patterns=[
        "app/db/engine.py",
        "app/db/session.py",
        "app/db/base_model.py",
        "app/db/migrations/",
    ],
    pattern_markdown="""\
**Engine** (`db/engine.py`):
- `create_engine()` from DATABASE_URL env var
- Async engine with `create_async_engine` for async FastAPI
- Echo mode configurable via DEBUG env var

**Session** (`db/session.py`):
- `async_session_maker` = `async_sessionmaker(engine, expire_on_commit=False)`
- `get_db()` async generator dependency for FastAPI
- Context manager pattern: `async with get_session() as session:`

**Base Model** (`db/base_model.py`):
- `BaseModel(DeclarativeBase)` with `id` (UUID primary key), `created_at`, `updated_at`
- `updated_at` auto-updates via `onupdate=func.now()`
- `to_dict()` method for serialisation
""",
    dependencies=["fastapi-config"],
))

_register(FoundationTemplate(
    id="prisma-persistence",
    name="Prisma Database Layer",
    category="persistence",
    tech_tags={"express", "node", "prisma", "typescript", "nextjs"},
    concept_tags={"database", "db", "persistence", "sql", "model", "crud"},
    description="Prisma ORM database layer with client singleton and migration workflow.",
    file_patterns=[
        "prisma/schema.prisma",
        "src/lib/prisma.ts",
    ],
    pattern_markdown="""\
**Client** (`lib/prisma.ts`):
- Singleton PrismaClient: `globalThis.__prisma ??= new PrismaClient()`
- Prevents connection exhaustion in dev (hot reload)

**Schema** (`prisma/schema.prisma`):
- `generator client { provider = "prisma-client-js" }`
- Base fields pattern: `id String @id @default(cuid())`, `createdAt DateTime @default(now())`, `updatedAt DateTime @updatedAt`
""",
))


# -----------------------------------------------------------------------------
# API STRUCTURE PATTERNS
# -----------------------------------------------------------------------------

_register(FoundationTemplate(
    id="fastapi-structure",
    name="FastAPI Project Structure",
    category="api",
    tech_tags={"fastapi", "python"},
    concept_tags={"api", "endpoint", "route", "rest", "backend", "server"},
    description="Standard FastAPI project structure with router registration, middleware, and error handling.",
    file_patterns=[
        "app/main.py",
        "app/config.py",
        "app/routers/__init__.py",
        "app/middleware/error_handler.py",
        "app/schemas/common.py",
    ],
    pattern_markdown="""\
**Entry point** (`app/main.py`):
- `app = FastAPI(title=..., version=...)` with lifespan for startup/shutdown
- Router includes: `app.include_router(feature.router, prefix="/api/v1/feature", tags=["feature"])`
- CORS middleware configured from ALLOWED_ORIGINS env var
- Global exception handler for consistent error responses

**Config** (`app/config.py`):
- Pydantic `BaseSettings` with env var loading
- Sections: database, auth, cors, feature flags
- `get_settings()` with `lru_cache` for singleton

**Error handler** (`middleware/error_handler.py`):
- Catches `HTTPException`, `RequestValidationError`, generic `Exception`
- Returns `{"detail": str, "status_code": int, "type": str}` consistently
- Logs full traceback for 500s, sanitised message to client

**Common schemas** (`schemas/common.py`):
- `ErrorResponse(detail, status_code, type)`
- `PaginatedResponse(items, total, page, page_size)`
- `HealthResponse(status, version, uptime)`
""",
))

_register(FoundationTemplate(
    id="express-structure",
    name="Express Project Structure",
    category="api",
    tech_tags={"express", "node", "typescript", "javascript"},
    concept_tags={"api", "endpoint", "route", "rest", "backend", "server"},
    description="Standard Express/TypeScript project structure with modular routing and error middleware.",
    file_patterns=[
        "src/index.ts",
        "src/config.ts",
        "src/middleware/errorHandler.ts",
        "src/routes/index.ts",
    ],
    pattern_markdown="""\
**Entry point** (`src/index.ts`):
- `const app = express()` with `app.use(express.json())`
- Route mounting: `app.use('/api/v1/feature', featureRouter)`
- Error handler last: `app.use(errorHandler)`
- Graceful shutdown on SIGTERM/SIGINT

**Config** (`src/config.ts`):
- `export const config` object reading from `process.env`
- Typed with defaults: `port: Number(process.env.PORT) || 3000`
- Validation at startup: throw if required vars missing

**Error handler** (`middleware/errorHandler.ts`):
- Express error middleware `(err, req, res, next)`
- `AppError` custom class with `statusCode`, `isOperational`
- Operational errors → client-safe message; unexpected → 500 + log
""",
))


# -----------------------------------------------------------------------------
# STATE MANAGEMENT PATTERNS
# -----------------------------------------------------------------------------

_register(FoundationTemplate(
    id="react-state-zustand",
    name="React State with Zustand",
    category="state",
    tech_tags={"react", "zustand", "typescript", "javascript", "nextjs"},
    concept_tags={"state", "store", "frontend", "ui"},
    description="Zustand-based state management with typed stores and persistence.",
    file_patterns=[
        "src/stores/useAppStore.ts",
        "src/stores/useAuthStore.ts",
    ],
    pattern_markdown="""\
**Store pattern** (`stores/useAppStore.ts`):
```
export const useAppStore = create<AppState>()(
  devtools(
    persist(
      (set, get) => ({
        // State
        isLoading: false,
        error: null,
        // Actions
        setLoading: (loading) => set({ isLoading: loading }),
        reset: () => set(initialState),
      }),
      { name: 'app-store' }
    )
  )
)
```
- Each feature gets its own store file
- Actions co-located with state (no separate action files)
- `persist` middleware for localStorage (opt-in per store)
- `devtools` middleware for Redux DevTools compatibility
""",
))

_register(FoundationTemplate(
    id="react-state-context",
    name="React State with Context + useReducer",
    category="state",
    tech_tags={"react", "typescript", "javascript"},
    concept_tags={"state", "context", "frontend", "ui"},
    description="Context + useReducer pattern for React state without external dependencies.",
    file_patterns=[
        "src/contexts/AppContext.tsx",
        "src/contexts/AuthContext.tsx",
        "src/hooks/useApp.ts",
    ],
    pattern_markdown="""\
**Context pattern** (`contexts/AppContext.tsx`):
- `AppContext` = `createContext<AppState & AppActions>()`
- `appReducer(state, action)` with typed action discriminated union
- `AppProvider` wraps children with `useReducer` + context value
- Custom hook: `useApp()` = `useContext(AppContext)` with null guard
""",
))


# -----------------------------------------------------------------------------
# CONFIG PATTERNS
# -----------------------------------------------------------------------------

_register(FoundationTemplate(
    id="fastapi-config",
    name="FastAPI Configuration",
    category="config",
    tech_tags={"fastapi", "python"},
    concept_tags={"config", "settings", "environment", "env"},
    description="Pydantic BaseSettings configuration with .env file support.",
    file_patterns=["app/config.py", ".env.example"],
    pattern_markdown="""\
**Config** (`app/config.py`):
- `class Settings(BaseSettings):` with `model_config = SettingsConfigDict(env_file=".env")`
- Grouped: `db_url: str`, `secret_key: str`, `debug: bool = False`
- `@lru_cache def get_settings() -> Settings:`
- Use via `settings = Depends(get_settings)` or import directly
""",
))

_register(FoundationTemplate(
    id="node-config",
    name="Node.js Configuration",
    category="config",
    tech_tags={"express", "node", "typescript", "javascript", "nextjs"},
    concept_tags={"config", "settings", "environment", "env"},
    description="Typed configuration with dotenv and runtime validation.",
    file_patterns=["src/config.ts", ".env.example"],
    pattern_markdown="""\
**Config** (`src/config.ts`):
- `import 'dotenv/config'` at top
- Typed config object with `z.object()` (zod) validation
- `export const config = validateConfig(process.env)`
- Throws at startup if required vars missing (fail fast)
""",
))


# -----------------------------------------------------------------------------
# ERROR HANDLING PATTERNS
# -----------------------------------------------------------------------------

_register(FoundationTemplate(
    id="python-error-handling",
    name="Python Error Hierarchy",
    category="error",
    tech_tags={"python", "fastapi"},
    concept_tags={"error", "exception", "handling"},
    description="Custom exception hierarchy for clean error handling in Python backends.",
    file_patterns=["app/errors.py"],
    pattern_markdown="""\
**Error hierarchy** (`app/errors.py`):
- `AppError(Exception)` base: `message`, `status_code`, `error_type`
- `NotFoundError(AppError)` — 404
- `ValidationError(AppError)` — 422
- `AuthenticationError(AppError)` — 401
- `AuthorisationError(AppError)` — 403
- `ConflictError(AppError)` — 409

All caught by global error handler middleware → consistent JSON response.
""",
))


# =============================================================================
# MATCHER — Find relevant templates for a spec
# =============================================================================

def match_templates(
    tech_stack: Optional[Dict[str, str]] = None,
    spec_concepts: Optional[List[str]] = None,
    spec_text: Optional[str] = None,
    max_templates: int = 5,
) -> MatchedTemplates:
    """
    Match foundation templates against a job's tech stack and concepts.

    Args:
        tech_stack: Dict with keys like frontend_framework, backend_framework, etc.
        spec_concepts: Extracted concept keywords from the spec
        spec_text: Raw spec text for keyword scanning
        max_templates: Maximum templates to return (highest scoring first)

    Returns:
        MatchedTemplates with ranked matches and reasons
    """
    tech_stack = tech_stack or {}
    spec_concepts = [c.lower() for c in (spec_concepts or [])]

    # Build a set of tech indicators from the stack
    tech_indicators: Set[str] = set()
    for key, value in tech_stack.items():
        if value:
            tech_indicators.add(value.lower())
            # Also add common aliases
            _v = value.lower()
            if "fastapi" in _v:
                tech_indicators.update({"fastapi", "python"})
            elif "express" in _v:
                tech_indicators.update({"express", "node", "javascript"})
            elif "react" in _v:
                tech_indicators.add("react")
            elif "next" in _v:
                tech_indicators.update({"nextjs", "react"})
            elif "vue" in _v:
                tech_indicators.add("vue")
            if "typescript" in _v:
                tech_indicators.add("typescript")
            if "python" in _v:
                tech_indicators.add("python")

    # Extract additional concepts from spec text
    text_concepts: Set[str] = set()
    if spec_text:
        _text_lower = spec_text.lower()
        # Scan for concept keywords
        concept_keywords = [
            "auth", "login", "register", "jwt", "token", "user",
            "database", "db", "persistence", "sql", "crud",
            "api", "endpoint", "route", "rest",
            "state", "store", "context",
            "config", "settings", "environment",
            "error", "exception", "handling",
            "migration", "schema", "model",
        ]
        for kw in concept_keywords:
            if kw in _text_lower:
                text_concepts.add(kw)

    all_concepts = set(spec_concepts) | text_concepts

    # Score each template
    scored: List[tuple] = []  # (score, template, reason)

    for tmpl in _REGISTRY:
        score = 0
        reasons = []

        # Tech match (strong signal)
        tech_overlap = tmpl.tech_tags & tech_indicators
        if tech_overlap:
            score += len(tech_overlap) * 3
            reasons.append(f"tech: {', '.join(tech_overlap)}")

        # Concept match
        concept_overlap = tmpl.concept_tags & all_concepts
        if concept_overlap:
            score += len(concept_overlap) * 2
            reasons.append(f"concepts: {', '.join(concept_overlap)}")

        # Only include if we have BOTH tech AND concept match
        # (prevents auth templates showing for non-auth projects)
        if tech_overlap and concept_overlap:
            scored.append((score, tmpl, " + ".join(reasons)))

    # Sort by score descending, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_templates]

    result = MatchedTemplates()
    for _score, tmpl, reason in top:
        result.templates.append(tmpl)
        result.match_reasons[tmpl.id] = reason

    if result.count:
        logger.info(
            "[foundation_templates] Matched %d templates: %s",
            result.count,
            [t.id for t in result.templates],
        )
    else:
        logger.debug("[foundation_templates] No templates matched")

    return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "FoundationTemplate",
    "MatchedTemplates",
    "match_templates",
    "FOUNDATION_TEMPLATES_BUILD_ID",
]
