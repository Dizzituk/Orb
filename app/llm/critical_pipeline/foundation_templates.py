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
from app.llm.critical_pipeline._foundation_templates_utils import FOUNDATION_TEMPLATES_BUILD_ID, FoundationTemplate, MatchedTemplates, _register, match_templates

logger = logging.getLogger(__name__)
print(f"[FOUNDATION_TEMPLATES_LOADED] BUILD_ID={FOUNDATION_TEMPLATES_BUILD_ID}")


# =============================================================================
# TEMPLATE SCHEMA
# =============================================================================


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

_REGISTRY: List[FoundationTemplate] = []


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


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "FoundationTemplate",
    "MatchedTemplates",
    "match_templates",
    "FOUNDATION_TEMPLATES_BUILD_ID",
]
