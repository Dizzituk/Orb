# Orb Architecture Map

**Version:** 0.5.0  
**Last Updated:** 29 November 2025  
**Purpose:** Canonical architecture reference for the Orb multi-LLM AI assistant system

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Backend (`Orb`)](#backend-orb)
4. [Desktop Client (`orb-desktop`)](#desktop-client-orb-desktop)
5. [Data Layer](#data-layer)
6. [Multi-LLM Orchestration](#multi-llm-orchestration)
7. [Future Capabilities](#future-capabilities)

---

## System Overview

Orb is a personal AI assistant built as a multi-component system with specialized LLM roles:

- **GPT (OpenAI)** — Fast/lightweight reasoning, conversational interface, linguistics
- **Claude (Anthropic)** — Flagship engineer, complex code, architecture design
- **Gemini (Google)** — Critic, reviewer, analyst, vision specialist

### Core Vision

Different AI models specialize in distinct roles while sharing a unified memory layer, enabling:
- Persistent knowledge management across conversations
- Task coordination and project organization
- Job-type-based automatic model routing
- Future file ingestion pipelines
- Mobile screen capture workflows

### Technology Stack

**Backend:**
- FastAPI (Python 3.13)
- SQLAlchemy ORM
- SQLite database
- Pydantic schemas

**Desktop Client:**
- Electron (v33.0.0)
- Vanilla JavaScript (no frameworks)
- HTML/CSS UI

**Platform:** Windows 11

---

## Component Architecture

```
D:/
├── Orb/                    # Backend (FastAPI server)
│   ├── main.py             # Application entrypoint
│   ├── chat_memory.py      # Legacy in-memory chat store (unused)
│   ├── app/
│   │   ├── db.py           # Database configuration
│   │   ├── memory/         # Memory subsystem
│   │   └── llm/            # LLM routing module (NEW)
│   ├── data/               # SQLite database storage
│   └── static/             # Web-based test UI
│
├── orb-desktop/            # Electron desktop client
│   ├── main.js             # Electron main process
│   ├── renderer.js         # Electron renderer process
│   ├── index.html          # UI structure
│   ├── styles.css          # UI styling
│   └── build/              # Build resources (icons)
│
└── orb-electron-data/      # Electron userData directory
    └── [cache, logs, settings] # NOT PART OF CODEBASE
```

---

## Backend (`Orb`)

**Location:** `D:/Orb/`  
**Framework:** FastAPI  
**Entry Point:** `main.py`

### Directory Structure

```
Orb/
├── main.py                 # FastAPI app, endpoints, context building
├── chat_memory.py          # Legacy in-memory chat store (not currently used)
├── .env                    # API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY)
├── start_orb_backend.bat   # Windows startup script
│
├── app/
│   ├── __init__.py
│   ├── db.py               # Database engine, session factory, Base
│   │
│   ├── memory/             # Memory subsystem
│   │   ├── __init__.py
│   │   ├── models.py       # SQLAlchemy ORM models
│   │   ├── schemas.py      # Pydantic request/response schemas
│   │   ├── service.py      # Business logic (CRUD operations)
│   │   └── router.py       # FastAPI route handlers
│   │
│   └── llm/                # LLM routing module (NEW in v0.5.0)
│       ├── __init__.py     # Exports: call_llm, LLMTask, LLMResult, JobType
│       ├── schemas.py      # JobType enum, LLMTask/LLMResult models, RoutingConfig
│       ├── clients.py      # Provider API wrappers (internal only)
│       └── router.py       # Main router with call_llm() function
│
├── data/
│   └── orb_memory.db       # SQLite database file
│
├── static/                 # Web-based test interface
│   ├── index.html
│   └── renderer.js
│
└── .venv/                  # Python virtual environment
```

### Core Modules

#### `main.py`

**Responsibilities:**
- FastAPI application initialization
- Database initialization on startup
- Context building from memory system
- Chat endpoint with job-type routing
- Direct LLM endpoint for testing
- Static file serving

**Key Functions:**
- `build_context_block(db, project_id)` — Fetches recent notes and open tasks

**Endpoints:**
- `GET /` — Serves static test UI
- `GET /ping` — Health check
- `POST /chat` — Main chat endpoint (routes via job_type)
- `POST /llm` — Direct LLM call (no project context, for testing)
- `GET /providers` — Check which API keys are configured
- `GET /job-types` — List all job types grouped by routing category
- `/memory/*` — All memory subsystem routes (via router)

#### `chat_memory.py`

**Status:** Legacy / Not Currently Used  
**Purpose:** In-memory chat storage with session management  
**Note:** Replaced by SQLite-based `Message` model in memory subsystem

#### `app/db.py`

**Responsibilities:**
- SQLAlchemy engine configuration
- Session factory setup
- Database initialization
- FastAPI dependency injection

**Key Components:**
- `DATABASE_URL` — Default: `sqlite:///./data/orb_memory.db`
- `engine` — SQLAlchemy engine with SQLite config
- `SessionLocal` — Session factory
- `Base` — Declarative base for ORM models
- `get_db()` — FastAPI dependency (yields session, ensures cleanup)
- `init_db()` — Creates all tables from models

---

### LLM Routing Module (`app/llm/`)

**NEW in v0.5.0** — Isolated routing layer for all LLM calls.

All other parts of Orb MUST call through this module. Never call raw provider APIs directly.

#### `schemas.py` — Types and Configuration

**`JobType` Enum:**
All recognized job types that determine routing:

| Category | Job Types |
|----------|-----------|
| GPT Only | `casual_chat`, `note_cleanup`, `copywriting`, `prompt_shaping`, `summary`, `explanation` |
| Medium Dev | `simple_code_change`, `small_bugfix` |
| Claude Primary | `complex_code_change`, `codegen_full_file`, `architecture_design`, `code_review`, `spec_review`, `refactor`, `implementation_plan` |
| High Stakes (Claude + Gemini) | `high_stakes_infra`, `security_sensitive_change`, `privacy_sensitive_change`, `public_app_packaging` |
| Gemini | `image_analysis`, `screenshot_analysis`, `video_analysis` |

**`Provider` Enum:**
- `OPENAI` — GPT models
- `ANTHROPIC` — Claude models
- `GOOGLE` — Gemini models

**`LLMTask` Model:**
```python
class LLMTask(BaseModel):
    job_type: JobType
    messages: list[dict]           # [{"role": "user", "content": "..."}]
    system_prompt: Optional[str]
    project_context: Optional[str]  # Notes + tasks context
    metadata: Optional[dict]
    force_provider: Optional[Provider]  # Override routing
```

**`LLMResult` Model:**
```python
class LLMResult(BaseModel):
    provider: Provider
    content: str
    critic_provider: Optional[Provider]  # For high-stakes: Gemini
    critic_review: Optional[str]
    job_type: JobType
    was_reviewed: bool
    usage: Optional[dict]
```

**`RoutingConfig` Class:**
Static configuration mapping job types to providers. Modify `SMART_PROVIDER` to change medium dev routing (default: Claude).

#### `clients.py` — Provider Wrappers

**Internal module — do not import directly.**

**Functions:**
- `call_openai(system_prompt, messages)` → `(content, usage)`
- `call_anthropic(system_prompt, messages)` → `(content, usage)`
- `call_google(system_prompt, messages)` → `(content, usage)`
- `check_provider_availability()` → `dict[str, bool]`

**Model Configuration:**
```python
OPENAI_MODEL = "gpt-4.1-mini"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
GEMINI_MODEL = "gemini-2.0-flash"
```

#### `router.py` — Main Router

**The single entry point for all LLM calls.**

**Main Function:**
```python
def call_llm(task: LLMTask) -> LLMResult:
    """
    Route an LLM task to the appropriate provider(s).
    
    - Determines provider based on job_type
    - Builds role-specific system prompt
    - Calls provider API
    - For high-stakes: gets Gemini critic review
    - Returns structured result
    """
```

**Convenience Functions:**
- `quick_chat(message, context)` → `str` — Always routes to GPT
- `request_code(message, context, high_stakes)` → `LLMResult` — Routes to Claude
- `review_work(content, context)` → `str` — Forces Gemini for review

**System Prompt Builders:**
- `_build_gpt_system_prompt(task)` — Concise, conversational
- `_build_claude_system_prompt(task)` — Engineering focus, full-file rules
- `_build_gemini_system_prompt(task, is_critic)` — Analytical/review focus

**Claude-Specific Rules (enforced in system prompt):**
1. Always ask for full file before modifying
2. Always return complete files, never diffs/snippets
3. Include all imports, functions, boilerplate

---

### Memory Subsystem (`app/memory/`)

The memory subsystem implements project-based knowledge management with five core entities.

#### `models.py` — ORM Models

**Entity:** `Project`
- Top-level container for all knowledge
- Fields: `id`, `name` (unique), `description`, `created_at`, `updated_at`
- Relationships: `notes`, `tasks`, `files`, `messages` (cascade delete)

**Entity:** `Note`
- Distilled knowledge units
- Fields: `id`, `project_id`, `title`, `content`, `tags`, `source`, `created_at`, `updated_at`
- Searchable by title/content, filterable by tags

**Entity:** `Task`
- Actionable items with status tracking
- Fields: `id`, `project_id`, `title`, `description`, `status`, `priority`, `created_at`, `updated_at`
- Status values: `todo`, `in_progress`, `done`
- Priority values: `low`, `medium`, `high`

**Entity:** `File`
- Document metadata tracking
- Fields: `id`, `project_id`, `path`, `original_name`, `file_type`, `description`, `created_at`

**Entity:** `Message`
- Conversation logging
- Fields: `id`, `project_id`, `role`, `content`, `created_at`
- Role values: `user`, `assistant`, `system`

#### `schemas.py` — Pydantic Schemas

**Project:** `ProjectCreate`, `ProjectUpdate`, `ProjectOut`
**Note:** `NoteCreate`, `NoteCreateForProject`, `NoteUpdate`, `NoteOut`
**Task:** `TaskCreate`, `TaskCreateForProject`, `TaskUpdate`, `TaskOut`
**File:** `FileCreate`, `FileOut`
**Message:** `MessageCreate`, `MessageOut`

#### `service.py` — Business Logic

CRUD operations for all entities with filtering and search support.

#### `router.py` — API Routes

All routes prefixed with `/memory`.

---

## Desktop Client (`orb-desktop`)

**Location:** `D:/orb-desktop/`  
**Framework:** Electron  

### Key Files

- `main.js` — Electron main process
- `renderer.js` — UI logic, API calls
- `index.html` — UI structure with inline styles
- `styles.css` — Additional styling

### Features

- Provider selector dropdown (GPT/Claude/Gemini)
- Chat history display with provider labels
- Attachment UI (files UI-only for now)
- Reset session button (clears backend messages)

---

## Data Layer

### Database

**Engine:** SQLite  
**Location:** `D:/Orb/data/orb_memory.db`

### Tables

- `projects` — Project containers
- `notes` — Knowledge notes
- `tasks` — Actionable tasks
- `files` — File metadata
- `messages` — Conversation logs

### Cascade Behavior

All child entities cascade delete when parent project is deleted.

---

## Multi-LLM Orchestration

### Job-Type Based Routing (v0.5.0)

The `/chat` endpoint routes requests based on `job_type`:

```json
{
  "project_id": 1,
  "message": "User question",
  "job_type": "casual_chat",
  "force_provider": null
}
```

### Routing Rules

| Job Type Category | Routes To | Example Job Types |
|-------------------|-----------|-------------------|
| Low-stakes text | **GPT only** | `casual_chat`, `summary`, `copywriting` |
| Medium dev | **Claude** (configurable) | `simple_code_change`, `small_bugfix` |
| Heavy dev/architecture | **Claude** | `codegen_full_file`, `architecture_design` |
| High-stakes critical | **Claude → Gemini review** | `security_sensitive_change`, `high_stakes_infra` |
| Vision/analysis | **Gemini** | `image_analysis`, `screenshot_analysis` |
| Unknown | GPT (text) or Claude (code-like) | — |

### Two-Step Flow (High-Stakes)

For high-stakes jobs:
1. Claude generates primary response
2. Gemini reviews Claude's output as critic
3. Both responses returned in `LLMResult`

### Response Structure

```json
{
  "project_id": 1,
  "provider": "anthropic",
  "reply": "Claude's response...",
  "was_reviewed": true,
  "critic_review": "Gemini's critique..."
}
```

### Role Specialization

#### GPT (OpenAI) — Fast/Lightweight

**Use for:**
- Casual chat
- Summaries
- Copywriting
- Cleaning messy notes
- Shaping text into structured specs

**Model:** `gpt-4.1-mini`

#### Claude (Anthropic) — Flagship Engineer

**Use for:**
- Complex code generation
- Full-file generation
- Backend modules and architecture
- Systems planning and reasoning
- Major refactors

**Model:** `claude-sonnet-4-20250514`

**Rules:**
- Always returns complete files (never diffs)
- Asks for full file before modifying
- Includes all imports and boilerplate

#### Gemini (Google) — Critic

**Use for:**
- Secondary review for high-stakes work
- Cross-checking plans
- Pointing out risks, gaps, inconsistencies
- Validating architecture
- Vision/video analysis (future)

**Model:** `gemini-2.0-flash`

### Context Injection

Before each LLM call, the system builds a context block containing:

1. **Recent Notes** (last 10)
2. **Open Tasks** (todo + in_progress)

This context is injected into the system prompt.

---

## API Reference Summary

### Chat API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chat` | POST | Main chat endpoint with job-type routing |
| `/llm` | POST | Direct LLM call (no project, for testing) |
| `/providers` | GET | Check configured providers |
| `/job-types` | GET | List all job types by category |
| `/ping` | GET | Health check |

### Memory API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/memory/projects` | POST | Create project |
| `/memory/projects` | GET | List projects |
| `/memory/projects/{id}` | GET | Get project |
| `/memory/projects/{id}` | PATCH | Update project |
| `/memory/projects/{id}` | DELETE | Delete project |
| `/memory/notes` | POST | Create note (flat) |
| `/memory/notes` | GET | List/search notes |
| `/memory/projects/{id}/notes` | POST | Create note (nested) |
| `/memory/projects/{id}/notes` | GET | List notes for project |
| `/memory/tasks` | POST | Create task (flat) |
| `/memory/tasks` | GET | List/filter tasks |
| `/memory/projects/{id}/tasks` | POST | Create task (nested) |
| `/memory/projects/{id}/tasks` | GET | List tasks for project |
| `/memory/files` | POST | Create file metadata |
| `/memory/files` | GET | List files |
| `/memory/messages` | POST | Create message |
| `/memory/messages` | GET | List messages |
| `/memory/projects/{id}/messages` | DELETE | Clear chat history |

---

## Future Capabilities

### Planned Features

1. **File Ingestion Pipelines** — Document processing, automatic note extraction
2. **Mobile Screen Capture** — Screenshot/video analysis via Gemini
3. **Legacy Chat Log Import** — Bulk import from previous conversations
4. **Enhanced Memory** — Vector embeddings, semantic search
5. **Streaming Responses** — Real-time token streaming
6. **Function Calling** — Tool use integration

### Architecture Extensibility

- **LLM Module** — Add new providers by extending `clients.py` and `RoutingConfig`
- **Memory Subsystem** — Add new entity types as new models/routers
- **Job Types** — Add new job types to `JobType` enum and routing rules
- **Database** — SQLite schema supports migration to PostgreSQL

---

## Development Workflow

### Starting the System

**Backend:**
```powershell
cd D:\Orb
.\.venv\Scripts\activate
uvicorn main:app --reload
```

**Desktop Client:**
```powershell
cd D:\orb-desktop
npm start
```

### Testing LLM Routing

```powershell
# Test GPT routing
Invoke-RestMethod -Uri "http://localhost:8000/llm" `
  -Method POST -ContentType "application/json" `
  -Body '{"job_type":"casual_chat","message":"Hello"}'

# Test Claude routing
Invoke-RestMethod -Uri "http://localhost:8000/llm" `
  -Method POST -ContentType "application/json" `
  -Body '{"job_type":"codegen_full_file","message":"Write a hello world function"}'

# Test high-stakes (Claude + Gemini review)
Invoke-RestMethod -Uri "http://localhost:8000/llm" `
  -Method POST -ContentType "application/json" `
  -Body '{"job_type":"security_sensitive_change","message":"Review this auth flow"}'
```

### Environment Configuration

Create `.env` in `D:/Orb/`:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

---

## Glossary

- **Project** — Top-level container for all knowledge
- **Note** — Distilled knowledge unit
- **Task** — Actionable item with status tracking
- **File** — Document metadata
- **Message** — Conversation log entry
- **Provider** — LLM service (openai/anthropic/google)
- **Job Type** — Task classification that determines routing
- **Context Block** — Notes + tasks injected into system prompts
- **Critic Review** — Gemini's analysis of Claude's output (high-stakes only)
- **LLMTask** — Structured request to the routing layer
- **LLMResult** — Structured response from routing layer

---

## Changelog

### v0.5.0 (29 Nov 2025)
- Added `app/llm/` module for job-type-based routing
- Implemented `call_llm()` as single entry point for all LLM calls
- Added `JobType` enum with routing rules
- Added two-step flow for high-stakes jobs (Claude + Gemini review)
- Added `/llm` endpoint for direct testing
- Added `/job-types` endpoint to list routing categories
- Claude now enforces full-file returns (no diffs/snippets)

### v0.4.0
- Added multi-provider support (GPT/Claude/Gemini)
- Added provider selector in frontend
- Role-specific system prompts per provider

### v0.3.0
- Added context injection (notes + tasks into prompts)

### v0.2.0
- Added project-aware chat with SQLite message logging
- Replaced session-based chat with project-based memory

### v0.1.0
- Initial memory subsystem (Projects, Notes, Tasks, Files, Messages)
- Basic chat endpoint with GPT

---

## End of Architecture Map

Update this document when:
- New modules are added
- Database schema changes
- API endpoints are modified
- Routing rules are changed
- New job types are added
