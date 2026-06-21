# Purpose: simple create utils 15 — discovery + per-language pattern extraction (split from _simple_create_utils_15.py).
# Called-by: app.pot_spec.grounded._simple_create_utils_15
# Depends-on: app.pot_spec.grounded._sbx_fs, app.pot_spec.grounded._simple_create_utils_13, app.pot_spec.grounded._simple_create_utils_14
# Last-renovated: 2026-06-21
from __future__ import annotations
import logging
import os
import re
from typing import Any, Dict, List
from app.pot_spec.grounded._simple_create_utils_13 import ARCHITECTURAL_FILE_PATTERNS, _score_integration_point
from app.pot_spec.grounded._simple_create_utils_14 import CONCEPT_DIRECTORY_PATTERNS
from app.pot_spec.grounded._sbx_fs import _sbx_read, _sbx_ls
logger = logging.getLogger(__name__)


def _find_integration_points(
    project_path: str,
    concepts: List[str],
    sandbox_client: Any = None,
) -> List[IntegrationPoint]:
    """
    v2.0: Find architecturally relevant integration points.
    
    Uses SPECIFIC file patterns (regex on full filename) instead of
    substring matching. Also searches for existing directories that
    match the task's concepts.
    """
    from ._simple_create_utils_16 import IntegrationPoint
    points = []
    
    skip_dirs = {
        'node_modules', '.git', '__pycache__', '.venv', 'venv',
        'dist', 'build', '.next', 'coverage', '.architecture',
        '_backup_before_audit', '_patches',
    }
    # v2.2: Extended to cover Android (.kt, .kts), Rust (.rs), Go (.go), config files (.toml, .xml, .gradle).
    # Without this, _scan_dir filters out Kotlin/Gradle/Android files BEFORE the
    # ARCHITECTURAL_FILE_PATTERNS loop ever runs — the v3.7 Android patterns sat dead.
    source_exts = ('.tsx', '.jsx', '.ts', '.js', '.py', '.css', '.kt', '.kts', '.rs', '.go', '.java', '.swift', '.toml', '.xml', '.gradle')

    def _scan_dir(dir_path: str, rel_root: str) -> None:
        try:
            entries = _sbx_ls(dir_path)
        except Exception as e:
            logger.warning("[simple_create] Error scanning %s: %s", dir_path, e)
            return
        for name in entries:
            if name in skip_dirs:
                continue
            full_path = os.path.join(dir_path, name)
            ext = os.path.splitext(name)[1].lower()
            if ext and ext in source_exts:
                # It's a source file
                for pattern, relevance, action in ARCHITECTURAL_FILE_PATTERNS:
                    if re.match(pattern, name, re.IGNORECASE):
                        points.append(IntegrationPoint(
                            file_path=full_path,
                            file_name=name,
                            relevance=relevance,
                            action=action,
                        ))
                        break
                for concept in concepts:
                    dir_patterns = CONCEPT_DIRECTORY_PATTERNS.get(concept, [])
                    for dir_pat in dir_patterns:
                        if dir_pat in rel_root.lower():
                            if not any(p.file_path == full_path for p in points):
                                points.append(IntegrationPoint(
                                    file_path=full_path,
                                    file_name=name,
                                    relevance=f"In '{dir_pat}/' directory (relevant to {concept})",
                                    action="reference",
                                ))
                            break
            elif not ext:
                # Likely a directory — recurse
                child_rel = os.path.join(rel_root, name) if rel_root != '.' else name
                _scan_dir(full_path, child_rel)

    try:
        _scan_dir(project_path, '.')
    except Exception as e:
        logger.warning("[simple_create] Error scanning project: %s", e)
    
    # Dedupe and prioritize
    seen = set()
    unique = []
    for p in points:
        if p.file_path not in seen:
            seen.add(p.file_path)
            unique.append(p)
    
    # v3.7: Score each integration point using content signals + path heuristics.
    # Drop negative-scored points (false positives like static/main.py).
    # Sort remainder: modify actions first, then highest score, then filename.
    scored = []
    dropped = []
    for p in unique:
        s = _score_integration_point(p.file_path, project_path)
        if s < 0:
            dropped.append((p.file_name, s))
        else:
            scored.append((p, s))
    
    if dropped:
        print(f"[simple_create] v3.7 DROPPED {len(dropped)} negative-scored integration point(s): "
              f"{[(name, sc) for name, sc in dropped]}")
        logger.info("[simple_create] v3.7 Dropped %d negative-scored points: %s", len(dropped), dropped)
    
    scored.sort(key=lambda x: (0 if x[0].action == "modify" else 1, -x[1], x[0].file_name))
    result = [p for p, _ in scored]
    
    return result[:15]  # Limit to top 15

def _extract_patterns(
    integration_points: List[IntegrationPoint],
    tech_stack: TechStack,
) -> Dict[str, str]:
    """Extract coding patterns from existing files.

    v2.10: Multi-language pattern extraction. Detects idiomatic usage of the
    libraries the project actually has, so the LLM can match existing style
    when proposing new code. Domain-agnostic — looks at file extension and
    pulls language-appropriate patterns.

    Pattern keys returned (when found, format: "category:filename"):
        persistence:    SharedPreferences / DataStore / JSON / Room / SQLAlchemy / pickle
        http_client:    OkHttp / Retrofit / Ktor / requests / httpx / aiohttp / fetch
        state_mgmt:     StateFlow / LiveData / RxJava / useState / Redux
        di:             Hilt / Koin / FastAPI Depends / manual
        component:      React fn / Compose @Composable / FastAPI router
        import_block:   first run of imports (style anchor)
    """
    from ._simple_create_utils_16 import IntegrationPoint
    from ._simple_create_utils_17 import TechStack
    patterns: Dict[str, str] = {}

    for point in integration_points:
        if point.action != "modify":
            continue

        content = _sbx_read(point.file_path)
        if not content:
            continue

        fname = point.file_name
        head = content[:4000]

        # ------------------------------------------------------------------
        # KOTLIN / ANDROID
        # ------------------------------------------------------------------
        if fname.endswith(".kt") or fname.endswith(".kts"):
            # Persistence — SharedPreferences / DataStore / Room
            if re.search(r"\bSharedPreferences\b|getSharedPreferences\(", content):
                m = re.search(r"[^\n]*SharedPreferences[^\n]{0,120}", content)
                if m:
                    patterns[f"persistence:{fname}"] = m.group(0).strip()[:200]
            elif re.search(r"\bDataStore\b|preferencesDataStore\(", content):
                patterns[f"persistence:{fname}"] = "DataStore (preferences)"
            elif re.search(r"@Entity\b|@Dao\b|RoomDatabase\b", content):
                patterns[f"persistence:{fname}"] = "Room (androidx.room) — entity/dao detected"

            # HTTP client — OkHttp / Retrofit / Ktor
            if re.search(r"\bOkHttpClient\b|okhttp3\.", content):
                m = re.search(r"OkHttpClient[^\n]{0,150}", content)
                patterns[f"http_client:{fname}"] = (m.group(0).strip() if m else "OkHttp")[:200]
            if re.search(r"\bRetrofit\b|retrofit2\.|@GET\(|@POST\(", content):
                m = re.search(r"Retrofit\.Builder[^\n]{0,150}|@(?:GET|POST|PUT|DELETE)\([^\n]{0,80}", content)
                patterns[f"http_client_retrofit:{fname}"] = (m.group(0).strip() if m else "Retrofit")[:200]
            if re.search(r"\bHttpClient\b\s*\(|io\.ktor\.client", content):
                patterns[f"http_client_ktor:{fname}"] = "Ktor HttpClient"

            # State management — StateFlow / LiveData
            if re.search(r"\bStateFlow\b|MutableStateFlow\(", content):
                m = re.search(r"(?:Mutable)?StateFlow[^\n]{0,120}", content)
                patterns[f"state_mgmt:{fname}"] = (m.group(0).strip() if m else "StateFlow")[:200]
            elif re.search(r"\bLiveData\b|MutableLiveData\(", content):
                patterns[f"state_mgmt:{fname}"] = "LiveData (legacy)"

            # DI — Hilt / Koin
            if re.search(r"@HiltAndroidApp|@AndroidEntryPoint|@Inject\b", content):
                patterns[f"di:{fname}"] = "Hilt (@HiltAndroidApp / @AndroidEntryPoint / @Inject)"
            elif re.search(r"\bKoin\b|startKoin\(", content):
                patterns[f"di:{fname}"] = "Koin"

            # Compose @Composable functions — anchor for new UI
            comp_match = re.search(r"@Composable[\s\n]+(?:private\s+|public\s+|internal\s+)?fun\s+\w+[^\n]{0,150}", content)
            if comp_match:
                patterns[f"component:{fname}"] = comp_match.group(0).strip()[:250]

            # Import block — style anchor
            imp_match = re.search(r"((?:^import\s+\S+\s*\n){2,})", content, re.MULTILINE)
            if imp_match:
                patterns[f"import_block:{fname}"] = imp_match.group(1)[:400]

        # ------------------------------------------------------------------
        # PYTHON / BACKEND
        # ------------------------------------------------------------------
        elif fname.endswith(".py"):
            # HTTP client — requests / httpx / aiohttp
            if re.search(r"\bimport\s+httpx\b|httpx\.(?:AsyncClient|Client|get|post)\(", content):
                patterns[f"http_client:{fname}"] = "httpx"
            elif re.search(r"\bimport\s+requests\b|requests\.(?:get|post|put|delete)\(", content):
                patterns[f"http_client:{fname}"] = "requests"
            elif re.search(r"\bimport\s+aiohttp\b|aiohttp\.ClientSession\(", content):
                patterns[f"http_client:{fname}"] = "aiohttp"

            # Web framework — FastAPI / Flask / Django routers
            if re.search(r"\bAPIRouter\(|@\w+\.(?:get|post|put|delete|patch)\(", content):
                m = re.search(r"@\w+\.(?:get|post|put|delete|patch)\([^\n]{0,150}", content)
                patterns[f"router:{fname}"] = (m.group(0).strip() if m else "FastAPI APIRouter")[:200]
            elif re.search(r"\bfrom\s+flask\s+import|@app\.route\(", content):
                patterns[f"router:{fname}"] = "Flask @app.route"

            # Persistence — SQLAlchemy / sqlite3 / JSON / pickle
            if re.search(r"\bfrom\s+sqlalchemy\b|Session\(|sessionmaker\(", content):
                patterns[f"persistence:{fname}"] = "SQLAlchemy"
            elif re.search(r"\bimport\s+sqlite3\b|sqlite3\.connect\(", content):
                patterns[f"persistence:{fname}"] = "sqlite3 (stdlib)"
            elif re.search(r"\bjson\.dump\(|\bjson\.load\(", content):
                patterns[f"persistence:{fname}"] = "JSON file (json module)"

            # Async style — asyncio / sync
            if re.search(r"^async\s+def\s+", content, re.MULTILINE):
                patterns[f"async_style:{fname}"] = "asyncio (async def)"

            # Pydantic models / response shapes
            if re.search(r"\bfrom\s+pydantic\b|\bclass\s+\w+\(BaseModel\)", content):
                m = re.search(r"class\s+\w+\(BaseModel\)[^\n]{0,80}", content)
                patterns[f"models:{fname}"] = (m.group(0).strip() if m else "pydantic BaseModel")[:200]

        # ------------------------------------------------------------------
        # REACT / TYPESCRIPT (existing behaviour preserved)
        # ------------------------------------------------------------------
        elif fname.endswith((".tsx", ".jsx")):
            comp_match = re.search(
                r'((?:export\s+)?(?:const|function)\s+\w+\s*[=:]\s*(?:\([^)]*\)|[^=])*\s*(?:=>|{)[^}]*(?:return\s*\()?[^)]*<)',
                head
            )
            if comp_match:
                patterns[f"component:{fname}"] = comp_match.group(0)[:500]
            import_match = re.search(r"^(import\s+.+\n)+", content, re.MULTILINE)
            if import_match:
                patterns[f"import_block:{fname}"] = import_match.group(0)[:300]

        # API call patterns (any language) — kept from prior version
        if "api" in fname.lower():
            fetch_match = re.search(
                r'((?:export\s+)?(?:async\s+)?(?:function|const)\s+\w+\s*[=:]?\s*(?:async\s*)?\([^)]*\)[^{]*{[^}]*fetch[^}]*})',
                content,
                re.DOTALL
            )
            if fetch_match:
                patterns["api_call_pattern"] = fetch_match.group(0)[:600]

    return patterns
