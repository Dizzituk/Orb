# FILE: app/pipeline_v2/scaffolds/kotlin_scaffolds.py
"""
Kotlin Scaffold Templates for Android/Jetpack Compose projects.

Generates deterministic skeleton files based on file naming conventions:
  *Entity.kt   → Room @Entity data class
  *Dao.kt      → Room @Dao interface
  *ViewModel.kt → ViewModel with StateFlow
  *UiState.kt  → data class for UI state
  *Screen.kt   → @Composable function with Scaffold
  *Service.kt  → Service class with suspend functions
  *Repository.kt → Repository with DAO injection
  *Adapter.kt  → Interface definition
  *Bridge.kt   → Singleton object with StateFlow

Package declarations are derived from the file path relative to source root.

v1.0 (2026-03-10): Initial implementation for Driver CoPilot.
"""
from __future__ import annotations

import os
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile


def generate_kotlin_skeleton(
    file_path: str,
    requirements: List[str],
    profile: "BuildTargetProfile",
) -> str:
    """Generate a Kotlin skeleton based on file name pattern."""
    basename = os.path.basename(file_path).replace(".kt", "")
    package = _derive_package(file_path, profile)

    if basename.endswith("Entity"):
        return _entity_skeleton(basename, package, requirements)
    elif basename.endswith("Dao"):
        return _dao_skeleton(basename, package, requirements)
    elif basename.endswith("ViewModel"):
        return _viewmodel_skeleton(basename, package, requirements)
    elif basename.endswith("UiState"):
        return _uistate_skeleton(basename, package, requirements)
    elif basename.endswith("Screen"):
        return _screen_skeleton(basename, package, requirements)
    elif basename.endswith("Service"):
        return _service_skeleton(basename, package, requirements)
    elif basename.endswith("Repository"):
        return _repository_skeleton(basename, package, requirements)
    elif basename.endswith("Adapter"):
        return _adapter_skeleton(basename, package, requirements)
    elif basename.endswith("Bridge"):
        return _bridge_skeleton(basename, package, requirements)
    elif basename.endswith("Result"):
        return _result_skeleton(basename, package, requirements)
    elif basename.endswith("Parser"):
        return _parser_skeleton(basename, package, requirements)
    else:
        return _generic_kotlin_skeleton(basename, package, requirements)


def _derive_package(file_path: str, profile: "BuildTargetProfile") -> str:
    """Derive the Kotlin package name from file path and profile."""
    norm = file_path.replace("\\", "/")

    # Strip absolute prefix if present
    proot = profile.project_root.replace("\\", "/").rstrip("/") + "/"
    if norm.lower().startswith(proot.lower()):
        norm = norm[len(proot):]

    # Find the java/ or kotlin/ source root marker
    for marker in ("java/", "kotlin/"):
        idx = norm.find(marker)
        if idx >= 0:
            pkg_path = norm[idx + len(marker):]
            # Remove filename
            pkg_path = pkg_path.rsplit("/", 1)[0] if "/" in pkg_path else ""
            return pkg_path.replace("/", ".")

    # Fallback: use profile package + relative subdirectory
    source_root = profile.source_root.replace("\\", "/").rstrip("/") + "/"
    if norm.lower().startswith(source_root.lower()):
        rel = norm[len(source_root):]
    else:
        rel = norm

    # Remove filename, convert to package
    if "/" in rel:
        subpkg = rel.rsplit("/", 1)[0].replace("/", ".")
        return f"{profile.package_name}.{subpkg}"
    return profile.package_name


def _req_block(requirements: List[str]) -> str:
    """Format requirements as a Kotlin doc comment block."""
    if not requirements:
        return ""
    lines = [" * Requirements:"]
    for r in requirements:
        lines.append(f" *   - {r}")
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════
# Template generators
# ═══════════════════════════════════════════════════════════════════

def _entity_skeleton(name: str, package: str, reqs: List[str]) -> str:
    table_name = _to_snake(name.replace("Entity", "")) + "s"
    return f"""package {package}

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * {name} — auto-generated scaffold.
 *
{_req_block(reqs)} */
@Entity(tableName = "{table_name}")
data class {name}(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    // TODO: Add entity fields from spec
)
"""


def _dao_skeleton(name: str, package: str, reqs: List[str]) -> str:
    entity_name = name.replace("Dao", "Entity")
    table_name = _to_snake(name.replace("Dao", "")) + "s"
    return f"""package {package}

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

/**
 * {name} — auto-generated scaffold.
 *
{_req_block(reqs)} */
@Dao
interface {name} {{
    @Query("SELECT * FROM {table_name} ORDER BY id DESC")
    fun getAll(): Flow<List<{entity_name}>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(entity: {entity_name}): Long

    @Update
    suspend fun update(entity: {entity_name})

    @Delete
    suspend fun delete(entity: {entity_name})

    // TODO: Add spec-specific queries
}}
"""


def _viewmodel_skeleton(name: str, package: str, reqs: List[str]) -> str:
    state_name = name.replace("ViewModel", "UiState")
    return f"""package {package}

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * {name} — auto-generated scaffold.
 *
{_req_block(reqs)} */
class {name} : ViewModel() {{

    private val _uiState = MutableStateFlow({state_name}())
    val uiState: StateFlow<{state_name}> = _uiState.asStateFlow()

    // TODO: Implement ViewModel logic from spec

    init {{
        // TODO: Initial data load
    }}
}}
"""


def _uistate_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

/**
 * {name} — auto-generated scaffold.
 *
{_req_block(reqs)} */
data class {name}(
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    // TODO: Add UI state fields from spec
)
"""


def _screen_skeleton(name: str, package: str, reqs: List[str]) -> str:
    vm_name = name.replace("Screen", "ViewModel")
    title = _to_display_name(name.replace("Screen", ""))
    return f"""package {package}

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel

/**
 * {name} — auto-generated scaffold.
 *
{_req_block(reqs)} */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun {name}(
    // viewModel: {vm_name} = viewModel(),
) {{
    // val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    Scaffold(
        topBar = {{
            TopAppBar(title = {{ Text("{title}") }})
        }}
    ) {{ paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {{
            Text("{title} — scaffold placeholder")
            // TODO: Implement screen UI from spec
        }}
    }}
}}
"""


def _service_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * {name} — auto-generated scaffold.
 *
{_req_block(reqs)} */
class {name} {{

    // TODO: Implement service methods from spec

    suspend fun performAction() = withContext(Dispatchers.IO) {{
        // TODO: Implement
    }}
}}
"""


def _repository_skeleton(name: str, package: str, reqs: List[str]) -> str:
    dao_name = name.replace("Repository", "Dao")
    return f"""package {package}

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * {name} — auto-generated scaffold.
 *
{_req_block(reqs)} */
class {name}(
    // private val dao: {dao_name},
) {{

    // TODO: Implement repository methods from spec

}}
"""


def _adapter_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

/**
 * {name} — auto-generated scaffold.
 *
{_req_block(reqs)} */
interface {name} {{

    // TODO: Define adapter interface from spec

}}
"""


def _bridge_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * {name} — auto-generated singleton bridge.
 *
{_req_block(reqs)} */
object {name} {{

    private val _state = MutableStateFlow<String?>(null)
    val state: StateFlow<String?> = _state.asStateFlow()

    fun update(value: String?) {{
        _state.value = value
    }}

    // TODO: Implement bridge methods from spec
}}
"""


def _result_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

/**
 * {name} — auto-generated scaffold.
 *
{_req_block(reqs)} */
data class {name}(
    val success: Boolean = false,
    val errorMessage: String? = null,
    // TODO: Add result fields from spec
)
"""


def _parser_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

/**
 * {name} — auto-generated scaffold.
 *
{_req_block(reqs)} */
class {name} {{

    // TODO: Implement parsing logic from spec

    fun parse(input: String): Any {{
        throw NotImplementedError("{name}.parse()")
    }}
}}
"""


def _generic_kotlin_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

/**
 * {name} — auto-generated scaffold.
 *
{_req_block(reqs)} */
// TODO: Implement {name} from spec
"""


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _to_snake(name: str) -> str:
    """PascalCase → snake_case."""
    result = []
    for i, c in enumerate(name):
        if c.isupper() and i > 0:
            result.append("_")
        result.append(c.lower())
    return "".join(result)


def _to_display_name(name: str) -> str:
    """PascalCase → 'Display Name'."""
    result = []
    for i, c in enumerate(name):
        if c.isupper() and i > 0:
            result.append(" ")
        result.append(c)
    return "".join(result)
