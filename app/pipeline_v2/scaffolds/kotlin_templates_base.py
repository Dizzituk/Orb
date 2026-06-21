# FILE: app/pipeline_v2/scaffolds/kotlin_templates_base.py
# Purpose: v1.0 base Kotlin/MVVM scaffold templates (Entity/Dao/ViewModel/Screen/...).
# Called-by: app.pipeline_v2.scaffolds.kotlin_scaffolds (shim dispatcher)
# Depends-on: app.pipeline_v2.scaffolds.kotlin_template_helpers
# Last-renovated: 2026-06-21
"""
v1.0 base Kotlin scaffold templates.

Split out of kotlin_scaffolds.py (BATCH 4) verbatim — the foundational MVVM
skeletons. Pure template data; dispatched by generate_kotlin_skeleton in the shim.
"""
from __future__ import annotations

from typing import List

from app.pipeline_v2.scaffolds.kotlin_template_helpers import (
    _req_block,
    _to_snake,
    _to_display_name,
)


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
