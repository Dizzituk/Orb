# FILE: app/pipeline_v2/scaffolds/kotlin_templates_android.py
# Purpose: v2.0 rich Android-specific Kotlin scaffold templates (Application/Activity/Nav/Api/DI/Theme/...).
# Called-by: app.pipeline_v2.scaffolds.kotlin_scaffolds (shim dispatcher)
# Depends-on: app.pipeline_v2.scaffolds.kotlin_template_helpers, app.pipeline_v2.build_targets (type-only)
# Last-renovated: 2026-06-21
"""
v2.0 rich Android-specific Kotlin scaffold templates.

Split out of kotlin_scaffolds.py (BATCH 4) verbatim — framework-rich patterns.
Pure template data; dispatched by generate_kotlin_skeleton in the shim.
"""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile

from app.pipeline_v2.scaffolds.kotlin_template_helpers import (
    _req_block,
    _to_snake,
    _to_display_name,
)


def _application_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

import android.app.Application
import android.util.Log

/**
 * {name} — Application class.
 *
{_req_block(reqs)} */
class {name} : Application() {{

    override fun onCreate() {{
        super.onCreate()
        Log.d("{name}", "Application created")
        // TODO: Initialise DI, logging, crash reporting
    }}

    companion object {{
        private const val TAG = "{name}"
    }}
}}
"""


def _main_activity_skeleton(name: str, package: str, reqs: List[str], profile: Optional["BuildTargetProfile"] = None) -> str:
    return f"""package {package}

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import {package}.navigation.AppNavGraph
import {package}.ui.theme.AppTheme

class {name} : ComponentActivity() {{

    override fun onCreate(savedInstanceState: Bundle?) {{
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {{
            AppTheme {{
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) {{
                    AppNavGraph()
                }}
            }}
        }}
    }}
}}
"""


def _activity_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent

class {name} : ComponentActivity() {{

    override fun onCreate(savedInstanceState: Bundle?) {{
        super.onCreate(savedInstanceState)
        setContent {{
            // TODO: Set up Compose content
        }}
    }}
}}
"""


def _nav_graph_skeleton(name: str, package: str, reqs: List[str]) -> str:
    # Derive the base package (strip .navigation suffix)
    base_pkg = package.rsplit(".", 1)[0] if package.endswith(".navigation") else package
    return f"""package {package}

import androidx.compose.runtime.Composable
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import {base_pkg}.ui.home.HomeScreen
import {base_pkg}.ui.chat.ChatScreen

@Composable
fun {name}() {{
    val navController = rememberNavController()

    NavHost(
        navController = navController,
        startDestination = Routes.Home.route,
    ) {{
        composable(Routes.Home.route) {{
            HomeScreen(
                onNavigateToChat = {{ navController.navigate(Routes.Chat.route) }}
            )
        }}
        composable(Routes.Chat.route) {{
            ChatScreen(
                onNavigateBack = {{ navController.popBackStack() }}
            )
        }}
        // TODO: Add additional routes from spec
    }}
}}
"""


def _routes_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

sealed class {name}(val route: String) {{
    data object Home : {name}("home")
    data object Chat : {name}("chat")
    data object Settings : {name}("settings")
    // TODO: Add additional routes from spec
}}
"""


def _api_client_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

/**
 * {name} — Retrofit client for backend communication.
 *
{_req_block(reqs)} */
object {name} {{

    private var baseUrl: String = "http://100.64.0.1:8000"  // Default Tailscale address

    private val loggingInterceptor = HttpLoggingInterceptor().apply {{
        level = HttpLoggingInterceptor.Level.BODY
    }}

    private val okHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .addInterceptor(loggingInterceptor)
        .build()

    private var retrofit: Retrofit = buildRetrofit()

    fun updateBaseUrl(url: String) {{
        baseUrl = url
        retrofit = buildRetrofit()
    }}

    fun <T> createService(serviceClass: Class<T>): T {{
        return retrofit.create(serviceClass)
    }}

    private fun buildRetrofit(): Retrofit {{
        return Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
    }}
}}
"""


def _api_service_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

import {package}.models.ChatMessage
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Response
import retrofit2.http.*

/**
 * {name} — API interface for Astra backend.
 *
{_req_block(reqs)} */
interface {name} {{

    @POST("/stream/chat")
    suspend fun sendMessage(
        @Body request: ChatMessage.Request,
    ): Response<ChatMessage.Response>

    @GET("/memory/messages")
    suspend fun getChatHistory(
        @Query("project_id") projectId: Int,
        @Query("limit") limit: Int = 50,
    ): Response<List<ChatMessage.Response>>

    @Multipart
    @POST("/upload")
    suspend fun uploadFile(
        @Part file: MultipartBody.Part,
        @Part("description") description: RequestBody? = null,
    ): Response<Map<String, Any>>

    @GET("/ping")
    suspend fun healthCheck(): Response<Map<String, Any>>

    // TODO: Add additional endpoints from spec
}}
"""


def _manager_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * {name} — manages lifecycle and state.
 *
{_req_block(reqs)} */
class {name} {{

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val _state = MutableStateFlow<{name}State>({name}State.Idle)
    val state: StateFlow<{name}State> = _state.asStateFlow()

    fun start() {{
        scope.launch {{
            _state.value = {name}State.Active
            // TODO: Implement from spec
        }}
    }}

    fun stop() {{
        _state.value = {name}State.Idle
    }}

    sealed class {name}State {{
        data object Idle : {name}State()
        data object Active : {name}State()
        data class Error(val message: String) : {name}State()
    }}
}}
"""


def _detector_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * {name} — detection service.
 *
{_req_block(reqs)} */
class {name} {{

    private val _isActive = MutableStateFlow(false)
    val isActive: StateFlow<Boolean> = _isActive.asStateFlow()

    private val _detected = MutableStateFlow(false)
    val detected: StateFlow<Boolean> = _detected.asStateFlow()

    fun startListening() {{
        _isActive.value = true
        // TODO: Implement detection loop from spec
    }}

    fun stopListening() {{
        _isActive.value = false
        _detected.value = false
    }}

    fun onDetected() {{
        _detected.value = true
        // TODO: Trigger callback/event
    }}
}}
"""


def _uploader_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File

/**
 * {name} — handles file/media uploads to the backend.
 *
{_req_block(reqs)} */
class {name}(private val context: Context) {{

    suspend fun uploadUri(uri: Uri, mimeType: String = "application/octet-stream"): Result<String> =
        withContext(Dispatchers.IO) {{
            try {{
                val inputStream = context.contentResolver.openInputStream(uri)
                    ?: return@withContext Result.failure(Exception("Cannot open URI"))
                val tempFile = File.createTempFile("upload_", ".tmp", context.cacheDir)
                tempFile.outputStream().use {{ out -> inputStream.copyTo(out) }}

                val requestBody = tempFile.asRequestBody(mimeType.toMediaTypeOrNull())
                val part = MultipartBody.Part.createFormData(
                    "file", tempFile.name, requestBody
                )

                // TODO: Call API service upload endpoint
                // val response = apiService.uploadFile(part)

                tempFile.delete()
                Result.success("Upload complete")
            }} catch (e: Exception) {{
                Result.failure(e)
            }}
        }}
}}
"""


def _picker_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.*

/**
 * {name} — composable for picking media from gallery.
 *
{_req_block(reqs)} */
@Composable
fun remember{name}(
    onPicked: (Uri) -> Unit,
): () -> Unit {{
    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) {{ uri ->
        uri?.let {{ onPicked(it) }}
    }}

    return {{ launcher.launch("*/*") }}
}}
"""


def _capture_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.*
import androidx.core.content.FileProvider
import java.io.File

/**
 * {name} — composable for capturing photos/videos with camera.
 *
{_req_block(reqs)} */
@Composable
fun remember{name}(
    onCaptured: (Uri) -> Unit,
): () -> Unit {{
    var photoUri by remember {{ mutableStateOf<Uri?>(null) }}

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) {{ success ->
        if (success) {{
            photoUri?.let {{ onCaptured(it) }}
        }}
    }}

    return {{
        // TODO: Create temp file URI via FileProvider
        // photoUri = FileProvider.getUriForFile(...)
        // launcher.launch(photoUri!!)
    }}
}}
"""


def _di_module_skeleton(name: str, package: str, reqs: List[str]) -> str:
    base_pkg = package.rsplit(".", 1)[0] if package.endswith(".di") else package
    return f"""package {package}

import {base_pkg}.network.AstraApiClient
import {base_pkg}.network.AstraApiService
import {base_pkg}.voice.SpeechRecognitionService
import {base_pkg}.voice.TextToSpeechService
import {base_pkg}.voice.WakeWordDetector
import {base_pkg}.voice.VoiceSessionManager

/**
 * {name} — dependency injection / service locator.
 *
{_req_block(reqs)} */
object {name} {{

    val apiService: AstraApiService by lazy {{
        AstraApiClient.createService(AstraApiService::class.java)
    }}

    val speechRecognition: SpeechRecognitionService by lazy {{
        SpeechRecognitionService()
    }}

    val textToSpeech: TextToSpeechService by lazy {{
        TextToSpeechService()
    }}

    val wakeWordDetector: WakeWordDetector by lazy {{
        WakeWordDetector()
    }}

    val voiceSessionManager: VoiceSessionManager by lazy {{
        VoiceSessionManager()
    }}

    fun updateBackendUrl(url: String) {{
        AstraApiClient.updateBaseUrl(url)
    }}
}}
"""


def _theme_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable

private val DarkColorScheme = darkColorScheme(
    primary = AstraPrimary,
    secondary = AstraSecondary,
    background = AstraDarkBackground,
    surface = AstraDarkSurface,
    onPrimary = AstraOnPrimary,
    onBackground = AstraOnDarkBackground,
    onSurface = AstraOnDarkSurface,
)

private val LightColorScheme = lightColorScheme(
    primary = AstraPrimary,
    secondary = AstraSecondary,
    background = AstraLightBackground,
    surface = AstraLightSurface,
    onPrimary = AstraOnPrimary,
    onBackground = AstraOnLightBackground,
    onSurface = AstraOnLightSurface,
)

@Composable
fun AppTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {{
    val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography(),
        content = content,
    )
}}
"""


def _color_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

import androidx.compose.ui.graphics.Color

// Astra brand colours
val AstraPrimary = Color(0xFF6C63FF)
val AstraSecondary = Color(0xFF03DAC5)
val AstraAccent = Color(0xFFFF6584)

// Dark theme
val AstraDarkBackground = Color(0xFF121212)
val AstraDarkSurface = Color(0xFF1E1E2E)
val AstraOnDarkBackground = Color(0xFFE1E1E6)
val AstraOnDarkSurface = Color(0xFFE1E1E6)

// Light theme
val AstraLightBackground = Color(0xFFFAFAFA)
val AstraLightSurface = Color(0xFFFFFFFF)
val AstraOnLightBackground = Color(0xFF1C1B1F)
val AstraOnLightSurface = Color(0xFF1C1B1F)

// Shared
val AstraOnPrimary = Color(0xFFFFFFFF)
val AstraSuccess = Color(0xFF4CAF50)
val AstraError = Color(0xFFE53935)
val AstraWarning = Color(0xFFFFC107)
"""


def _data_model_skeleton(name: str, package: str, reqs: List[str]) -> str:
    return f"""package {package}

import com.google.gson.annotations.SerializedName

/**
 * {name} — data transfer object.
 *
{_req_block(reqs)} */
data class {name}(
    @SerializedName("id")
    val id: String = "",
    @SerializedName("content")
    val content: String = "",
    @SerializedName("role")
    val role: String = "user",
    @SerializedName("timestamp")
    val timestamp: Long = System.currentTimeMillis(),
    // TODO: Add fields from spec
) {{

    data class Request(
        @SerializedName("project_id")
        val projectId: Int = 1,
        @SerializedName("message")
        val message: String = "",
    )

    data class Response(
        @SerializedName("content")
        val content: String = "",
        @SerializedName("role")
        val role: String = "assistant",
        @SerializedName("provider")
        val provider: String? = null,
        @SerializedName("model")
        val model: String? = null,
    )
}}
"""


def _composable_component_skeleton(name: str, package: str, reqs: List[str]) -> str:
    display = _to_display_name(name)
    return f"""package {package}

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MicOff
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * {name} — reusable composable component.
 *
{_req_block(reqs)} */
@Composable
fun {name}(
    isActive: Boolean = false,
    onClick: () -> Unit = {{}},
    modifier: Modifier = Modifier,
) {{
    IconButton(
        onClick = onClick,
        modifier = modifier.size(64.dp),
    ) {{
        Icon(
            imageVector = if (isActive) Icons.Default.Mic else Icons.Default.MicOff,
            contentDescription = "{display}",
            tint = if (isActive) MaterialTheme.colorScheme.primary
                   else MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.size(32.dp),
        )
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
