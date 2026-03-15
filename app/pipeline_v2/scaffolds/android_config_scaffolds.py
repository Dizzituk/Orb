# FILE: app/pipeline_v2/scaffolds/android_config_scaffolds.py
"""
Android Configuration File Scaffolds.

Generates deterministic, near-complete config files for Android projects:
  build.gradle.kts (root and app module)
  settings.gradle.kts
  gradle.properties
  AndroidManifest.xml
  gradle-wrapper.properties

These files are NOT stubs — they're 90%+ complete and should compile
as-is with minimal modification by the agentic builder.

v1.0 (2026-03-12): Initial implementation.
"""
from __future__ import annotations

import os
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile


def generate_android_config(
    file_path: str,
    requirements: List[str],
    profile: Optional["BuildTargetProfile"] = None,
) -> Optional[str]:
    """Generate config file content based on filename. Returns None if not a config file."""
    norm = file_path.replace("\\", "/")
    basename = os.path.basename(norm)
    
    pkg = profile.package_name if profile else "com.astra.app"
    proj_name = profile.project_name if profile else "AstraApp"

    # App module build.gradle.kts (check FIRST — more specific)
    if basename == "build.gradle.kts" and ("/app/" in norm or norm.startswith("app/")):
        return _app_build_gradle(pkg)
    
    # Root build.gradle.kts
    if basename == "build.gradle.kts":
        return _root_build_gradle(proj_name)
    
    # settings.gradle.kts
    if basename == "settings.gradle.kts":
        return _settings_gradle(proj_name)
    
    # gradle.properties
    if basename == "gradle.properties":
        return _gradle_properties()
    
    # AndroidManifest.xml
    if basename == "AndroidManifest.xml":
        return _android_manifest(pkg, proj_name)
    
    # gradle-wrapper.properties
    if basename == "gradle-wrapper.properties":
        return _gradle_wrapper_properties()
    
    return None


def _root_build_gradle(project_name: str) -> str:
    return f'''// Top-level build file for {project_name}
plugins {{
    id("com.android.application") version "8.2.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.22" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.0" apply false
}}
'''


def _app_build_gradle(package_name: str) -> str:
    return f'''plugins {{
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}}

android {{
    namespace = "{package_name}"
    compileSdk = 34

    defaultConfig {{
        applicationId = "{package_name}"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }}

    buildTypes {{
        release {{
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }}
    }}

    compileOptions {{
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }}

    kotlinOptions {{
        jvmTarget = "17"
    }}

    buildFeatures {{
        compose = true
    }}
}}

dependencies {{
    // Compose BOM
    val composeBom = platform("androidx.compose:compose-bom:2024.02.00")
    implementation(composeBom)
    androidTestImplementation(composeBom)

    // Compose
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")

    // Core
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.7.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
    implementation("androidx.activity:activity-compose:1.8.2")
    implementation("androidx.navigation:navigation-compose:2.7.7")

    // Networking
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")

    // CameraX (for media capture)
    implementation("androidx.camera:camera-core:1.3.1")
    implementation("androidx.camera:camera-camera2:1.3.1")
    implementation("androidx.camera:camera-lifecycle:1.3.1")
    implementation("androidx.camera:camera-view:1.3.1")

    // DataStore (for settings/preferences)
    implementation("androidx.datastore:datastore-preferences:1.0.0")

    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
}}
'''


def _settings_gradle(project_name: str) -> str:
    slug = project_name.replace(" ", "").replace("-", "")
    return f'''pluginManagement {{
    repositories {{
        google()
        mavenCentral()
        gradlePluginPortal()
    }}
}}

dependencyResolution {{
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {{
        google()
        mavenCentral()
    }}
}}

rootProject.name = "{project_name}"
include(":app")
'''


def _gradle_properties() -> str:
    return '''# Project-wide Gradle settings.
org.gradle.jvmargs=-Xmx2048m -Dfile.encoding=UTF-8
org.gradle.parallel=true
org.gradle.caching=true

# AndroidX
android.useAndroidX=true

# Kotlin
kotlin.code.style=official

# Compose
android.enableJetifier=false
'''


def _android_manifest(package_name: str, project_name: str) -> str:
    # Derive the Application class name from package
    app_class = package_name.rsplit(".", 1)[-1].capitalize() if "." in package_name else "App"
    # Use AstraApp as the convention
    app_class_fqn = f".AstraApp"
    
    return f'''<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <!-- Network access for Tailscale/backend connection -->
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />

    <!-- Voice/audio -->
    <uses-permission android:name="android.permission.RECORD_AUDIO" />

    <!-- Camera for photo/video capture -->
    <uses-permission android:name="android.permission.CAMERA" />

    <!-- File/media access -->
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"
        android:maxSdkVersion="32" />
    <uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
    <uses-permission android:name="android.permission.READ_MEDIA_VIDEO" />

    <!-- Foreground service for voice session -->
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE_MICROPHONE" />

    <application
        android:name="{app_class_fqn}"
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="{project_name}"
        android:roundIcon="@mipmap/ic_launcher_round"
        android:supportsRtl="true"
        android:theme="@style/Theme.Material3.DynamicColors.DayNight"
        android:usesCleartextTraffic="true">

        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:theme="@style/Theme.Material3.DynamicColors.DayNight">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>

</manifest>
'''


def _gradle_wrapper_properties() -> str:
    return '''distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
distributionUrl=https\\://services.gradle.org/distributions/gradle-8.5-bin.zip
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists
'''


def copy_gradle_wrapper(project_root: str) -> bool:
    """Copy Gradle wrapper files from a known Android project.

    The wrapper JAR, gradlew, and gradlew.bat are binary/large files
    that can't be generated from templates. We copy them from the
    first available Android project that already has them.
    """
    import os
    import shutil

    target_wrapper_dir = os.path.join(project_root, "gradle", "wrapper")
    os.makedirs(target_wrapper_dir, exist_ok=True)

    # Find a source project with a working Gradle wrapper
    source_dirs = [
        r"D:\Astra Android Folder\AndroidDriverCopilot",
        r"D:\AndroidDriverCopilot",
    ]
    source = None
    for d in source_dirs:
        jar = os.path.join(d, "gradle", "wrapper", "gradle-wrapper.jar")
        if os.path.isfile(jar):
            source = d
            break

    if not source:
        return False

    files_to_copy = [
        ("gradle/wrapper/gradle-wrapper.jar", "gradle/wrapper/gradle-wrapper.jar"),
        ("gradle/wrapper/gradle-wrapper.properties", "gradle/wrapper/gradle-wrapper.properties"),
        ("gradlew", "gradlew"),
        ("gradlew.bat", "gradlew.bat"),
    ]
    copied = 0
    for src_rel, dst_rel in files_to_copy:
        src = os.path.join(source, src_rel)
        dst = os.path.join(project_root, dst_rel)
        if os.path.isfile(src) and not os.path.isfile(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    return copied > 0
