@echo off
title ASTRA Unified Boot
color 0A

echo.
echo  =======================================
echo       ASTRA - Unified Boot Sequence
echo  =======================================
echo.

:: ── 1. Backend API (FastAPI on port 8000) ────────────────────────────
echo [1/3] Starting Backend API...
cd /d D:\Orb
start "ASTRA Backend" cmd /k "cd /d D:\Orb && .venv\Scripts\activate && python -m uvicorn main:app --reload --port 8000"
timeout /t 2 /nobreak >nul

:: ── 2. TTS Microservice (FastAPI on port 8001) ──────────────────────
echo [2/3] Starting TTS Service...
start "ASTRA TTS" cmd /k "cd /d D:\Orb && .venv\Scripts\activate && python -m uvicorn app.voice.tts_server:app --port 8001"
timeout /t 2 /nobreak >nul

:: ── 3. Electron Frontend ─────────────────────────────────────────────
echo [3/3] Starting Electron Desktop...
start "ASTRA Desktop" cmd /k "cd /d D:\orb-desktop && npm run electron:dev"

echo.
echo  =======================================
echo       All services launched!
echo  =======================================
echo.
echo   Backend API:  http://localhost:8000
echo   TTS Service:  http://localhost:8001
echo   Desktop:      Electron (dev mode)
echo.
echo  Close this window at any time - services
echo  run independently in their own windows.
echo.
timeout /t 10
