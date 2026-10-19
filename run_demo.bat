@echo off
setlocal enabledelayedexpansion
title Face Image Quality Assessment System - Demo Launcher

echo =========================================================================
echo       Face Image Quality Assessment (FIQA) & Surveillance System
echo =========================================================================
echo.

:: 1. Verify Python Virtual Environment
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\
    echo Please create it first:
    echo    python -m venv .venv
    echo    .\.venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: 2. Ensure Frontend is Built
if not exist "frontend\dist\index.html" (
    echo [INFO] Compiling production frontend bundle...
    pushd frontend
    call npm install
    call npm run build
    popd
    echo [INFO] Frontend compiled successfully.
)

:: 3. Detect Local IP Address for Team LAN Access
set LOCAL_IP=127.0.0.1
for /f "tokens=4" %%a in ('route print ^| findstr 0.0.0.0.*0.0.0.0') do (
    if not "%%a"=="" set LOCAL_IP=%%a
)

echo.
echo -------------------------------------------------------------------------
echo   Application is Ready! Showcase URLs:
echo -------------------------------------------------------------------------
echo   - Local Browser:         http://localhost:8000
echo   - Team LAN (Office WiFi): http://%LOCAL_IP%:8000
echo   - Interactive API Docs:  http://localhost:8000/docs
echo   - Health Status:         http://localhost:8000/api/health
echo -------------------------------------------------------------------------
echo.
echo Press Ctrl+C in this window anytime to stop the server.
echo.

:: 4. Start Unified Production Server (Serves React UI + FastAPI Backend)
.\.venv\Scripts\uvicorn.exe api.main:app --host 0.0.0.0 --port 8000

pause
