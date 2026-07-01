@echo off
setlocal
set "SCRIPT_DIR=%~dp0"

if not exist "%SCRIPT_DIR%SETUP.ps1" (
    echo [ERROR] SETUP.ps1 not found next to this file.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%SETUP.ps1"
set EXITCODE=%errorlevel%

exit /b %EXITCODE%
