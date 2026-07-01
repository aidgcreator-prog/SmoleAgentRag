@echo off
setlocal
set "SCRIPT_DIR=%~dp0"

if not exist "%SCRIPT_DIR%RUN.ps1" (
    echo [ERROR] RUN.ps1 not found next to this file.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%RUN.ps1"
set EXITCODE=%errorlevel%

if %EXITCODE% neq 0 (
    echo.
    echo Script exited with code %EXITCODE%. See messages above.
    pause
)

exit /b %EXITCODE%
