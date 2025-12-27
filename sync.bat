@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync_to_github.ps1"
if %errorlevel% neq 0 (
    echo Sync failed with error code %errorlevel%
    pause
) else (
    echo Sync completed successfully
    timeout /t 3 >nul
)
