@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
title VTM Spark — Uninstall

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0backend\packaging\uninstall.ps1"
rem 0 = all removed, 2 = some steps need attention.
set "EC=%ERRORLEVEL%"
pause
endlocal & exit /b %EC%
