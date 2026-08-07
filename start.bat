@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
title VTM Noble

REM Same-window menu: Smart Build + Start (kill orphans is automatic).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\start-menu.ps1"
set "EC=%ERRORLEVEL%"
endlocal & exit /b %EC%
