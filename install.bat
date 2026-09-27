@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
title VTM Studio — Install

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0backend\packaging\start-menu.ps1" -Action install
rem 0 = all OK, 1 = app build failed, 2 = built but some steps need attention.
set "EC=%ERRORLEVEL%"
endlocal & exit /b %EC%
