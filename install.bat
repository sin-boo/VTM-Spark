@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
title VTM Noble — Install

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0backend\packaging\start-menu.ps1" -Action install
set "EC=%ERRORLEVEL%"
endlocal
exit %EC%
