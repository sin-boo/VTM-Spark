@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
title VTM Spark — Update

rem One block: cmd reads it whole before running it, so the update may replace
rem this file while it runs.
(
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0backend\packaging\update.ps1"
  rem 0 = updated or already current, 1 = failed, otherwise install.bat's code.
  call set "EC=%%ERRORLEVEL%%"
  pause
  setlocal EnableDelayedExpansion
  exit /b !EC!
)
