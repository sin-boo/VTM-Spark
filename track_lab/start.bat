@echo off
cd /d "%~dp0"
title Track Lab
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 (
  echo.
  echo Track Lab did not start. The error is above.
  pause
)
