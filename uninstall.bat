@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
title VTM Spark — Uninstall
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0backend\packaging\uninstall.ps1"
rem 0 = all removed, 2 = some steps need attention, 3 = delete this folder too.
set "EC=%ERRORLEVEL%"
pause
if not "%EC%"=="3" endlocal & exit /b %EC%
rem Last line: the folder (this file included) cannot go while cmd runs from it.
rem (goto) ends this batch file first; the rest of the line still runs.
cd /d "%TEMP%" & (goto) 2>nul & rd /s /q "%ROOT%" & if exist "%ROOT%" (echo Some files could not be deleted - close anything using them and delete "%ROOT%". & pause)