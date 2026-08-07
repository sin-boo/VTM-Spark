@echo off
setlocal
cd /d "%~dp0"

set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul

set "PY="
if exist "%~dp0.venv-build\Scripts\python.exe" (
  set "PY=%~dp0.venv-build\Scripts\python.exe"
  echo Using .venv-build.
) else if exist "%~dp0.venv\Scripts\python.exe" (
  set "PY=%~dp0.venv\Scripts\python.exe"
  echo Using .venv.
) else (
  echo No venv found. Run start.bat → [1] Smart Build first.
  goto :error
)

if not exist "%~dp0test-pose\input" (
  echo Missing test-pose\input — expected existing folder with input images.
  goto :error
)
if not exist "%~dp0test-pose\output" mkdir "%~dp0test-pose\output"

echo.
echo Pose vision TEST_MODE — schema-mapped face+iris+dwpose.
echo No DiT / VAE / crop / normalize / geometric repair.
echo Input : %~dp0test-pose\input
echo Output: %~dp0test-pose\output
echo.

"%PY%" test_pose.py
if errorlevel 1 goto :error
goto :end

:error
echo.
echo test_pose failed. Review the message above.
pause

:end
endlocal
