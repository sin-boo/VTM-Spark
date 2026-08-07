@echo off
setlocal
cd /d "%~dp0"

set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul

set "PY="
if exist "%~dp0..\pipeline\i1\torch_train\.venv\Scripts\python.exe" (
  set "PY=%~dp0..\pipeline\i1\torch_train\.venv\Scripts\python.exe"
  echo Using torch_train venv.
) else if exist "%~dp0.venv\Scripts\python.exe" (
  set "PY=%~dp0.venv\Scripts\python.exe"
  echo Using vtm-noble\.venv.
) else if exist "%~dp0..\UI\.venv\Scripts\python.exe" (
  set "PY=%~dp0..\UI\.venv\Scripts\python.exe"
  echo Using UI\.venv.
) else (
  echo No venv found. Creating vtm-noble\.venv and installing requirements...
  python -m venv .venv
  if errorlevel 1 goto :error
  set "PY=%~dp0.venv\Scripts\python.exe"
  "%PY%" -m pip install --upgrade pip
  if errorlevel 1 goto :error
  "%PY%" -m pip install --disable-pip-version-check -r requirements.txt
  if errorlevel 1 goto :error
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
