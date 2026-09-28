@echo off
:: Register DirectShow filters as "VTM Spark" (needs admin once).
setlocal
cd /d "%~dp0"

>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
  echo Requesting administrative privileges to add the VTM Spark camera...
  echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\vtm_spark_cam_uac.vbs"
  echo UAC.ShellExecute "cmd.exe", "/c ""%~s0""", "", "runas", 1 >> "%temp%\vtm_spark_cam_uac.vbs"
  "%temp%\vtm_spark_cam_uac.vbs"
  del "%temp%\vtm_spark_cam_uac.vbs"
  exit /B
)

echo Installing virtual camera: VTM Spark
regsvr32 /s "UnityCaptureFilter32.dll" "/i:UnityCaptureName=VTM Spark"
if errorlevel 1 (
  echo ERROR: 32-bit filter register failed
  exit /B 1
)
regsvr32 /s "UnityCaptureFilter64.dll" "/i:UnityCaptureName=VTM Spark"
if errorlevel 1 (
  echo ERROR: 64-bit filter register failed
  exit /B 1
)
echo Done. Device name: VTM Spark
exit /B 0
