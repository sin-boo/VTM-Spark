VTM Spark (virtual webcam)

Bundled DirectShow filters from Unity Capture (schellingb/UnityCapture):
  https://github.com/schellingb/UnityCapture

Registered by install.bat / first Virtual camera use as:
  "VTM Spark"

"VTM Spark Camera Setup.exe" does the registering (source:
backend\packaging\cam-setup.cs, built by build-run.ps1). Windows' admin
prompt shows its name and the VTM Spark logo. Started without admin it
asks for it itself. Exit codes: 0 done, 1223 prompt declined, 2 a filter
is missing, 3 registering failed.

Appears in OBS, Discord, Zoom, etc. as a Video Capture Device.
Python pushes frames via pyvirtualcam (unitycapture backend).
