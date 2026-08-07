@echo off
REM Local Windows launcher. On the pod, prefer ./train_ref.sh (builds cache there).
setlocal
cd /d "%~dp0"

if not defined SEND2POD_ROOT (
  set "SEND2POD_ROOT=%~dp0.."
)

set "WORKDIR=%SEND2POD_ROOT%\workdir\ref_30m"
set "DATA_DIR=%SEND2POD_ROOT%\data\train_512"
set "LATENT_CACHE=%DATA_DIR%\latents_cache.pt"

if not exist "%WORKDIR%" mkdir "%WORKDIR%"

echo SEND2POD_ROOT=%SEND2POD_ROOT%
echo workdir=%WORKDIR%
echo data=%DATA_DIR%

if not exist "%LATENT_CACHE%" (
  echo Building latent cache on this machine...
  python -m scripts.cache_latents --data-dir "%DATA_DIR%" --output "%LATENT_CACHE%" --batch-size 16 --hue-augments 2 --device cuda
)

python -m training.main ^
  --config configs/scratch_30m_ref.py ^
  --workdir "%WORKDIR%" ^
  --data_dir "%DATA_DIR%" ^
  --latent_cache "%LATENT_CACHE%" ^
  --grad_ckpt

endlocal
