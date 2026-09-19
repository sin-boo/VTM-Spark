# -*- mode: python ; coding: utf-8 -*-

"""Minimal PyInstaller spec: launcher only (no torch / transformers).



Memory-safe. The ML stack lives in dist/VTMNoble/runtime/ and is NOT frozen.

"""



from pathlib import Path



ROOT = Path(SPECPATH).resolve().parents[1]  # backend/packaging -> install root



a = Analysis(

    [str(ROOT / "backend" / "packaging" / "launcher.py")],

    pathex=[str(ROOT / "backend" / "packaging")],

    binaries=[],

    datas=[],

    hiddenimports=[],

    hookspath=[],

    hooksconfig={},

    runtime_hooks=[],

    excludes=[

        "torch",

        "torchvision",

        "torchaudio",

        "transformers",

        "diffusers",

        "accelerate",

        "safetensors",

        "huggingface_hub",

        "tokenizers",

        "cv2",

        "ultralytics",

        "mediapipe",

        "onnxruntime",

        "numpy",

        "PIL",

        "fastapi",

        "uvicorn",

        "webview",

        "backend",

        "tkinter",

        "matplotlib",

        "scipy",

        "pandas",

        "sklearn",

        "datasets",

        "tensorflow",

        "tensorboard",

        "IPython",

        "pytest",

    ],

    noarchive=False,

)



pyz = PYZ(a.pure)



exe = EXE(

    pyz,

    a.scripts,

    a.binaries,

    a.datas,

    [],

    name="VTMNoble",

    debug=False,

    bootloader_ignore_signals=False,

    strip=False,

    upx=False,

    console=True,

    disable_windowed_traceback=False,

)

