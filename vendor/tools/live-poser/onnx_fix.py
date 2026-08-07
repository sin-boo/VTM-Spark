"""Force safe ONNX Runtime providers before OpenSeeFace loads models.

Full provider lists (TensorRT + CUDA) crash on this machine with errors like
provider_options index errors / unsupported LeakyRelu fused conv. Prefer CPU
(and CUDA only when explicitly requested and healthy).
"""

from __future__ import annotations

import os

_PATCHED = False


def patch_onnx_providers(*, prefer_cuda: bool = False) -> list[str]:
    """Monkeypatch onnxruntime so OpenSeeFace sessions use a safe provider list."""
    global _PATCHED
    import onnxruntime as ort

    available = list(ort.get_available_providers())
    providers: list[str] = ['CPUExecutionProvider']
    if prefer_cuda and 'CUDAExecutionProvider' in available:
        # Skip TensorRT — it often breaks plugin registration / indexing.
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    if _PATCHED:
        return providers

    _OrigSession = ort.InferenceSession

    def _Session(*args, **kwargs):
        # Always override whatever OpenSeeFace passes (full available list).
        kwargs['providers'] = list(providers)
        kwargs.pop('provider_options', None)
        return _OrigSession(*args, **kwargs)

    ort.InferenceSession = _Session  # type: ignore[misc, assignment]

    # OpenSeeFace calls this private API for its providers list.
    try:
        import onnxruntime.capi._pybind_state as _state

        _state.get_available_providers = lambda: list(providers)  # type: ignore[misc]
    except Exception:
        pass

    os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
    _PATCHED = True
    return providers
