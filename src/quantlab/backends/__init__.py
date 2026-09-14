from __future__ import annotations

from types import ModuleType

from . import gguf_backend, llmcompressor_backend, modelopt_backend

_REGISTRY: dict[str, ModuleType] = {
    "llmcompressor": llmcompressor_backend,
    "modelopt": modelopt_backend,
    "gguf": gguf_backend,
}


def get_backend(name: str) -> ModuleType:
    try:
        return _REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"unknown backend {name!r}, must be one of {sorted(_REGISTRY)}") from e
