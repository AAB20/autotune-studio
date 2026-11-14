"""AutoTune Studio — browser-based LLM fine-tuning for notebook runtimes with zero row configuration."""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["Studio", "__version__"]


def __getattr__(name: str):  # lazy import keeps `import autotune_studio` light
    if name == "Studio":
        from .studio import Studio

        return Studio
    raise AttributeError(name)
