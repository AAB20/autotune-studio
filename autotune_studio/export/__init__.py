"""Adapter / merged model export (light import; heavy deps are loaded inside ``export_model``)."""

from __future__ import annotations

from .model_export import export_model

__all__ = ["export_model"]
