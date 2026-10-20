"""Voice Activity Detection."""
from __future__ import annotations

from .backends import EnergyVad, SileroVad, WebRtcVad, cache_dir
from .core import (
    StateMachine,
    VadBackend,
    VadParams,
    VadState,
    calculate_audio_volume,
    exp_smoothing,
)
from .processor import VadProcessor

__all__ = [
    "EnergyVad",
    "SileroVad",
    "StateMachine",
    "VadBackend",
    "VadParams",
    "VadProcessor",
    "VadState",
    "WebRtcVad",
    "cache_dir",
    "calculate_audio_volume",
    "exp_smoothing",
]
