"""Módulo de Instrumentos Virtuais (Base para a Banda Virtual)."""

from app.instruments.base import VirtualInstrument, VirtualBand
from app.instruments.bass_model import BassNoteEvent, BassDecision, BassPatternType, BassNoteValue, BassHarmonySource
from app.instruments.bass_player import BassPlayer
from app.instruments.registry import VirtualPlayerRegistry, AdaptiveMusicalClock

__all__ = [
    "VirtualInstrument",
    "VirtualBand",
    "BassPlayer",
    "BassNoteEvent",
    "BassDecision",
    "BassPatternType",
    "BassNoteValue",
    "BassHarmonySource",
    "VirtualPlayerRegistry",
    "AdaptiveMusicalClock",
]
