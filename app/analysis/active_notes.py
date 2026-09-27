"""Persistência curta das 12 classes de nota para áudio polifônico.

O pitch f0 permanece um diagnóstico separado: acordes e arpejos usam o
cromagrama, que reúne simultaneamente todas as classes de nota audíveis.
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from app.music.theory import PITCH_CLASSES, get_scale_notes
from app.analysis.chord_detector import _parse_key_hint


@dataclass(frozen=True)
class ActiveNoteState:
    chroma: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    notes: Dict[str, float] = field(default_factory=dict)


class ActiveNoteTracker:
    """Retém parciais de um arpejo por instantes, sem inventar notas ausentes."""

    def __init__(self, decay_seconds: float = 0.23):
        self._decay_seconds = max(0.05, float(decay_seconds))
        self.reset()

    def reset(self) -> None:
        self._salience = np.zeros(12, dtype=np.float32)
        self._active: set[int] = set()
        self._last_timestamp: float | None = None

    def update(self, chroma: np.ndarray, timestamp: float, key: str = "--",
               audio_activity: float = 1.0) -> ActiveNoteState:
        raw = np.asarray(chroma, dtype=np.float32)
        if raw.shape != (12,) or not np.all(np.isfinite(raw)):
            self.reset()
            return ActiveNoteState()
        timestamp = float(timestamp)
        if (self._last_timestamp is not None and
                (timestamp < self._last_timestamp or timestamp - self._last_timestamp > 0.8)):
            self.reset()
        if audio_activity < 0.005 or float(np.max(raw)) < 0.05:
            self.reset()
            self._last_timestamp = timestamp
            return ActiveNoteState()

        raw = np.maximum(0.0, raw)
        raw /= max(1e-9, float(np.max(raw)))
        # Um ataque largo/percussivo energiza quase todas as classes e não
        # contém informação harmônica suficiente para nomear notas ou acorde.
        if int(np.count_nonzero(raw >= 0.32)) >= 8:
            raw = np.zeros(12, dtype=np.float32)
        delta = (max(0.0, timestamp - self._last_timestamp)
                 if self._last_timestamp is not None else 0.0)
        decay = float(np.exp(-delta / self._decay_seconds)) if delta else 0.0
        self._salience = np.maximum(raw, self._salience * decay)
        peak = float(np.max(self._salience))
        if peak > 1e-9:
            self._salience /= peak
        self._salience[self._salience < 0.08] = 0.0
        self._last_timestamp = timestamp

        # A escala desempata notas acusticamente próximas; 8% não consegue
        # vencer um cromatismo forte. Os valores publicados continuam acústicos.
        key_info = _parse_key_hint(key)
        scale = (set(get_scale_notes(*key_info)) if key_info is not None else set())
        selected: set[int] = set()
        for index, name in enumerate(PITCH_CLASSES):
            acoustic = float(self._salience[index])
            vote = acoustic * (1.08 if name in scale else 1.0)
            threshold = 0.23 if index in self._active else 0.32
            if vote >= threshold:
                selected.add(index)
        self._active = selected
        notes = {PITCH_CLASSES[index]: round(float(self._salience[index]), 3)
                 for index in sorted(selected, key=lambda i: -self._salience[i])}
        # O classificador de acordes vê somente classes que passaram pelo
        # limiar de nota ativa. Resíduos harmônicos fracos continuam no
        # cromagrama bruto para diagnóstico, mas não completam uma tríade.
        harmonic_chroma = np.zeros(12, dtype=np.float32)
        for index in selected:
            harmonic_chroma[index] = self._salience[index]
        return ActiveNoteState(harmonic_chroma, notes)
