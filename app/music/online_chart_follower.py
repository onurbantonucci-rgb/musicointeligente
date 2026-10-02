"""Seguidor probabilístico da cifra AO VIVO (filtro de um HMM, só com o passado).

O cursor da cifra avança quando o detector confirma o PRÓXIMO acorde. Isso é
rápido, mas frágil em músicas cíclicas (worship: D#m C#4 B9 F# o tempo todo): se
uma troca não é detectada, o cursor espera aquele acorde voltar — no ciclo
seguinte — e cada troca perdida soma mais um ciclo de atraso.

Este seguidor olha o cromagrama em quadros de ~0,25 s e mantém a probabilidade de
a banda estar em cada acorde da cifra. Cada quadro pode ficar no acorde, avançar
um ou pular um (cifra com acorde a mais ou digitado errado). Como a evidência se
acumula, ele "percebe" que C#4, B9 e F# já soaram e alcança a banda em vez de
esperar. Não usa BPM nem compasso: só tempo em segundos e a ordem da cifra.
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from app.analysis.chart_harmony_aligner import chord_template
from app.music.transposition import transpose_chord_symbol


class OnlineChartFollower:
    FRAME_SECONDS = 0.25
    KAPPA = 8.0          # quanto um quadro de áudio pesa (nitidez da evidência)
    P_ADVANCE = 0.08     # chance, por quadro, de passar ao próximo acorde da cifra
    P_SKIP = 0.005       # chance, por quadro, de pular um acorde da cifra
    MIN_ENERGY = 1e-6

    def __init__(self, event_chords: Sequence[str], capo_semitones: int = 0):
        self._count = len(event_chords)
        written = self._templates(event_chords, 0)
        sounding = (self._templates(event_chords, capo_semitones)
                    if int(capo_semitones) % 12 else None)
        # A gravação pode soar na forma escrita ou com o capotraste: vale a melhor.
        self._domains = [written] + ([sounding] if sounding is not None else [])
        self._log_stay = np.log(1.0 - self.P_ADVANCE - self.P_SKIP)
        self._log_advance = np.log(self.P_ADVANCE)
        self._log_skip = np.log(self.P_SKIP)
        self._logp: Optional[np.ndarray] = None
        self._frame_start: Optional[float] = None
        self._frame_sum = np.zeros(12)

    @staticmethod
    def _templates(chords: Sequence[str], shift: int) -> np.ndarray:
        rows = np.zeros((len(chords), 12))
        for j, chord in enumerate(chords):
            symbol = transpose_chord_symbol(chord, shift) if shift else chord
            template = chord_template(symbol) if symbol and symbol != "--" else None
            if template is not None:
                rows[j] = template
        return rows

    @property
    def active(self) -> bool:
        return self._logp is not None

    def reset(self, event_index: Optional[int] = None) -> None:
        """Recomeça com certeza total em ``event_index`` (None = desligado)."""
        self._frame_start = None
        self._frame_sum = np.zeros(12)
        if event_index is None or self._count == 0:
            self._logp = None
            return
        self._logp = np.full(self._count, -np.inf)
        self._logp[max(0, min(self._count - 1, int(event_index)))] = 0.0

    def floor(self, event_index: int) -> None:
        """O cursor confirmou ``event_index``: a banda não está antes dele."""
        if self._logp is None or not 0 < event_index < self._count:
            return
        if not np.isfinite(self._logp[event_index:]).any():
            self.reset(event_index)
            return
        self._logp[:event_index] = -np.inf
        self._logp -= np.logaddexp.reduce(self._logp[np.isfinite(self._logp)])

    def observe(self, timestamp: float, chroma) -> Optional[Tuple[int, float]]:
        """Soma um quadro de análise; a cada ~0,25 s devolve (acorde mais provável, prob.)."""
        if self._logp is None:
            return None
        if self._frame_start is None:
            self._frame_start = timestamp
        vector = np.asarray(chroma, dtype=np.float64)
        if vector.shape == (12,) and np.all(np.isfinite(vector)):
            self._frame_sum += vector
        if timestamp - self._frame_start < self.FRAME_SECONDS:
            return None
        frame = self._frame_sum
        self._frame_start = timestamp
        self._frame_sum = np.zeros(12)
        return self._step(frame)

    def _step(self, frame: np.ndarray) -> Tuple[int, float]:
        logp = self._logp
        norm = float(np.linalg.norm(frame))
        if norm > self.MIN_ENERGY:
            unit = frame / norm
            similarity = np.max([np.clip(d @ unit, 0.0, 1.0) for d in self._domains], axis=0)
            emission = self.KAPPA * (similarity - similarity.max())
        else:
            emission = np.zeros(self._count)  # silêncio: não é evidência de nada
        new = logp + self._log_stay
        new[1:] = np.logaddexp(new[1:], logp[:-1] + self._log_advance)
        new[2:] = np.logaddexp(new[2:], logp[:-2] + self._log_skip)
        new += emission
        new -= np.logaddexp.reduce(new)
        self._logp = new
        best = int(np.argmax(new))
        return best, float(np.exp(new[best]))
