"""Alinhamento offline entre a CIFRA e o ÁUDIO, batida por batida.

O baixo em modo cifra precisa de duas coisas: a harmonia exata da cifra e o
tempo exato do áudio. A cifra (Cifra Club) diz QUAIS acordes vêm e em que
ORDEM, mas não quantos compassos cada um dura; o áudio diz QUANDO as batidas
acontecem. Como o app toca um arquivo, dá para ouvir a música inteira antes:

1. ``compute_beat_chroma`` resume a harmonia de cada batida (cromagrama médio
   entre uma batida e a seguinte).
2. ``align_chart_to_beats`` encaixa a sequência de acordes da cifra nessas
   batidas por programação dinâmica (Viterbi): cada batida recebe um evento da
   cifra, a ordem da cifra é respeitada e as trocas preferem o tempo 1.

O resultado (``HarmonyTimeline``) diz, para qualquer instante do arquivo, qual
acorde DA CIFRA soa e onde caem as batidas e os tempos 1 — sem esperar o áudio
confirmar nada durante a reprodução.
"""

from dataclasses import dataclass, field
import logging
import math
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from app.music.chord_chart import parse_chord
from app.music.theory import note_to_pc
from app.music.transposition import transpose_chord_symbol

logger = logging.getLogger(__name__)

_ANALYSIS_SR = 22050
_HOP = 512

# Intervalos (semitons acima da tônica) e pesos do molde de cada acorde.
_QUALITY_INTERVALS = {
    "major": ((0, 1.0), (4, 0.8), (7, 0.8)),
    "minor": ((0, 1.0), (3, 0.8), (7, 0.8)),
    "dim": ((0, 1.0), (3, 0.8), (6, 0.8)),
    "aug": ((0, 1.0), (4, 0.8), (8, 0.8)),
    "sus4": ((0, 1.0), (5, 0.8), (7, 0.8)),
    "sus2": ((0, 1.0), (2, 0.8), (7, 0.8)),
}
_EXTENSION_INTERVALS = {
    "7": ((10, 0.45),), "9": ((10, 0.4), (2, 0.3)), "11": ((10, 0.4), (5, 0.3)),
    "13": ((10, 0.4), (9, 0.3)), "maj7": ((11, 0.45),), "maj9": ((11, 0.4), (2, 0.3)),
    "m7": ((10, 0.45),), "m9": ((10, 0.4), (2, 0.3)), "m11": ((10, 0.4), (5, 0.3)),
    "dim7": ((9, 0.45),), "m7(b5)": ((10, 0.45),), "add9": ((2, 0.35),), "6": ((9, 0.4),),
}

# Custos (em unidades de "semelhança por batida") de uma troca de acorde,
# conforme a posição da batida no compasso.
_CHANGE_COST_DOWNBEAT = 0.10
_CHANGE_COST_HALF_BAR = 0.35
_CHANGE_COST_OFFBEAT = 0.90
_SKIP_EVENT_COST = 1.50       # pular um acorde da cifra
_JUMP_SECTION_COST = 2.50     # voltar/avançar para o início de outra seção
_OUTSIDE_REWARD_SILENT = 0.60  # antes/depois da música, em silêncio
_OUTSIDE_REWARD_SOUND = 0.30   # antes/depois da música, com som (introdução fora da cifra)

MIN_CONFIDENCE = 0.55


def chord_template(symbol: str) -> Optional[np.ndarray]:
    """Molde de 12 classes de altura (normalizado) para um símbolo de acorde."""
    try:
        parsed = parse_chord(symbol)
    except Exception:
        return None
    root = note_to_pc(parsed.root)
    if root < 0:
        return None
    template = np.zeros(12, dtype=np.float64)
    for interval, weight in _QUALITY_INTERVALS.get(parsed.quality, _QUALITY_INTERVALS["major"]):
        template[(root + interval) % 12] = max(template[(root + interval) % 12], weight)
    for interval, weight in _EXTENSION_INTERVALS.get(parsed.extension, ()):
        template[(root + interval) % 12] = max(template[(root + interval) % 12], weight)
    bass = note_to_pc(parsed.bass_note) if parsed.bass_note else root
    if bass >= 0 and bass != root:
        template[bass] = max(template[bass], 0.6)
    return template / np.linalg.norm(template)


@dataclass
class BeatChroma:
    """Harmonia média de cada batida do arquivo (pré-análise, independente da cifra)."""
    beat_times: np.ndarray            # início de cada batida (B+1 valores: o último fecha a última)
    chroma: np.ndarray                # (B, 12), normalizado por linha
    energy: np.ndarray                # (B,) RMS médio da batida
    duration: float

    @property
    def beat_count(self) -> int:
        return int(self.chroma.shape[0])

    def silent_mask(self) -> np.ndarray:
        if self.energy.size == 0:
            return np.zeros(0, dtype=bool)
        loud = float(np.percentile(self.energy, 90))
        return self.energy < max(1e-5, 0.05 * loud)


def complete_beat_grid(beat_times: Sequence[float], bpm: float, duration: float) -> np.ndarray:
    """Estende a grade do beat_track até cobrir o arquivo todo e preenche falhas internas."""
    beats = np.asarray(beat_times, dtype=np.float64)
    beats = beats[(beats >= 0.0) & (beats <= duration)]
    period = 60.0 / bpm if bpm > 0 else (float(np.median(np.diff(beats))) if len(beats) > 2 else 0.5)
    if len(beats) == 0:
        return np.arange(0.0, duration + period, period)
    filled = [float(beats[0])]
    for t in beats[1:]:
        gap = float(t) - filled[-1]
        missing = int(round(gap / period)) - 1
        if missing >= 1:
            step = gap / (missing + 1)
            filled.extend(filled[-1] + step * (i + 1) for i in range(missing))
        filled.append(float(t))
    start = filled[0]
    before = []
    while start - period >= 0.0:
        start -= period
        before.append(start)
    grid = before[::-1] + filled
    while grid[-1] + period <= duration + period:
        grid.append(grid[-1] + period)
    return np.asarray(grid, dtype=np.float64)


def compute_beat_chroma(mono_audio: np.ndarray, sample_rate: int,
                        beat_times: Sequence[float], bpm: float) -> Optional[BeatChroma]:
    """Cromagrama médio por batida. Retorna None se a análise não for possível."""
    try:
        import librosa
        from scipy.signal import resample_poly
    except Exception:
        return None
    audio = np.asarray(mono_audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if len(audio) < sample_rate * 2 or sample_rate <= 0:
        return None
    duration = len(audio) / float(sample_rate)
    if sample_rate != _ANALYSIS_SR:
        g = math.gcd(int(sample_rate), _ANALYSIS_SR)
        audio = resample_poly(audio, _ANALYSIS_SR // g, int(sample_rate) // g)
    audio = audio.astype(np.float32)
    try:
        chroma = librosa.feature.chroma_cqt(y=audio, sr=_ANALYSIS_SR, hop_length=_HOP)
    except Exception:
        chroma = librosa.feature.chroma_stft(y=audio, sr=_ANALYSIS_SR, hop_length=_HOP, n_fft=4096)
    rms = librosa.feature.rms(y=audio, hop_length=_HOP)[0]
    frames = min(chroma.shape[1], len(rms))
    chroma, rms = chroma[:, :frames], rms[:frames]
    frame_times = librosa.frames_to_time(np.arange(frames), sr=_ANALYSIS_SR, hop_length=_HOP)

    grid = complete_beat_grid(beat_times, bpm, duration)
    if len(grid) < 3:
        return None
    count = len(grid) - 1
    beat_chroma = np.zeros((count, 12), dtype=np.float64)
    energy = np.zeros(count, dtype=np.float64)
    for k in range(count):
        start, end = grid[k], grid[k + 1]
        period = end - start
        # Ignora o começo do ataque (palhetada/transiente) e o fim (antecipações).
        lo, hi = np.searchsorted(frame_times, (start + 0.12 * period, end - 0.05 * period))
        if hi <= lo:
            lo, hi = np.searchsorted(frame_times, (start, end))
        if hi <= lo:
            continue
        vector = chroma[:, lo:hi].mean(axis=1)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            beat_chroma[k] = vector / norm
        energy[k] = float(rms[lo:hi].mean())
    return BeatChroma(beat_times=grid, chroma=beat_chroma, energy=energy, duration=duration)


@dataclass
class Pulse:
    time: float
    beat_index: int
    subdivision: int         # 0 = na batida; 1..n-1 = subdivisões
    bar: int                 # compasso da MÚSICA (0 = anacruse antes do primeiro tempo 1)
    beat_in_bar: int         # 1..beats_per_bar
    period: float            # duração local da batida (s)


@dataclass
class HarmonyTimeline:
    """Acorde da cifra para cada batida do áudio, com tempos 1 localizados."""
    beat_times: np.ndarray                 # B+1 fronteiras
    chords: List[Optional[str]]            # B acordes (None = fora da música / silêncio)
    event_indices: List[int] = field(default_factory=list)  # evento da cifra por batida (-1 = fora)
    beats_per_bar: int = 4
    downbeat_phase: int = 0                # batida k é tempo 1 quando (k - fase) % beats_per_bar == 0
    confidence: float = 0.0
    transposition: int = 0

    @property
    def beat_count(self) -> int:
        return len(self.chords)

    def beat_index_at(self, timestamp: float) -> int:
        """Índice da batida que contém ``timestamp`` (-1 antes da primeira)."""
        return int(np.searchsorted(self.beat_times, timestamp, side="right")) - 1

    def chord_at_beat(self, index: int) -> Optional[str]:
        if 0 <= index < len(self.chords):
            return self.chords[index]
        return None

    def chord_at(self, timestamp: float) -> Optional[str]:
        return self.chord_at_beat(self.beat_index_at(timestamp))

    def beat_position(self, index: int) -> Tuple[int, int]:
        """(compasso, tempo) da batida ``index`` na grade de compassos da música."""
        offset = index - self.downbeat_phase
        return offset // self.beats_per_bar + 1, offset % self.beats_per_bar + 1

    def downbeat_times(self) -> np.ndarray:
        idx = np.arange(self.downbeat_phase % self.beats_per_bar, self.beat_count, self.beats_per_bar)
        return self.beat_times[idx]

    def pulses(self, start: float, end: float, interval_beats: float) -> Iterator[Pulse]:
        """Pulsos da grade rítmica do baixo em [start, end), presos às batidas do áudio."""
        if end <= start or self.beat_count == 0:
            return
        first = max(0, self.beat_index_at(start))
        subdivisions = max(1, int(round(1.0 / interval_beats))) if interval_beats < 1.0 else 1
        step = max(1, min(self.beats_per_bar, int(round(interval_beats)))) if interval_beats >= 1.0 else 1
        for k in range(first, self.beat_count):
            beat_start = float(self.beat_times[k])
            if beat_start >= end:
                break
            period = float(self.beat_times[k + 1] - beat_start)
            bar, beat_in_bar = self.beat_position(k)
            if (beat_in_bar - 1) % step:
                continue
            for i in range(subdivisions):
                t = beat_start + period * i / subdivisions
                if start <= t < end:
                    yield Pulse(t, k, i, bar, beat_in_bar, period)


def _event_templates(chords: Sequence[str], shift: int) -> Tuple[np.ndarray, List[str]]:
    symbols = [transpose_chord_symbol(c, shift) if shift else c for c in chords]
    templates = np.zeros((len(symbols), 12), dtype=np.float64)
    for j, symbol in enumerate(symbols):
        template = chord_template(symbol)
        if template is not None:
            templates[j] = template
    return templates, symbols


def _viterbi(similarity: np.ndarray, silent: np.ndarray, phase: int, beats_per_bar: int,
             section_starts: np.ndarray) -> Tuple[float, np.ndarray]:
    """Melhor caminho monotônico pela cifra. Estados: 0=antes, 1..J=eventos, J+1=depois."""
    beats, events = similarity.shape
    states = events + 2
    neg = -1e18
    outside = np.where(silent, _OUTSIDE_REWARD_SILENT, _OUTSIDE_REWARD_SOUND)
    emission = np.empty((beats, states))
    emission[:, 0] = outside
    emission[:, 1:events + 1] = similarity
    emission[:, events + 1] = outside

    def change_cost(beat: int) -> float:
        position = (beat - phase) % beats_per_bar
        if position == 0:
            return _CHANGE_COST_DOWNBEAT
        if beats_per_bar % 2 == 0 and position == beats_per_bar // 2:
            return _CHANGE_COST_HALF_BAR
        return _CHANGE_COST_OFFBEAT

    score = np.full(states, neg)
    score[0] = emission[0, 0]
    score[1] = emission[0, 1] - change_cost(0)
    back = np.zeros((beats, states), dtype=np.int32)
    back[0] = np.arange(states)
    ev = np.arange(1, events + 1)
    starts = section_starts + 1  # índices de estado
    for b in range(1, beats):
        change = change_cost(b)
        best = score.copy()
        arg = np.arange(states)
        # avança um evento (inclui antes -> primeiro evento)
        cand = score[:-2] - change
        take = cand > best[1:-1]
        best[1:-1] = np.where(take, cand, best[1:-1])
        arg[1:-1] = np.where(take, np.arange(0, events), arg[1:-1])
        # pula um evento da cifra
        if events >= 3:
            cand = score[1:events - 1] - change - _SKIP_EVENT_COST   # estados 1..J-2
            idx = np.arange(3, events + 1)                           # estados 3..J
            take = cand > best[idx]
            best[idx] = np.where(take, cand, best[idx])
            arg[idx] = np.where(take, np.arange(1, events - 1), arg[idx])
        # salta para o início de uma seção (repetição não escrita / seção omitida)
        if len(starts):
            event_scores = score[1:events + 1]
            origin = int(np.argmax(event_scores)) + 1
            cand_value = score[origin] - change - _JUMP_SECTION_COST
            for s in starts:
                if s != origin + 1 and cand_value > best[s]:
                    best[s] = cand_value
                    arg[s] = origin
        # termina (qualquer evento -> depois)
        origin = int(np.argmax(score[ev])) + 1
        if score[origin] - 0.2 > best[events + 1]:
            best[events + 1] = score[origin] - 0.2
            arg[events + 1] = origin
        score = best + emission[b]
        back[b] = arg
    final = int(np.argmax(score))
    path = np.empty(beats, dtype=np.int32)
    path[-1] = final
    for b in range(beats - 1, 0, -1):
        path[b - 1] = back[b, path[b]]
    return float(score[final]), path


def align_chart_to_beats(beat_chroma: BeatChroma, chart_chords: Sequence[str],
                         beats_per_bar: int = 4, capo_semitones: int = 0,
                         section_starts: Sequence[int] = ()) -> Optional[HarmonyTimeline]:
    """Encaixa a sequência de acordes da cifra nas batidas do áudio.

    Retorna None quando não há material suficiente. ``confidence`` mede o quanto
    o áudio concorda com o caminho escolhido; abaixo de ``MIN_CONFIDENCE`` o
    chamador deve preferir o acompanhamento ao vivo.
    """
    chords = [c for c in chart_chords if c and c != "--"]
    if beat_chroma is None or beat_chroma.beat_count < 4 or not chords:
        return None
    beats_per_bar = max(2, int(beats_per_bar))
    silent = beat_chroma.silent_mask()
    starts = np.asarray(sorted({int(s) for s in section_starts if 0 < int(s) < len(chords)}),
                        dtype=np.int64)
    best = None
    shifts = [0] + ([int(capo_semitones) % 12] if int(capo_semitones) % 12 else [])
    for shift in shifts:
        templates, symbols = _event_templates(chords, shift)
        similarity = np.clip(beat_chroma.chroma @ templates.T, 0.0, 1.0)
        similarity[silent] = 0.0
        for phase in range(beats_per_bar):
            total, path = _viterbi(similarity, silent, phase, beats_per_bar, starts)
            if best is None or total > best[0]:
                best = (total, path, phase, shift, symbols, similarity)
    _, path, phase, shift, symbols, similarity = best
    events = len(symbols)
    event_indices = [int(s) - 1 if 1 <= s <= events else -1 for s in path]
    timeline_chords: List[Optional[str]] = [
        symbols[j] if j >= 0 and not silent[k] else None for k, j in enumerate(event_indices)]

    # Confiança: semelhança média do caminho e quantas batidas escolheram o
    # acorde que melhor explica o áudio entre os acordes da própria cifra.
    inside = [k for k, j in enumerate(event_indices) if j >= 0 and not silent[k]]
    if not inside:
        return None
    chosen = np.array([similarity[k, event_indices[k]] for k in inside])
    best_any = similarity[inside].max(axis=1)
    agreement = float(np.mean(chosen >= best_any - 0.05))
    confidence = float(0.5 * np.mean(chosen) + 0.5 * agreement)
    timeline = HarmonyTimeline(
        beat_times=beat_chroma.beat_times, chords=timeline_chords,
        event_indices=event_indices, beats_per_bar=beats_per_bar,
        downbeat_phase=int(phase), confidence=confidence, transposition=int(shift))
    logger.info("Cifra alinhada ao áudio: %d batidas, fase do tempo 1=%d, transposição=%+d, "
                "confiança=%.2f", timeline.beat_count, phase, shift, confidence)
    return timeline
