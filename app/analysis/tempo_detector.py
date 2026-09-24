"""Detecção de Andamento (BPM), Rastreamento de Batidas (Beat Tracking) e Métricas Rítmicas."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import librosa


@dataclass
class BeatEvent:
    """Evento temporal de pulso rítmico."""
    timestamp: float = 0.0          # Tempo exato do beat em segundos
    beat_number: int = 1            # Contagem dentro do compasso (1, 2, 3, 4)
    confidence: float = 0.0         # Força/clareza do onset
    bpm_at_beat: float = 120.0      # Andamento instantâneo no instante do beat


@dataclass
class TempoResult:
    """Resultado da análise rítmica de um bloco de áudio."""
    bpm: float = 0.0                # Andamento estimado em batidas por minuto
    is_beat: bool = False           # Verdadeiro se um pulso de batida ocorreu neste frame
    beat_number: int = 1            # Número do tempo (1 a 4, etc.)
    beat_position: float = 0.0      # Posição fracionária no compasso [0.0 a 1.0]
    time_signature: str = "4/4"     # Fórmula de compasso estimada
    confidence: float = 0.0         # Confiança da periodicidade rítmica
    beat_timestamp: Optional[float] = None  # Pulso realmente observado, não a grade prevista


class TempoDetector(ABC):
    """Interface abstrata para detecção de BPM e grade rítmica."""

    @abstractmethod
    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int) -> float:
        pass


class BeatTracker(ABC):
    """Interface abstrata para rastreamento de batidas em tempo real."""

    @abstractmethod
    def get_tempo_at_time(self, timestamp: float, tolerance_sec: float = 0.075) -> TempoResult:
        pass


class RhythmDetector(ABC):
    """Interface abstrata para análise de padrões rítmicos."""

    @abstractmethod
    def analyze_subdivision(self, audio_data: np.ndarray, sample_rate: int) -> str:
        pass


class OnsetTempoDetector(TempoDetector):
    """Detector de BPM e Beat Tracker baseado em envelope de transientes (onset strength).
    
    Combina pré-análise da grade de batidas e sincronização em tempo real frame a frame.
    """

    def __init__(self):
        self._bpm: float = 0.0
        self._beat_times: np.ndarray = np.array([], dtype=np.float32)
        self._last_beat_idx: int = -1
        self._beats_per_measure: int = 4
        self._previous_energy: float = 0.0
        self._live_pulse: Optional[float] = None

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def beat_times(self) -> np.ndarray:
        return self._beat_times

    def reset(self) -> None:
        self._bpm = 0.0
        self._beat_times = np.array([], dtype=np.float32)
        self._last_beat_idx = -1
        self._previous_energy = 0.0
        self._live_pulse = None

    def observe_chunk(self, audio_data: np.ndarray, sample_rate: int, timestamp: float) -> None:
        """Detecta ataques do sinal atual quando não há grade pré-analisada."""
        if len(self._beat_times) or sample_rate <= 0 or len(audio_data) == 0:
            return
        energy = float(np.sqrt(np.mean(np.square(np.asarray(audio_data, dtype=np.float64)))))
        if (energy > 0.025 and energy > max(0.012, self._previous_energy) * 1.7
                and (self._live_pulse is None or timestamp - self._live_pulse > 0.25)):
            self._live_pulse = timestamp
        self._previous_energy = 0.65 * self._previous_energy + 0.35 * energy

    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Analisa o sinal completo para extrair o BPM global e a grade precisa de tempos."""
        if len(audio_data) < sample_rate * 2 or sample_rate <= 0:
            return 0.0

        try:
            # Decimação para ~22050 Hz acelera o cálculo de onset STFT em mais de 3x mantendo precisão rítmica
            if sample_rate > 22050:
                hop = 2
                decimated_audio = audio_data[::hop]
                eff_sr = sample_rate // hop
            else:
                decimated_audio = audio_data
                eff_sr = sample_rate

            # Extrair envelope de onset e calcular BPM e frames de batidas
            tempo, beat_frames = librosa.beat.beat_track(y=decimated_audio, sr=eff_sr)
            self._bpm = float(np.mean(tempo))
            self._beat_times = librosa.frames_to_time(beat_frames, sr=eff_sr).astype(np.float32)
            self._last_beat_idx = -1
            print(f"[TempoDetector] BPM detectado: {self._bpm:.1f} | Total de batidas: {len(self._beat_times)}")
            return self._bpm
        except Exception as e:
            print(f"[TempoDetector] Erro ao detectar tempo: {e}")
            return self._analyze_audio_fallback(audio_data, sample_rate)

    def _analyze_audio_fallback(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Rastreador NumPy usado quando a biblioteca principal não está disponível."""
        audio = np.asarray(audio_data, dtype=np.float64)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        if len(audio) < sample_rate * 2:
            self._bpm = 0.0
            return 0.0

        decimation = max(1, int(np.ceil(sample_rate / 22050.0)))
        signal = audio[::decimation]
        effective_rate = sample_rate / decimation
        frame_size = 1024
        hop_size = 256
        frame_count = 1 + (len(signal) - frame_size) // hop_size
        if frame_count < 4:
            self._bpm = 0.0
            return 0.0

        window = np.hanning(frame_size)
        onset = np.zeros(frame_count, dtype=np.float64)
        previous = None
        for index in range(frame_count):
            start = index * hop_size
            magnitude = np.abs(np.fft.rfft(signal[start:start + frame_size] * window))
            magnitude /= max(1e-9, float(np.linalg.norm(magnitude)))
            if previous is not None:
                onset[index] = float(np.sum(np.maximum(magnitude - previous, 0.0)))
            previous = magnitude
        onset = np.convolve(onset, np.ones(3, dtype=np.float64) / 3.0, mode="same")
        threshold = float(np.percentile(onset, 75.0))
        candidates = np.where(
            (onset[1:-1] > onset[:-2]) &
            (onset[1:-1] >= onset[2:]) &
            (onset[1:-1] > threshold)
        )[0] + 1
        min_distance = max(1, int(0.28 * effective_rate / hop_size))
        peaks = []
        for candidate in candidates:
            if not peaks or candidate - peaks[-1] >= min_distance:
                peaks.append(int(candidate))
            elif onset[candidate] > onset[peaks[-1]]:
                peaks[-1] = int(candidate)
        if len(peaks) < 4:
            self._bpm = 0.0
            self._beat_times = np.array([], dtype=np.float32)
            return 0.0

        peak_times = np.asarray(peaks, dtype=np.float64) * hop_size / effective_rate
        intervals = np.diff(peak_times)
        intervals = intervals[(intervals >= 0.28) & (intervals <= 1.20)]
        if len(intervals) < 3:
            self._bpm = 0.0
            self._beat_times = np.array([], dtype=np.float32)
            return 0.0
        interval = float(np.median(intervals))
        bpm = 60.0 / interval
        if bpm < 70.0:
            bpm *= 2.0
            interval /= 2.0
        elif bpm > 190.0:
            bpm /= 2.0
            interval *= 2.0
        self._bpm = float(bpm)
        duration = len(audio) / float(sample_rate)
        self._beat_times = np.arange(peak_times[0], duration, interval, dtype=np.float32)
        self._last_beat_idx = -1
        print(f"[TempoDetector] BPM por fallback: {self._bpm:.1f} | "
              f"Total de batidas: {len(self._beat_times)}")
        return self._bpm

    def get_tempo_at_time(self, timestamp: float, tolerance_sec: float = 0.075) -> TempoResult:
        """Determina se o instante atual de reprodução coincide com um pulso de batida."""
        if self._bpm <= 0.0:
            pulse = self._live_pulse if self._live_pulse is not None and abs(timestamp - self._live_pulse) <= tolerance_sec else None
            return TempoResult(bpm=0.0, is_beat=pulse is not None, beat_number=1,
                               beat_position=0.0, confidence=0.75 if pulse is not None else 0.0,
                               beat_timestamp=pulse)

        # Se tivermos a grade pré-calculada
        if len(self._beat_times) > 0:
            # Encontrar a batida mais próxima
            diffs = np.abs(self._beat_times - timestamp)
            closest_idx = int(np.argmin(diffs))
            min_diff = diffs[closest_idx]

            is_beat = (min_diff <= tolerance_sec)
            beat_number = (closest_idx % self._beats_per_measure) + 1

            # Calcular fase fracionária no compasso (0.0 a 1.0)
            measure_duration = (60.0 / self._bpm) * self._beats_per_measure
            beat_pos = (timestamp % measure_duration) / measure_duration

            return TempoResult(
                bpm=round(self._bpm, 1),
                is_beat=is_beat,
                beat_number=beat_number,
                beat_position=float(beat_pos),
                time_signature="4/4",
                confidence=0.88,
                beat_timestamp=float(self._beat_times[closest_idx]) if is_beat else None,
            )

        # Caso contrário (estimativa analítica por tempo)
        beat_interval = 60.0 / self._bpm
        measure_duration = beat_interval * self._beats_per_measure
        beat_phase = (timestamp % beat_interval) / beat_interval
        is_beat = (beat_phase < 0.15 or beat_phase > 0.85)
        current_beat = int((timestamp % measure_duration) / beat_interval) + 1
        beat_pos = (timestamp % measure_duration) / measure_duration

        return TempoResult(
            bpm=round(self._bpm, 1),
            is_beat=is_beat,
            beat_number=current_beat,
            beat_position=float(beat_pos),
            time_signature="4/4",
            confidence=0.75,
            beat_timestamp=self._live_pulse if self._live_pulse is not None
            and abs(timestamp - self._live_pulse) <= tolerance_sec else None,
        )
