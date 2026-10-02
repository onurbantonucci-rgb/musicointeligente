"""Detecção de Andamento (BPM), Rastreamento de Batidas (Beat Tracking) e Métricas Rítmicas."""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
import librosa
from scipy.signal import decimate as _scipy_decimate

logger = logging.getLogger(__name__)


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
    # Próxima batida da pré-análise a partir deste instante (o analisador olha ~90 ms
    # à frente, então um ataque recém-detectado costuma estar LOGO DEPOIS do instante).
    next_beat_timestamp: Optional[float] = None


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
        # Compasso ouvido na faixa: 6 = compasso composto (grade em colcheias, ex.: 6/8).
        # None = compasso simples (quem usa a grade mantém a sua fórmula, ex.: 4/4).
        self._audio_beats_per_bar: Optional[int] = None

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def beats_per_bar(self) -> Optional[int]:
        """Batidas da grade por compasso detectadas no ÁUDIO (6 em 6/8), ou None."""
        return self._audio_beats_per_bar

    @property
    def beat_times(self) -> np.ndarray:
        return self._beat_times

    # Compasso composto: o pulso escolhido pelo rastreador quase não se repete na
    # música (autocorrelação baixa), mas o DOBRO dele se repete muito — o pulso real
    # é o dobro (semínima pontuada) e ele se divide em 3 colcheias.
    _COMPOUND_PULSE_RATIO = 0.6
    _COMPOUND_MIN_DOUBLE = 0.4

    @classmethod
    def _compound_eighth_bpm(cls, envelope: np.ndarray, sample_rate: int,
                             beat_frames) -> Optional[float]:
        """BPM das colcheias quando a faixa é em compasso composto (6/8), senão None."""
        frames = np.asarray(beat_frames)
        if len(frames) < 8 or len(envelope) < 64:
            return None
        period = float(np.median(np.diff(frames)))      # em quadros do envelope (hop 512)
        if period <= 2:
            return None
        signal = np.asarray(envelope, dtype=np.float64)
        # Suaviza (±2 quadros): pulsos que caem a um quadro de distância contam juntos.
        signal = np.convolve(signal, np.hanning(5) / np.hanning(5).sum(), mode="same")
        signal = signal - signal.mean()
        size = int(round(period * 2)) + 3
        if size >= len(signal):
            return None
        ac = librosa.autocorrelate(signal, max_size=size + 1)
        if ac[0] <= 0:
            return None
        ac = ac / ac[0]

        def at(lag: float) -> float:
            i = int(round(lag))
            return float(np.max(ac[max(1, i - 1):i + 2]))

        single, double = at(period), at(2 * period)
        if double < cls._COMPOUND_MIN_DOUBLE or single >= cls._COMPOUND_PULSE_RATIO * double:
            return None
        hop_seconds = 512.0 / sample_rate
        eighth_seconds = (2.0 * period / 3.0) * hop_seconds
        return 60.0 / eighth_seconds

    def reset_live_state(self) -> None:
        """Zera só o rastreamento ao vivo; a grade pré-analisada da faixa continua."""
        self._last_beat_idx = -1
        self._previous_energy = 0.0
        self._live_pulse = None

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
            # Decimação para ~22050 Hz acelera o cálculo de onset STFT em mais de 3x mantendo precisão rítmica.
            # scipy.signal.decimate aplica um filtro anti-aliasing antes de subamostrar; um
            # simples audio[::2] deixaria conteúdo acima de Nyquist se dobrar de volta no
            # espectro (aliasing), contaminando o envelope de onset usado para achar as batidas.
            if sample_rate > 22050:
                hop = 2
                decimated_audio = _scipy_decimate(audio_data, hop).astype(np.float32)
                eff_sr = sample_rate // hop
            else:
                decimated_audio = audio_data
                eff_sr = sample_rate

            # Extrair envelope de onset e calcular BPM e frames de batidas
            envelope = librosa.onset.onset_strength(y=decimated_audio, sr=eff_sr)
            tempo, beat_frames = librosa.beat.beat_track(onset_envelope=envelope, sr=eff_sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=eff_sr)
            self._audio_beats_per_bar = None
            tight_kwargs = {}
            eighth_bpm = self._compound_eighth_bpm(envelope, eff_sr, beat_frames)
            if eighth_bpm is not None:
                # Compasso composto (6/8, 12/8): a grade passa a ser de colcheias e o
                # compasso tem 6 delas. Tudo vem do áudio, nunca do BPM escrito na cifra.
                tempo, beat_frames = librosa.beat.beat_track(
                    onset_envelope=envelope, sr=eff_sr, start_bpm=eighth_bpm, tightness=400)
                beat_times = librosa.frames_to_time(beat_frames, sr=eff_sr)
                tight_kwargs = {"start_bpm": eighth_bpm}
                self._audio_beats_per_bar = 6
            # Levadas sincopadas fazem o rastreador "escorregar" meia batida e seguir
            # nos contratempos. Um rastreamento rígido (não escorrega, mas acompanha mal
            # variações de andamento) serve de referência de FASE para corrigir só os
            # trechos escorregados, mantendo a flexibilidade no resto da música.
            _, tight_frames = librosa.beat.beat_track(onset_envelope=envelope, sr=eff_sr,
                                                      tightness=1600, **tight_kwargs)
            beat_times = self._repair_half_beat_slips(
                beat_times, librosa.frames_to_time(tight_frames, sr=eff_sr))
            self._beat_times = np.asarray(beat_times, dtype=np.float32)
            # A fase do beat_track tem viés sistemático (~+30 ms com quadros de 23 ms):
            # alinhamos a grade ao ataque real, medido num envelope de alta resolução.
            self._beat_times = self._refine_beat_phase(decimated_audio, eff_sr, self._beat_times)
            # O tempo do librosa é quantizado na grade do tempograma (ex.: 117.45 em vez
            # de 120). As posições das batidas são bem mais precisas: a inclinação da
            # regressão tempo-de-batida × índice dá o período real e elimina a deriva.
            self._bpm = self._refine_bpm(self._beat_times, float(np.mean(tempo)))
            self._last_beat_idx = -1
            logger.info("BPM detectado: %.1f | Total de batidas: %d", self._bpm, len(self._beat_times))
            return self._bpm
        except Exception as e:
            logger.warning("Erro ao detectar tempo via librosa, usando fallback: %s", e)
            return self._analyze_audio_fallback(audio_data, sample_rate)

    @staticmethod
    def _refine_beat_phase(audio: np.ndarray, sample_rate: int,
                           beat_times: np.ndarray) -> np.ndarray:
        """Desloca a grade de batidas para o ataque real do áudio (correção de fase global).

        O ``librosa.beat.beat_track`` marca as batidas com resolução de um quadro
        (~23 ms) e com atraso sistemático em relação ao início do ataque. Medimos,
        num envelope de onset com hop curto (~3 ms), o pico de ataque mais próximo
        de cada batida e aplicamos a MEDIANA desses desvios à grade inteira.
        A grade continua uniforme (o BPM refinado não muda); só a fase é ajustada.
        """
        if len(beat_times) < 4 or sample_rate <= 0 or len(audio) < sample_rate:
            return beat_times
        hop = max(32, int(round(sample_rate * 0.003)))
        envelope = librosa.onset.onset_strength(y=np.asarray(audio, dtype=np.float32),
                                                sr=sample_rate, hop_length=hop)
        times = librosa.times_like(envelope, sr=sample_rate, hop_length=hop)
        window = 0.07
        offsets = []
        for beat in beat_times:
            lo, hi = np.searchsorted(times, (beat - window, beat + window))
            if hi - lo < 3:
                continue
            segment = envelope[lo:hi]
            if float(np.max(segment)) <= 0.0:
                continue
            offsets.append(float(times[lo + int(np.argmax(segment))] - beat))
        # Exige evidência consistente: poucas batidas ou desvios espalhados = sem ajuste.
        if len(offsets) < 4 or float(np.std(offsets)) > 0.035:
            return beat_times
        shift = float(np.median(offsets))
        if abs(shift) < 0.002 or abs(shift) > 0.06:
            return beat_times
        return np.clip(beat_times + shift, 0.0, None).astype(np.float32)

    @staticmethod
    def _repair_half_beat_slips(beats: np.ndarray, reference: np.ndarray,
                                min_run: int = 4) -> np.ndarray:
        """Volta para a fase certa os trechos que ficaram meia batida deslocados.

        Um escorregão aparece como um intervalo brusco de ~0,5 ou ~1,5 período
        entre batidas vizinhas (variação de andamento é gradual e não gera isso).
        Os trechos entre escorregões alternam de fase; ``reference`` (uma grade
        rígida, que não escorrega) decide qual das duas fases é a do tempo.
        """
        beats = np.asarray(beats, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        if len(beats) < 8 or len(reference) < 8:
            return beats
        intervals = np.diff(beats)
        local = np.array([np.median(intervals[max(0, i - 8):i + 9]) for i in range(len(intervals))])
        deviation = intervals / local - 1.0
        # O escorregão pode ser brusco (um intervalo de 0,5/1,5 período) ou espalhado
        # por até 4 intervalos; antes e depois dele o andamento é estável. Variação
        # real de andamento soma muito pouco em 4 batidas.
        slips = []
        transition = set()        # batidas no meio de um escorregão gradual: sem fase definida
        i = 1
        while i < len(deviation) - 1:
            found = False
            if abs(deviation[i - 1]) < 0.12 and abs(deviation[i]) >= 0.12:
                for width in range(1, 5):
                    after = deviation[i + width:i + width + 3]
                    total = float(deviation[i:i + width].sum())
                    if 0.35 <= abs(total) <= 0.65 and len(after) and np.all(np.abs(after) < 0.12):
                        slips.append(i + width - 1)
                        transition.update(range(i + 1, i + width))
                        i += width
                        found = True
                        break
            if not found:
                i += 1
        slips = np.asarray(slips, dtype=int)
        if not len(slips):
            return beats
        parity = np.zeros(len(beats), dtype=int)
        for s in slips:
            parity[s + 1:] ^= 1

        def phase_error(times: np.ndarray) -> np.ndarray:
            idx = np.clip(np.searchsorted(reference, times), 1, len(reference) - 1)
            nearest = np.where(np.abs(reference[idx - 1] - times) < np.abs(reference[idx] - times),
                               reference[idx - 1], reference[idx])
            period = float(np.median(np.diff(reference)))
            return (times - nearest) / period

        errors = phase_error(beats)
        mean_abs = [float(np.mean(np.abs(errors[parity == p]))) if np.any(parity == p) else 1.0
                    for p in (0, 1)]
        wrong = int(np.argmax(mean_abs))
        if abs(mean_abs[0] - mean_abs[1]) < 0.15:
            return beats                       # sem evidência clara de qual fase é a certa
        repaired = beats.copy()
        segment_starts = [0] + [int(s) + 1 for s in slips]
        segment_ends = [int(s) + 1 for s in slips] + [len(beats)]
        for a, b in zip(segment_starts, segment_ends):
            if parity[a] != wrong or b - a < min_run:
                continue
            direction = np.sign(np.median(errors[a:b])) or 1.0
            repaired[a:b] -= direction * 0.5 * local[min(a, len(local) - 1)]
        # Nas emendas surgem batidas duplicadas (meio período de distância) ou
        # lacunas (um período e meio): remove as primeiras, preenche as segundas.
        repaired = np.delete(repaired, sorted(transition))
        repaired.sort()
        typical = float(np.median(intervals))
        cleaned = [repaired[0]]
        for b in repaired[1:]:
            gap = b - cleaned[-1]
            if gap < 0.6 * typical:
                continue
            missing = int(round(gap / typical)) - 1
            for m in range(missing):
                cleaned.append(cleaned[-1] + gap / (missing + 1))
            cleaned.append(b)
        return np.asarray(cleaned)
    @staticmethod
    def _refine_bpm(beat_times: np.ndarray, coarse_bpm: float) -> float:
        """Refina o BPM pela regressão linear das batidas, com fallback no valor bruto.

        Só aceita o refinamento se ficar perto do valor bruto (±8%): protege contra
        grades de batida irregulares (rubato, falhas) que distorceriam a inclinação.
        """
        if len(beat_times) < 8 or coarse_bpm <= 0:
            return coarse_bpm
        idx = np.arange(len(beat_times), dtype=np.float64)
        slope, _ = np.polyfit(idx, beat_times.astype(np.float64), 1)
        if slope <= 0:
            return coarse_bpm
        refined = 60.0 / slope
        if abs(refined - coarse_bpm) / coarse_bpm > 0.08:
            return coarse_bpm
        return float(refined)

    def _analyze_audio_fallback(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Rastreador NumPy usado quando a biblioteca principal não está disponível."""
        audio = np.asarray(audio_data, dtype=np.float64)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        if len(audio) < sample_rate * 2:
            self._bpm = 0.0
            return 0.0

        decimation = max(1, int(np.ceil(sample_rate / 22050.0)))
        signal = _scipy_decimate(audio, decimation) if decimation > 1 else audio
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
        logger.info("BPM por fallback: %.1f | Total de batidas: %d", self._bpm, len(self._beat_times))
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
            earliest = timestamp - 0.04
            first_beat = float(self._beat_times[0])
            if earliest < first_beat:
                # O beat_track costuma pular o primeiro ataque da música. A grade é
                # uniforme (BPM refinado), então a estendemos para trás a partir da
                # primeira batida detectada, mantendo a mesma fase.
                period = 60.0 / self._bpm
                steps_back = math.floor((first_beat - earliest) / period)
                next_beat = first_beat - steps_back * period
            else:
                next_idx = int(np.searchsorted(self._beat_times, earliest))
                next_beat = (float(self._beat_times[next_idx])
                             if next_idx < len(self._beat_times) else None)

            return TempoResult(
                bpm=round(self._bpm, 1),
                is_beat=is_beat,
                beat_number=beat_number,
                beat_position=float(beat_pos),
                time_signature="4/4",
                confidence=0.88,
                beat_timestamp=float(self._beat_times[closest_idx]) if is_beat else None,
                next_beat_timestamp=next_beat,
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
