"""Detectores de Pitch (Frequência Fundamental f0) para o Virtual Band AI.

Implementa arquitetura plugável com múltiplos algoritmos:
- Autocorrelação Normalizada com Interpolação Parabólica (Rápido, robusto para notas graves e agudas)
- Harmonic Product Spectrum - HPS (Otimizado para timbres com múltiplos harmônicos)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type
import numpy as np

from app.music.theory import hz_to_note_name


@dataclass
class PitchResult:
    """Resultado da análise de pitch de um bloco de áudio."""
    frequency_hz: float = 0.0     # Frequência fundamental f0 em Hz
    note_name: str = "--"         # Nome da nota (ex: "G")
    octave: int = 0               # Oitava (ex: 2 para G2)
    full_note: str = "--"         # Nome completo (ex: "G2")
    cents_deviation: float = 0.0  # Desvio em cents em relação à afinação temperada (-50 a +50)
    confidence: float = 0.0       # Confiança da estimativa (0.0 a 1.0)
    is_voiced: bool = False       # Se o sinal possui periodicidade definida ou é ruído


class PitchDetector(ABC):
    """Interface abstrata para detectores de pitch intercambiáveis."""

    @abstractmethod
    def detect(self, audio_chunk: np.ndarray, sample_rate: int) -> PitchResult:
        """Executa a detecção de pitch no bloco de áudio fornecido."""
        pass

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Nome legível do algoritmo."""
        pass


class AutocorrelationPitchDetector(PitchDetector):
    """Detector de pitch baseado em autocorrelação normalizada com interpolação parabólica.
    
    Funciona tanto para notas puras quanto para instrumentos de cordas e teclas (baixo, violão, guitarra, piano).
    """

    def __init__(self, fmin: float = 35.0, fmax: float = 1800.0, silence_threshold: float = 0.006):
        self._fmin = fmin
        self._fmax = fmax
        self._silence_threshold = silence_threshold

    @property
    def algorithm_name(self) -> str:
        return "Autocorrelação (Recomendado)"

    def detect(self, audio_chunk: np.ndarray, sample_rate: int) -> PitchResult:
        if len(audio_chunk) < 256 or sample_rate <= 0:
            return PitchResult()

        # Verificação de energia (silêncio / ruído de fundo)
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        if rms < self._silence_threshold:
            return PitchResult(is_voiced=False, confidence=0.0)

        # Remover nível DC e aplicar janela de Hanning
        x = audio_chunk - np.mean(audio_chunk)
        w = np.hanning(len(x))
        xw = x * w

        # Autocorrelação rápida via FFT O(N log N)
        n_fft = 2 ** int(np.ceil(np.log2(2 * len(xw))))
        X = np.fft.rfft(xw, n=n_fft)
        r = np.fft.irfft(np.abs(X) ** 2)[:len(xw)]

        if r[0] <= 1e-12:
            return PitchResult(is_voiced=False, confidence=0.0)

        r_norm = r / r[0]

        # Faixa de busca em amostras (lag)
        min_lag = max(2, int(sample_rate / self._fmax))
        max_lag = min(len(r_norm) - 2, int(sample_rate / self._fmin))

        if min_lag >= max_lag:
            return PitchResult(is_voiced=False, confidence=0.0)

        # Encontrar o primeiro vale para evitar a descida imediata do lag 0
        d = np.diff(r_norm)
        valleys = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
        start_search = min_lag
        if len(valleys) > 0:
            start_search = max(min_lag, int(valleys[0]))

        if start_search >= max_lag:
            return PitchResult(is_voiced=False, confidence=0.0)

        search_region = r_norm[start_search:max_lag]
        if len(search_region) == 0:
            return PitchResult(is_voiced=False, confidence=0.0)

        peak_rel = int(np.argmax(search_region))
        peak_idx = start_search + peak_rel
        peak_val = float(r_norm[peak_idx])

        # Se o pico não tiver correlação suficiente, é sinal não-periódico/ruído
        if peak_val < 0.28:
            return PitchResult(is_voiced=False, confidence=peak_val)

        # Refinamento por interpolação parabólica do pico
        k = peak_idx
        if 0 < k < len(r_norm) - 1:
            alpha = float(r_norm[k - 1])
            beta = float(r_norm[k])
            gamma = float(r_norm[k + 1])
            denom = 2.0 * (alpha - 2.0 * beta + gamma)
            delta = (alpha - gamma) / denom if abs(denom) > 1e-9 else 0.0
            if abs(delta) > 1.0:
                delta = 0.0
            true_lag = k + delta
        else:
            true_lag = float(k)

        if true_lag <= 0:
            return PitchResult(is_voiced=False, confidence=0.0)

        f0 = float(sample_rate / true_lag)

        # Conversão para nota musical
        note, octave, cents = hz_to_note_name(f0)
        confidence = float(np.clip(peak_val, 0.0, 1.0))

        return PitchResult(
            frequency_hz=round(f0, 2),
            note_name=note,
            octave=octave,
            full_note=f"{note}{octave}",
            cents_deviation=round(cents, 1),
            confidence=confidence,
            is_voiced=True
        )


class HPSPitchDetector(PitchDetector):
    """Detector de pitch baseado em Harmonic Product Spectrum (HPS).
    
    Adequado para instrumentos polifônicos/timbres com harmônicos bem definidos.
    """

    def __init__(self, fmin: float = 35.0, fmax: float = 1800.0, num_harmonics: int = 4, silence_threshold: float = 0.006):
        self._fmin = fmin
        self._fmax = fmax
        self._num_harmonics = num_harmonics
        self._silence_threshold = silence_threshold

    @property
    def algorithm_name(self) -> str:
        return "Harmonic Product Spectrum (HPS)"

    def detect(self, audio_chunk: np.ndarray, sample_rate: int) -> PitchResult:
        if len(audio_chunk) < 256 or sample_rate <= 0:
            return PitchResult()

        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        if rms < self._silence_threshold:
            return PitchResult(is_voiced=False, confidence=0.0)

        x = audio_chunk - np.mean(audio_chunk)
        xw = x * np.hanning(len(x))
        n_fft = max(8192, len(xw) * 2)
        mag = np.abs(np.fft.rfft(xw, n=n_fft))

        bin_freq = sample_rate / float(n_fft)
        min_bin = max(1, int(self._fmin / bin_freq))
        max_bin = min(len(mag) - 1, int(self._fmax / bin_freq))

        if min_bin >= max_bin:
            return PitchResult(is_voiced=False, confidence=0.0)

        # Multiplicação harmônica (compressão espectral)
        hps = mag.copy()
        for h in range(2, self._num_harmonics + 1):
            dec = mag[::h]
            hps[:len(dec)] *= dec

        search = hps[min_bin:max_bin]
        if len(search) == 0:
            return PitchResult(is_voiced=False, confidence=0.0)

        peak_rel = int(np.argmax(search))
        peak_bin = min_bin + peak_rel

        # Interpolação parabólica no domínio da frequência
        k = peak_bin
        if 0 < k < len(hps) - 1:
            alpha = float(hps[k - 1])
            beta = float(hps[k])
            gamma = float(hps[k + 1])
            denom = 2.0 * (alpha - 2.0 * beta + gamma)
            delta = (alpha - gamma) / denom if abs(denom) > 1e-9 else 0.0
            true_bin = k + delta
        else:
            true_bin = float(k)

        f0 = float(true_bin * bin_freq)
        if f0 < self._fmin or f0 > self._fmax:
            return PitchResult(is_voiced=False, confidence=0.0)

        note, octave, cents = hz_to_note_name(f0)
        mean_val = float(np.mean(search)) + 1e-12
        prominence = float(hps[peak_bin]) / mean_val
        confidence = float(np.clip(prominence / 40.0, 0.0, 1.0))

        return PitchResult(
            frequency_hz=round(f0, 2),
            note_name=note,
            octave=octave,
            full_note=f"{note}{octave}",
            cents_deviation=round(cents, 1),
            confidence=confidence,
            is_voiced=confidence > 0.25
        )


# Dicionário de algoritmos disponíveis para seleção na interface gráfica
AVAILABLE_PITCH_DETECTORS: Dict[str, Type[PitchDetector]] = {
    "Autocorrelação": AutocorrelationPitchDetector,
    "HPS": HPSPitchDetector,
}


def create_pitch_detector(name: str = "Autocorrelação") -> PitchDetector:
    """Instancia o detector de pitch pelo identificador textual."""
    cls = AVAILABLE_PITCH_DETECTORS.get(name, AutocorrelationPitchDetector)
    return cls()
