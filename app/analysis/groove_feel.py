"""Escuta do GROOVE: como a banda está tocando, além do BPM.

Duas músicas a 68 BPM podem pedir baixos opostos: uma com acordes longos e muito
espaço, outra com ataques curtos e síncopes. Este módulo mede a levada compasso
a compasso a partir da força dos ataques (envelope de onset) em cada
semicolcheia da grade de batidas do ÁUDIO:

- densidade de ataques, espaço (silêncio rítmico), força dos tempos;
- subdivisão predominante (semínima / colcheia / semicolcheia);
- síncope (ataque em posição fraca sem ataque no tempo forte seguinte);
- energia relativa à música e tendência (crescendo / decrescendo);
- sensação half-time, swing (só quando claramente mensurável), estabilidade;
- pulso grave e densidade percussiva, quando a análise fornece essas bandas.

Nada muda por causa de um quadro ou de um compasso isolado: os números passam
por médias móveis (curta ≈ 2–4 compassos, longa ≈ 8 compassos) e o rótulo do
groove só troca depois de PERSISTIR (histerese).

O mesmo código serve à pré-análise do arquivo (alimentado compasso a compasso,
sem olhar para frente) e ao ao vivo (``LiveOnsetGrid`` monta a grade a partir
do fluxo espectral de cada bloco de áudio e da fase do relógio).
"""

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Optional

import numpy as np

# Rótulos de groove (família rítmica percebida).
SPACIOUS = "SPACIOUS"        # acordes longos, muito espaço
STEADY = "STEADY"            # pulso regular nos tempos
DRIVING = "DRIVING"          # colcheias constantes, energia alta
SYNCOPATED = "SYNCOPATED"    # ataques em contratempo
BUSY = "BUSY"                # muitas semicolcheias
GROOVE_LABELS = (SPACIOUS, STEADY, DRIVING, SYNCOPATED, BUSY)

_ACTIVE = 0.35               # força mínima (0..1) para contar como ataque


@dataclass
class GrooveProfile:
    tempo_bpm: float = 0.0
    meter: str = "4/4"
    pulse_strength: float = 0.0        # 0..1 força média nos tempos
    onset_density: float = 0.0         # 0..1 fração das semicolcheias com ataque
    subdivision: str = "QUARTER"       # QUARTER / EIGHTH / SIXTEENTH
    syncopation: float = 0.0           # 0..1
    space_ratio: float = 1.0           # 0..1 fração da grade sem ataque
    energy: float = 0.0                # 0..1 relativo ao mais forte já ouvido na música
    energy_trend: float = 0.0          # -1..1 (log2 recente / anterior)
    percussive_density: float = 0.0    # 0..1 (banda aguda), se medida
    low_frequency_pulse: float = 0.0   # 0..1 (banda grave nos tempos), se medida
    straightness: float = 1.0          # 1 = reto
    swing_probability: float = 0.0
    half_time_probability: float = 0.0
    groove_stability: float = 0.0      # 0..1 semelhança entre compassos recentes
    confidence: float = 0.0
    label: str = "--"
    bars_observed: int = 0

    def to_dict(self) -> Dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}

    @property
    def density_level(self) -> str:
        d = self.onset_density
        return ("VERY_LOW" if d < 0.12 else "LOW" if d < 0.25 else
                "MEDIUM" if d < 0.45 else "HIGH")


@dataclass
class BarFeatures:
    """Medidas de UM compasso (antes da suavização)."""
    density: float
    pulse: float
    syncopation: float
    eighth: float
    sixteenth: float
    swing: float
    half_time: float
    energy: float
    percussive: float = -1.0
    low_pulse: float = -1.0
    grid: Optional[np.ndarray] = None


def bar_features(grid: np.ndarray, energy: float = 0.0,
                 high_grid: Optional[np.ndarray] = None,
                 low_grid: Optional[np.ndarray] = None) -> BarFeatures:
    """Medidas rítmicas de um compasso. ``grid``: (tempos, 4) força 0..1 por semicolcheia."""
    g = np.clip(np.asarray(grid, dtype=np.float64), 0.0, 1.0)
    if g.ndim == 1:
        g = g.reshape(-1, 4)
    beats = g.shape[0]
    active = g >= _ACTIVE
    density = float(active.mean()) if g.size else 0.0
    pulse = float(g[:, 0].mean()) if beats else 0.0
    eighth = float(g[:, 2].mean()) if beats else 0.0
    sixteenth = float(((g[:, 1] + g[:, 3]) / 2.0).mean()) if beats else 0.0
    # Síncope: ataque numa posição fraca que NÃO é seguido de ataque na posição forte
    # logo depois (o acento "salta" o tempo).
    sync = 0.0
    for b in range(beats):
        next_down = g[b + 1, 0] if b + 1 < beats else g[0, 0]
        sync += max(0.0, g[b, 2] - next_down)
        sync += 0.5 * max(0.0, g[b, 3] - next_down) + 0.5 * max(0.0, g[b, 1] - g[b, 2])
    syncopation = float(min(1.0, sync / max(1, beats) * 1.6))
    # Swing: a colcheia do contratempo chega atrasada (cai perto da 4ª semicolcheia).
    swing = float(np.clip((g[:, 3].mean() - g[:, 2].mean()) * 2.0, 0.0, 1.0)) if beats else 0.0
    half_time = 0.0
    if beats == 4:
        half_time = float(np.clip((g[2, 0] - (g[1, 0] + g[3, 0]) / 2.0) * 2.0, 0.0, 1.0) *
                          (1.0 - min(1.0, density * 2.0)))
    percussive = float((np.asarray(high_grid) >= _ACTIVE).mean()) if high_grid is not None else -1.0
    low_pulse = (float(np.clip(np.asarray(low_grid).reshape(-1, 4)[:, 0].mean(), 0.0, 1.0))
                 if low_grid is not None else -1.0)
    return BarFeatures(density, pulse, syncopation, eighth, sixteenth, swing, half_time,
                       float(max(0.0, energy)), percussive, low_pulse, g)


class GrooveFeelAnalyzer:
    """Perfil de groove suavizado, alimentado compasso a compasso (só o passado)."""

    SHORT_ALPHA = 0.40       # ≈ últimos 2–4 compassos
    LONG_ALPHA = 0.13        # ≈ últimos 8 compassos
    LABEL_PERSIST_BARS = 2   # compassos consecutivos para trocar o rótulo
    LABEL_MIN_HOLD_BARS = 3  # depois de trocar, segura o rótulo novo
    TREND_WINDOW = 6

    def __init__(self, beats_per_bar: int = 4, meter: str = "",
                 calibrated_energy: bool = False) -> None:
        # ``calibrated_energy``: a energia já vem calibrada (~1 = trecho forte típico,
        # pré-análise do arquivo). Ao vivo, sem calibração, usa o mais forte já ouvido.
        self.calibrated_energy = calibrated_energy
        self.beats_per_bar = max(1, int(beats_per_bar))
        self.meter = meter or ("6/8" if self.beats_per_bar == 6 else f"{self.beats_per_bar}/4")
        self.reset()

    def reset(self) -> None:
        self._short: Dict[str, float] = {}
        self._long: Dict[str, float] = {}
        self._energies: List[float] = []
        self._raw_energies: List[float] = []
        self._energy_ref = 0.0
        self._grids: List[np.ndarray] = []
        self._label = "--"
        self._pending_label = "--"
        self._pending_bars = 0
        self._bars_since_change = 0
        self._label_changes = 0
        self._profile = GrooveProfile(meter=self.meter)
        self._bpm = 0.0

    @property
    def profile(self) -> GrooveProfile:
        return self._profile

    @property
    def label_changes(self) -> int:
        return self._label_changes

    def observe_bar(self, grid: np.ndarray, energy: float = 0.0, bpm: float = 0.0,
                    high_grid: Optional[np.ndarray] = None,
                    low_grid: Optional[np.ndarray] = None) -> GrooveProfile:
        """Inclui um compasso que acabou de soar e devolve o perfil atualizado."""
        grid = np.asarray(grid, dtype=np.float64).reshape(-1, 4)
        peak = float(np.max(grid)) if grid.size else 0.0
        if bpm > 0:
            self._bpm = bpm if self._bpm <= 0 else 0.8 * self._bpm + 0.2 * bpm
        self._energy_ref = max(self._energy_ref * 0.995, float(energy))
        if self.calibrated_energy:
            rel_energy = float(np.clip(energy, 0.0, 1.0))
        else:
            rel_energy = float(energy) / self._energy_ref if self._energy_ref > 1e-9 else 0.0
        if peak <= 1e-9:
            # Compasso sem ataque (pausa, fermata): é espaço, não "groove novo".
            features = BarFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rel_energy,
                                   grid=np.zeros_like(grid))
        else:
            # A grade já vem normalizada pela música (≈1 = ataque forte típico):
            # normalizar por compasso faria um compasso fraco parecer cheio.
            features = bar_features(grid, rel_energy, high_grid, low_grid)
        self._update_smoothed(features)
        self._energies.append(rel_energy)
        self._raw_energies.append(max(0.0, float(energy)))
        self._raw_energies = self._raw_energies[-16:]
        self._grids.append(features.grid if features.grid is not None else np.zeros_like(grid))
        self._grids = self._grids[-8:]
        self._energies = self._energies[-16:]
        self._profile = self._build_profile()
        return self._profile

    # ------------------------------------------------------------ suavização
    def _update_smoothed(self, f: BarFeatures) -> None:
        values = {"density": f.density, "pulse": f.pulse, "syncopation": f.syncopation,
                  "eighth": f.eighth, "sixteenth": f.sixteenth, "swing": f.swing,
                  "half_time": f.half_time, "energy": f.energy}
        if f.percussive >= 0:
            values["percussive"] = f.percussive
        if f.low_pulse >= 0:
            values["low_pulse"] = f.low_pulse
        for name, value in values.items():
            for store, alpha in ((self._short, self.SHORT_ALPHA), (self._long, self.LONG_ALPHA)):
                store[name] = value if name not in store else store[name] + alpha * (value - store[name])

    def _trend(self) -> float:
        # Energia BRUTA: a razão recente/anterior não depende da calibração.
        values = self._raw_energies
        if len(values) < 3:
            return 0.0
        recent = float(np.mean(values[-2:]))
        before = values[-2 - self.TREND_WINDOW:-2]
        base = float(np.mean(before)) if before else recent
        if base <= 1e-6 or recent <= 1e-6:
            return 0.0
        return float(np.clip(math.log2(recent / base), -1.0, 1.0))

    def _stability(self) -> float:
        grids = self._grids
        if len(grids) < 2:
            return 0.0
        current = grids[-1].ravel()
        reference = np.mean([g.ravel() for g in grids[-4:-1]], axis=0)
        na, nb = np.linalg.norm(current), np.linalg.norm(reference)
        if na < 1e-9 and nb < 1e-9:
            return 1.0
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.clip(current @ reference / (na * nb), 0.0, 1.0))

    def _raw_label(self, s: Dict[str, float]) -> str:
        density, sync, energy = s["density"], s["syncopation"], s["energy"]
        if density < 0.16 and sync < 0.25:
            return SPACIOUS
        if sync >= 0.38 and density >= 0.16:
            return SYNCOPATED
        if density >= 0.55 or s["sixteenth"] >= 0.45:
            return BUSY
        if energy >= 0.65 and density >= 0.3:
            return DRIVING
        return STEADY

    def _build_profile(self) -> GrooveProfile:
        s, bars = self._short, len(self._energies)
        candidate = self._raw_label(s)
        self._bars_since_change += 1
        if self._label == "--":
            self._label = candidate
            self._bars_since_change = 0
        elif candidate != self._label:
            self._pending_bars = self._pending_bars + 1 if candidate == self._pending_label else 1
            self._pending_label = candidate
            if (self._pending_bars >= self.LABEL_PERSIST_BARS and
                    self._bars_since_change >= self.LABEL_MIN_HOLD_BARS):
                self._label = candidate
                self._label_changes += 1
                self._bars_since_change = 0
                self._pending_bars = 0
        else:
            self._pending_bars = 0
            self._pending_label = "--"
        eighth, sixteenth = s["eighth"], s["sixteenth"]
        subdivision = ("SIXTEENTH" if sixteenth >= 0.35 else
                       "EIGHTH" if eighth >= 0.3 or sixteenth >= 0.2 else "QUARTER")
        swing = s["swing"] if subdivision == "EIGHTH" and s["swing"] >= 0.35 else 0.0
        stability = self._stability()
        confidence = float(min(1.0, bars / 4.0) * (0.55 + 0.45 * stability))
        return GrooveProfile(
            tempo_bpm=round(self._bpm, 2), meter=self.meter,
            pulse_strength=s["pulse"], onset_density=s["density"], subdivision=subdivision,
            syncopation=s["syncopation"], space_ratio=1.0 - s["density"],
            energy=s["energy"], energy_trend=self._trend(),
            percussive_density=s.get("percussive", s["sixteenth"]),
            low_frequency_pulse=s.get("low_pulse", s["pulse"]),
            straightness=1.0 - swing, swing_probability=swing,
            half_time_probability=s["half_time"], groove_stability=stability,
            confidence=confidence, label=self._label, bars_observed=bars)


class LiveOnsetGrid:
    """Monta a grade (tempos × 4 semicolcheias) de cada compasso AO VIVO.

    Cada bloco de áudio traz um valor de fluxo espectral (ataque) e a fase do
    relógio musical; o valor vai para a semicolcheia correspondente (máximo).
    Quando o compasso vira, a grade fechada é devolvida para o analisador.
    """

    def __init__(self, beats_per_bar: int = 4):
        self.beats_per_bar = max(1, int(beats_per_bar))
        self._bar: Optional[int] = None
        self._grid = np.zeros((self.beats_per_bar, 4))
        self._energy: List[float] = []
        self._previous_spectrum: Optional[np.ndarray] = None
        self._flux_ref = 1e-9

    def reset(self) -> None:
        self._bar = None
        self._grid = np.zeros((self.beats_per_bar, 4))
        self._energy = []
        self._previous_spectrum = None
        self._flux_ref = 1e-9

    def flux(self, audio_chunk: np.ndarray) -> float:
        """Fluxo espectral positivo do bloco (barato: uma FFT de 2048 pontos)."""
        x = np.asarray(audio_chunk, dtype=np.float64)[-2048:]
        if x.size < 256:
            return 0.0
        spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
        spectrum = np.log1p(spectrum)
        previous = self._previous_spectrum
        self._previous_spectrum = spectrum
        if previous is None or previous.shape != spectrum.shape:
            return 0.0
        return float(np.maximum(0.0, spectrum - previous).sum())

    def observe(self, bar: int, beat: int, beat_position: float, flux: float,
                energy: float) -> Optional[tuple]:
        """Registra um bloco; devolve (grade, energia média) quando um compasso termina."""
        finished = None
        if self._bar is not None and bar != self._bar:
            if self._grid.max() > 0:
                grid = self._grid / self._flux_ref
                finished = (np.clip(grid, 0.0, 1.0), float(np.mean(self._energy)) if self._energy else 0.0)
            self._grid = np.zeros((self.beats_per_bar, 4))
            self._energy = []
        self._bar = bar
        b = min(self.beats_per_bar - 1, max(0, int(beat) - 1))
        slot = min(3, max(0, int(float(beat_position) * 4.0)))
        self._grid[b, slot] = max(self._grid[b, slot], flux)
        self._flux_ref = max(self._flux_ref * 0.999, flux)
        self._energy.append(float(energy))
        return finished
