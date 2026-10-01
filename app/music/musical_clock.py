"""Relógio Musical (MusicalClock) sincronizado com a posição real do áudio.

Converte tempo contínuo (segundos) e andamento (BPM) em posições musicais discretas:
compasso (bar), tempo (beat), fase do tempo (beat_position) e fórmula de compasso (meter).
"""

from typing import Tuple, Optional
import math
from app.music.constants import (
    DEFAULT_METER,
    DEFAULT_BEATS_PER_BAR,
    DEFAULT_BEAT_UNIT,
    DEFAULT_BPM,
    BEAT_PULSE_TOLERANCE_SEC,
)


class MusicalClock:
    """Relógio rítmico que calcula a métrica musical a partir da posição temporal real do áudio.
    
    Projetado para suportar qualquer fórmula de compasso (4/4, 3/4, 2/4, 6/8, etc.),
    e manter sincronismo instantâneo com seek, play, pause e stop.
    """

    def __init__(self, bpm: float = DEFAULT_BPM, meter: str = DEFAULT_METER):
        self._bpm = max(0.0, float(bpm))
        self._initial_bpm = self._bpm
        self._target_bpm = self._bpm
        self._total_beats = 0.0
        self._last_observation: Optional[float] = None
        self._pending_tempo: Optional[float] = None
        self._tempo_confidence = 0.0
        self._phase_confidence = 0.0
        self._phase_error_ms = 0.0
        self._tracking_state = "HOLDOVER"
        self._meter = meter
        self._beats_per_bar, self._beat_unit = self._parse_meter(meter)
        # Em que batida do áudio cai o TEMPO 1 da música (em batidas desde t=0).
        # Sem isso, os compassos eram numerados a partir de t=0: uma introdução
        # com silêncio ou contagem deslocava todas as barras de compasso.
        self._downbeat_offset: float = 0.0

        # Estado rítmico instantâneo
        self._elapsed_time: float = 0.0
        self._bar: int = 1
        self._beat: int = 1
        self._beat_position: float = 0.0       # [0.0 a 1.0) dentro do tempo atual
        self._bar_position: float = 0.0        # [0.0 a 1.0) dentro do compasso atual
        self._is_beat: bool = False

    @staticmethod
    def _parse_meter(meter_str: str) -> Tuple[int, int]:
        """Extrai numerador (tempos por compasso) e denominador (unidade de tempo) da fórmula."""
        try:
            parts = meter_str.strip().split('/')
            if len(parts) == 2:
                num = int(parts[0])
                den = int(parts[1])
                if num > 0 and den > 0:
                    return num, den
        except Exception:
            pass
        return DEFAULT_BEATS_PER_BAR, DEFAULT_BEAT_UNIT

    @property
    def bpm(self) -> float:
        return self._bpm

    @bpm.setter
    def bpm(self, value: float) -> None:
        self._bpm = max(0.0, float(value))
        self._target_bpm = self._bpm

    @property
    def initial_bpm(self) -> float:
        return self._initial_bpm

    @property
    def current_bpm(self) -> float:
        return self._bpm

    @property
    def target_bpm(self) -> float:
        return self._target_bpm

    @property
    def beat_phase(self) -> float:
        return self._beat_position

    @property
    def total_beats(self) -> float:
        """Coordenada musical contínua usada para duração harmônica, nunca em segundos."""
        return self._total_beats

    @property
    def downbeat_offset(self) -> float:
        """Batida (desde t=0) que é o tempo 1 do compasso 1 da música."""
        return self._downbeat_offset

    def beats_at(self, timestamp: float) -> float:
        """Coordenada em batidas (desde t=0) de um instante próximo ao atual."""
        return self._total_beats - (self._elapsed_time - float(timestamp)) / self.beat_duration

    def align_downbeat_at(self, timestamp: float, snap_phase: bool = False) -> None:
        """Faz da batida mais próxima de ``timestamp`` o tempo 1 do compasso 1.

        Sem ``snap_phase`` só a numeração de compassos/tempos muda. Com
        ``snap_phase`` (quando ``timestamp`` é uma batida REAL observada no áudio),
        a fase do relógio também é fixada nela: se a música entra no meio de uma
        batida da grade que começou em t=0 (ex.: 3,5 batidas de silêncio), o
        arredondamento sozinho escolheria a batida vizinha e deslocaria tudo.
        """
        beats = self.beats_at(timestamp)
        if snap_phase:
            nearest = math.floor(beats + 0.5)
            self._total_beats += nearest - beats
            beats = float(nearest)
            self._phase_error_ms = 0.0
            # A batida ancorada vale como pulso observado. Sem isso, enquanto nenhum
            # pulso tivesse sido visto, update() recalcularia total_beats a partir do
            # tempo decorrido e apagaria este ajuste de fase no tick seguinte.
            self._last_observation = float(timestamp)
            self._tracking_state = "TRACKING"
            self._phase_confidence = max(self._phase_confidence, 0.8)
        self._downbeat_offset = float(math.floor(beats + 0.5))
        self.update(self._elapsed_time)

    @property
    def tempo_confidence(self) -> float:
        return self._tempo_confidence

    @property
    def phase_confidence(self) -> float:
        return self._phase_confidence

    @property
    def phase_error_ms(self) -> float:
        return self._phase_error_ms

    @property
    def tracking_state(self) -> str:
        return self._tracking_state

    @property
    def meter(self) -> str:
        return self._meter

    @property
    def beats_per_bar(self) -> int:
        return self._beats_per_bar

    @property
    def beat_unit(self) -> int:
        return self._beat_unit

    @property
    def bar(self) -> int:
        """Número do compasso atual (1-indexado: 1, 2, 3...)."""
        return self._bar

    @property
    def beat(self) -> int:
        """Número do tempo no compasso atual (1-indexado: 1 a beats_per_bar)."""
        return self._beat

    @property
    def beat_position(self) -> float:
        """Posição fracionária dentro do tempo atual [0.0 a 1.0)."""
        return self._beat_position

    @property
    def bar_position(self) -> float:
        """Posição fracionária dentro do compasso atual [0.0 a 1.0)."""
        return self._bar_position

    @property
    def elapsed_time(self) -> float:
        """Tempo decorrido no áudio em segundos."""
        return self._elapsed_time

    @property
    def is_beat(self) -> bool:
        """Indica se o instante atual coincide com o clique/pulso da batida."""
        return self._is_beat

    @property
    def beat_duration(self) -> float:
        """Duração de um tempo rítmico em segundos."""
        if self._bpm <= 0:
            return 0.5  # Padrão seguro para 120 BPM
        # Para semínima (unidade 4), 60 / BPM
        # Se for colcheia (unidade 8), tempo = (60 / BPM) * (4 / 8)
        return (60.0 / self._bpm) * (4.0 / self._beat_unit)

    @property
    def bar_duration(self) -> float:
        """Duração total de um compasso completo em segundos."""
        return self.beat_duration * self._beats_per_bar

    def set_meter(self, meter_str: str) -> None:
        """Configura a fórmula de compasso (ex: '4/4', '3/4', '6/8')."""
        self._beats_per_bar, self._beat_unit = self._parse_meter(meter_str)
        self._meter = f"{self._beats_per_bar}/{self._beat_unit}"
        self.update(self._elapsed_time)

    def reset(self) -> None:
        """Reinicia o relógio musical para o início (t=0.0)."""
        self._elapsed_time = 0.0
        self._bar = 1
        self._beat = 1
        self._beat_position = 0.0
        self._bar_position = 0.0
        self._is_beat = False
        self._total_beats = 0.0
        self._last_observation = None
        self._pending_tempo = None
        self._target_bpm = self._bpm
        self._tempo_confidence = 0.0
        self._phase_confidence = 0.0
        self._phase_error_ms = 0.0
        self._tracking_state = "HOLDOVER"
        self._downbeat_offset = 0.0

    def seek(self, timestamp: float) -> None:
        """Move o relógio diretamente para o timestamp especificado."""
        self.update(timestamp)

    def tick(self, dt: float) -> None:
        """Avança o relógio musical em dt segundos."""
        self.update(self._elapsed_time + dt)

    def update(self, timestamp: float, bpm: float = None, external_is_beat: bool = None,
               beat_timestamp: Optional[float] = None, observation_confidence: float = 0.0) -> None:
        """Atualiza a posição musical com base no timestamp real do áudio.
        
        Garante sincronia instantânea em caso de seek, pause, reprodução contínua ou stop.
        """
        previous_time = self._elapsed_time
        self._elapsed_time = max(0.0, float(timestamp))
        if bpm is not None and bpm > 0:
            self.bpm = bpm
        dt = self._elapsed_time - previous_time
        if dt < -0.01:
            # Seek da fonte: descarta observações que pertenciam ao trecho anterior,
            # mas preserva onde cai o tempo 1 (é propriedade da música, não da leitura).
            downbeat_offset = self._downbeat_offset
            self.reset()
            self._downbeat_offset = downbeat_offset
            self._elapsed_time = max(0.0, float(timestamp))
            dt = self._elapsed_time
        if self._last_observation is None:
            self._total_beats = self._elapsed_time / self.beat_duration
        else:
            if dt > 0:
                # Filtro de primeira ordem: ajusta andamento sem reinterpretar o passado.
                alpha = 1.0 - math.exp(-dt / 1.25)
                old_bpm = self._bpm
                self._bpm += (self._target_bpm - self._bpm) * alpha
                self._total_beats += dt / self.beat_duration if old_bpm == self._bpm else (
                    dt * (old_bpm + self._bpm) / (2.0 * 60.0) * (4.0 / self._beat_unit))
            if self._elapsed_time - self._last_observation > 2.5:
                self._tracking_state = "HOLDOVER"
                self._phase_confidence = max(0.0, self._phase_confidence - max(0.0, dt) * 0.1)
                self._tempo_confidence = max(0.0, self._tempo_confidence - max(0.0, dt) * 0.04)
        if beat_timestamp is not None and observation_confidence >= 0.6:
            self.observe_pulse(beat_timestamp, observation_confidence)

        if self._bpm <= 0.0:
            self._bar = 1
            self._beat = 1
            self._beat_position = 0.0
            self._bar_position = 0.0
            self._is_beat = False
            return

        b_dur = self.beat_duration
        measure_dur = self.bar_duration

        if b_dur <= 0 or measure_dur <= 0:
            return

        # Total de batidas decorridas desde o início (com precisão de ponto flutuante)
        # Adicionar epsilon para evitar imprecisão numérica em múltiplos exatos (ex: 0.500000000001)
        eps = 1e-6
        total_beats = self._total_beats - self._downbeat_offset + eps

        # Compasso (1-indexado)
        self._bar = int(total_beats // self._beats_per_bar) + 1

        # Tempo no compasso (1-indexado de 1 a beats_per_bar)
        self._beat = int(total_beats % self._beats_per_bar) + 1

        # Posição fracionária dentro do tempo [0.0 a 1.0)
        self._beat_position = float(total_beats % 1.0)

        # Posição fracionária dentro do compasso [0.0 a 1.0)
        self._bar_position = float((total_beats % self._beats_per_bar) / self._beats_per_bar)

        # Verificação do pulso de batida
        if external_is_beat is not None:
            self._is_beat = external_is_beat
        else:
            time_into_beat = self._beat_position * b_dur
            # Pulso no início da batida (janela de tolerância)
            self._is_beat = (time_into_beat <= BEAT_PULSE_TOLERANCE_SEC or
                             time_into_beat >= (b_dur - BEAT_PULSE_TOLERANCE_SEC))

    def observe_pulse(self, timestamp: float, confidence: float = 1.0) -> bool:
        """PLL discreto: aceita pulso plausível, filtra BPM e corrige fase aos poucos."""
        if confidence < 0.6 or self._bpm <= 0 or timestamp > self._elapsed_time + 0.1:
            return False
        if self._last_observation is None:
            # Um ataque isolado só arma a hipótese; não altera a fase nem o BPM.
            self._last_observation = timestamp
            self._tracking_state = "UNCERTAIN"
            return False
        if self._last_observation is not None:
            interval = timestamp - self._last_observation
            if interval <= 0.15:
                return False
            expected = self.beat_duration
            if interval > expected * 4.5:
                # Retorno após perda: o intervalo inteiro não é um único beat.
                self._pending_tempo = None
            else:
                count = max(1, round(interval / expected))
                proposed = (60.0 / (interval / count)) * (4.0 / self._beat_unit)
                if count > 4 or abs(proposed / self._bpm - 1.0) > 0.23:
                    if self._tempo_confidence == 0:
                        self._last_observation = timestamp
                    return False
                if abs(proposed / self._bpm - 1.0) > 0.08:
                    if self._pending_tempo is None or abs(proposed / self._pending_tempo - 1.0) > 0.05:
                        self._pending_tempo = proposed
                        return False
                self._pending_tempo = None
                self._target_bpm += (proposed - self._target_bpm) * min(0.5, 0.32 * confidence)
        observed_beats = self._total_beats - (self._elapsed_time - timestamp) / self.beat_duration
        phase_error = observed_beats - round(observed_beats)
        # Qualquer fase circular é corrigida com ganho limitado, sem salto de beat.
        self._phase_error_ms = phase_error * self.beat_duration * 1000.0
        self._total_beats -= phase_error * min(0.45, 0.30 * confidence)
        self._last_observation = timestamp
        self._tempo_confidence = min(1.0, max(self._tempo_confidence, confidence * 0.8))
        self._phase_confidence = min(1.0, max(self._phase_confidence, confidence * (1.0 - abs(phase_error))))
        self._tracking_state = "TRACKING"
        return True
