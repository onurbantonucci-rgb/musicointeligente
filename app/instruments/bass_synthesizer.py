"""Sintetizador Local de Baixo Elétrico (BassSynthesizer).

Responsável por sintetizar som de contrabaixo elétrico puro e quente em tempo real:
- Síntese acústica aditiva com harmônicos calibrados (fundamental, 2º e 3º harmônicos).
- Pluck transient (ataque percussivo característico da corda dedilhada ou palhetada).
- Envelope ADSR suave com decaimento natural de instrumento de corda.
- Gerenciamento de vozes polifônicas/monofônicas thread-safe.
- Renderização em blocos para mixagem contínua no AudioPlayer com zero latência.
"""

import threading
from collections import deque
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from app.music.theory import midi_to_hz
from app.instruments.bass_decision import midi_to_note_name
from app.music.constants import BASS_DEFAULT_VOLUME


@dataclass(frozen=True)
class ScheduledBassNote:
    scheduled_beat: tuple
    scheduled_time: float
    midi_note: int
    velocity: int
    duration: float
    source: str
    confidence: float
    generation_id: int
    note: str = "--"
    musical_time: float = 0.0          # Horário do beat; pode diferir do horário de execução compensado


class ActiveVoice:
    """Representa uma nota em execução no sintetizador."""

    def __init__(self, freq: float, velocity: int, duration_sec: float, sample_rate: int):
        self.freq = freq
        self.velocity = float(velocity) / 127.0
        self.duration_samples = int(duration_sec * sample_rate)
        self.release_samples = int(0.035 * sample_rate)  # 35 ms de release suave
        self.total_samples = self.duration_samples + self.release_samples
        self.current_sample = 0
        self.is_finished = False

        # Pré-cálculo do envelope temporal ADSR
        t = np.arange(self.total_samples, dtype=np.float32) / float(sample_rate)

        # 1. Ataque rápido (10 ms)
        att_len = max(1, int(0.010 * sample_rate))
        attack = np.ones(self.total_samples, dtype=np.float32)
        if att_len < self.total_samples:
            attack[:att_len] = np.linspace(0.0, 1.0, att_len, dtype=np.float32)

        # 2. Decaimento natural de corda com sustain
        decay = np.exp(-t / 0.40) * 0.70 + 0.30

        # 3. Release suave ao final da duração
        release = np.ones(self.total_samples, dtype=np.float32)
        if self.duration_samples < self.total_samples:
            rel_len = self.total_samples - self.duration_samples
            release[self.duration_samples:] = np.linspace(1.0, 0.0, rel_len, dtype=np.float32)

        envelope = attack * decay * release

        # Harmônicos calibrados de contrabaixo elétrico:
        # Fundamental (1.0), 2º harmônico quente (0.45), 3º harmônico com punch (0.18), 4º harmônico (0.06)
        w1 = 2.0 * np.pi * self.freq
        wave = (
            1.00 * np.sin(w1 * t) +
            0.45 * np.sin(2.0 * w1 * t) +
            0.18 * np.sin(3.0 * w1 * t) +
            0.06 * np.sin(4.0 * w1 * t)
        )

        # Transiente de palheta/dedilhado nos primeiros 12 ms. Ele é um ruído
        # curto, determinístico e sem relação harmônica com a fundamental: o
        # antigo seno em 5*f podia colorir D como F# no ataque.
        pluck_len = min(self.total_samples, int(0.012 * sample_rate))
        if pluck_len > 0:
            index = np.arange(pluck_len, dtype=np.float32)
            noise = np.sin((index + 1.0) * 12.9898) * 43758.5453
            noise = (noise - np.floor(noise)) * 2.0 - 1.0
            # Diferenciação remove a componente grave e preserva o caráter de
            # ataque sem sugerir nenhuma terça específica.
            noise[1:] -= noise[:-1]
            peak = max(float(np.max(np.abs(noise))), 1e-6)
            pluck = (noise / peak) * np.exp(-t[:pluck_len] / 0.003) * 0.12
            wave[:pluck_len] += pluck

        # Sinal completo da nota normalizado e amplificado pela dinâmica (velocity)
        raw_signal = wave * envelope * (self.velocity * 0.85)
        self.signal = raw_signal.astype(np.float32)

    def render(self, frames: int) -> np.ndarray:
        """Extrai a fatia do sinal para o bloco atual."""
        if self.is_finished or self.current_sample >= self.total_samples:
            self.is_finished = True
            return np.zeros(frames, dtype=np.float32)

        rem = self.total_samples - self.current_sample
        take = min(frames, rem)

        out = np.zeros(frames, dtype=np.float32)
        out[:take] = self.signal[self.current_sample:self.current_sample + take]

        self.current_sample += take
        if self.current_sample >= self.total_samples:
            self.is_finished = True

        return out


class BassSynthesizer:
    """Sintetizador de Baixo Elétrico com mixagem em tempo real e controle de dinâmica."""

    def __init__(self, sample_rate: int = 44100, volume: float = BASS_DEFAULT_VOLUME):
        self._sample_rate = sample_rate
        self._volume = volume
        self._enabled = True
        self.strict_mono = False
        self._lock = threading.Lock()
        self._voices: List[ActiveVoice] = []
        self._scheduled = {}  # chave (compasso, beat) -> (timestamp, midi, velocidade, duração)
        self._generation_id = 0
        self._rendered_beats = deque(maxlen=64)
        self._playing_event: Optional[ScheduledBassNote] = None
        self._last_render_delay_ms = 0.0
        self._rendered_until_time: Optional[float] = None

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, val: float) -> None:
        self._volume = max(0.0, min(1.0, float(val)))

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = bool(val)

    def set_sample_rate(self, sr: int) -> None:
        with self._lock:
            self._sample_rate = sr
            self._voices.clear()
            self._scheduled.clear()
            self._playing_event = None
            self._rendered_until_time = None

    def clear(self) -> None:
        """Interrompe todas as vozes ativas."""
        with self._lock:
            self._voices.clear()
            self._scheduled.clear()
            self._generation_id = 0
            self._rendered_beats.clear()
            self._playing_event = None
            self._last_render_delay_ms = 0.0
            self._rendered_until_time = None

    @property
    def playing_event(self) -> Optional[ScheduledBassNote]:
        """Último ataque que realmente entrou no buffer de áudio e ainda soa."""
        with self._lock:
            return self._playing_event if self._voices else None

    @property
    def last_render_delay_ms(self) -> float:
        with self._lock:
            return self._last_render_delay_ms

    @property
    def rendered_until_time(self) -> Optional[float]:
        """Fim do último buffer que já foi entregue à saída de áudio."""
        with self._lock:
            return self._rendered_until_time

    @property
    def scheduled_events(self) -> List[ScheduledBassNote]:
        with self._lock:
            return sorted(self._scheduled.values(), key=lambda event: event.scheduled_time)

    def was_rendered(self, beat_key: tuple) -> bool:
        """Distingue uma previsão cancelada de uma batida que já soou."""
        with self._lock:
            return beat_key in self._rendered_beats

    def set_generation(self, generation_id: int) -> None:
        """Invalida a projeção antiga sem cortar notas que já estão soando."""
        with self._lock:
            if generation_id > self._generation_id:
                self._scheduled.clear()
                self._rendered_beats.clear()
                self._generation_id = generation_id

    def cancel_scheduled(self, key: Optional[tuple] = None) -> None:
        """Cancela ataques futuros, ou só o pulso indicado, sem cortar a voz atual."""
        with self._lock:
            if key is None:
                self._scheduled.clear()
            else:
                self._scheduled.pop(key, None)

    def stop_voices(self) -> None:
        """Encerra vozes ativas ao trocar para um padrão de uma nota por vez."""
        with self._lock:
            self._voices.clear()
            self._playing_event = None

    def schedule_note(self, key: tuple, start_time: float, midi_note: int,
                      velocity: int, duration: float, source: str = "chart",
                      confidence: float = 1.0, generation_id: int = 0,
                      note: str = "--", musical_time: Optional[float] = None) -> None:
        """Arma uma nota para a linha de áudio; atualizações substituem a previsão anterior."""
        if not self._enabled or midi_note <= 0:
            return
        with self._lock:
            if generation_id != self._generation_id or key in self._rendered_beats:
                return
            self._scheduled[key] = ScheduledBassNote(
                key, start_time, midi_note, velocity, duration,
                source, confidence, generation_id, note,
                start_time if musical_time is None else musical_time)
            if len(self._scheduled) > 8:
                oldest = min(self._scheduled, key=lambda item: self._scheduled[item].scheduled_time)
                del self._scheduled[oldest]

    def trigger_note(self, midi_note: int, velocity: int = 100, duration: float = 0.5) -> None:
        """Dispara uma nova nota de baixo imediatamente."""
        if not self._enabled or midi_note <= 0:
            return

        freq = midi_to_hz(midi_note)
        if freq < 20.0 or freq > 1000.0:
            return

        voice = ActiveVoice(
            freq=freq,
            velocity=velocity,
            duration_sec=duration,
            sample_rate=self._sample_rate
        )

        with self._lock:
            # No modo Fundamentais, nenhum ataque compartilha a voz anterior.
            if self.strict_mono:
                self._voices.clear()
            elif len(self._voices) >= 2:
                self._voices = self._voices[-1:]
            self._voices.append(voice)
            self._playing_event = ScheduledBassNote(
                (), 0.0, midi_note, velocity, duration, "immediate", 1.0,
                self._generation_id, midi_to_note_name(midi_note))
            self._last_render_delay_ms = 0.0

    def render_chunk(self, frames: int, sample_rate: int, current_pos: float = 0.0) -> np.ndarray:
        """Gera um bloco de áudio estéreo (frames, 2) pronto para mixagem no AudioPlayer."""
        if not self._enabled or self._volume <= 0.0:
            return np.zeros((frames, 2), dtype=np.float32)

        if sample_rate != self._sample_rate:
            self.set_sample_rate(sample_rate)

        mono_mix = np.zeros(frames, dtype=np.float32)

        with self._lock:
            if self.strict_mono:
                end_pos = current_pos + frames / sample_rate
                due = sorted(((key, item) for key, item in self._scheduled.items()
                              if item.scheduled_time < end_pos),
                             key=lambda pair: pair[1].scheduled_time)
                voices = self._voices
                cursor = 0
                for key, event in due:
                    del self._scheduled[key]
                    if event.scheduled_time < current_pos - .015:
                        continue
                    offset = max(0, round((event.scheduled_time - current_pos) * sample_rate))
                    if offset >= frames:
                        continue
                    if offset > cursor:
                        segment = np.zeros(offset - cursor, dtype=np.float32)
                        for voice in voices:
                            segment += voice.render(offset - cursor)
                        # Saída curta da voz antiga evita clique no corte.
                        fade = min(len(segment), max(1, int(.004 * sample_rate)))
                        segment[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
                        mono_mix[cursor:offset] += segment
                    voices = [ActiveVoice(midi_to_hz(event.midi_note), event.velocity,
                                          event.duration, sample_rate)]
                    self._rendered_beats.append(key)
                    self._playing_event = event
                    self._last_render_delay_ms = max(
                        0.0, (current_pos - event.scheduled_time) * 1000.0)
                    cursor = offset
                if cursor < frames:
                    for voice in voices:
                        mono_mix[cursor:] += voice.render(frames - cursor)
                self._voices = [voice for voice in voices if not voice.is_finished]
                if not self._voices:
                    self._playing_event = None
            else:
                active_voices = []
                for voice in self._voices:
                    mono_mix += voice.render(frames)
                    if not voice.is_finished:
                        active_voices.append(voice)
                if current_pos > 0 or self._scheduled:
                    end_pos = current_pos + frames / sample_rate
                    due = sorted(((key, item) for key, item in self._scheduled.items()
                                  if item.scheduled_time < end_pos),
                                 key=lambda pair: pair[1].scheduled_time)
                    for key, event in due:
                        del self._scheduled[key]
                        start, midi, velocity, duration = (event.scheduled_time, event.midi_note,
                                                           event.velocity, event.duration)
                        # Evento vencido não entra no bloco e nunca é disparado em rajada.
                        if start < current_pos - 0.015:
                            continue
                        offset = max(0, round((start - current_pos) * sample_rate))
                        if offset >= frames:
                            continue
                        voice = ActiveVoice(midi_to_hz(midi), velocity, duration, sample_rate)
                        self._rendered_beats.append(key)
                        mono_mix[offset:] += voice.render(frames - offset)
                        self._playing_event = event
                        self._last_render_delay_ms = max(
                            0.0, (current_pos - event.scheduled_time) * 1000.0)
                        if not voice.is_finished:
                            active_voices.append(voice)
                self._voices = active_voices
                if not self._voices:
                    self._playing_event = None
            self._rendered_until_time = current_pos + frames / sample_rate

        # Aplica volume master do baixo com proteção estrita contra saturação/clipping
        mono_mix = np.clip(mono_mix * self._volume, -1.0, 1.0)

        # Retorna estéreo (L e R idênticos para baixo centrado)
        return np.column_stack([mono_mix, mono_mix]).astype(np.float32)

    def synthesize_note_to_array(
        self,
        midi_note: int,
        duration: float,
        velocity: int = 100,
        sr: int = 44100
    ) -> np.ndarray:
        """Método utilitário para renderizar uma nota completa em array NumPy (ideal para testes)."""
        freq = midi_to_hz(midi_note)
        voice = ActiveVoice(freq, velocity, duration, sr)
        return voice.signal
