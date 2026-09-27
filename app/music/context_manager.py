"""Gerenciador do Contexto Musical (MusicalContextManager) do Virtual Band AI.

Atua como ponte orquestradora entre os detectores brutos de DSP/MIR e o estado musical compartilhado:
  1. Aplica estabilização temporal em acordes e tonalidade.
  2. Alimenta o histórico cronológico de acordes e tonalidades.
  3. Sincroniza o MusicalClock com o tempo real do áudio.
  4. Consolida métricas quadripartidas de latência (processamento, janela, estabilização e estimada).
  5. Atualiza o MusicalContext thread-safe pronto para consumo pela UI e futuros músicos virtuais.
"""

from typing import Optional, Dict
import numpy as np

from app.music.musical_context import MusicalContext
from app.music.musical_clock import MusicalClock
from app.analysis.chord_history import ChordHistory, ChordStabilizer
from app.analysis.key_history import KeyHistory, KeyStabilizer
from app.analysis.pitch_detector import PitchResult
from app.analysis.chord_detector import Chord
from app.analysis.key_detector import KeyResult
from app.analysis.tempo_detector import TempoResult
from app.utils.timing import LatencyMetrics
from app.music.follow_confidence import calculate_follow_confidence


class MusicalContextManager:
    """Orquestrador do estado musical de alto nível e memória temporal."""

    def __init__(self, sample_rate: int = 44100, bpm: float = 120.0, meter: str = "4/4"):
        self._sample_rate = sample_rate

        # Contexto musical ativo
        self._context = MusicalContext(sample_rate=sample_rate, bpm=bpm, meter=meter)

        # Relógio rítmico musical
        self._clock = MusicalClock(bpm=bpm, meter=meter)

        # Estabilizador e Histórico de Acordes
        self._chord_history = ChordHistory()
        self._chord_stabilizer = ChordStabilizer(history=self._chord_history)

        # Estabilizador e Histórico de Tonalidade
        self._key_history = KeyHistory()
        self._key_stabilizer = KeyStabilizer(history=self._key_history)

    @property
    def context(self) -> MusicalContext:
        return self._context

    @property
    def clock(self) -> MusicalClock:
        return self._clock

    @property
    def chord_history(self) -> ChordHistory:
        return self._chord_history

    @property
    def chord_stabilizer(self) -> ChordStabilizer:
        return self._chord_stabilizer

    @property
    def key_history(self) -> KeyHistory:
        return self._key_history

    @property
    def key_stabilizer(self) -> KeyStabilizer:
        return self._key_stabilizer

    def reset(self) -> None:
        """Reinicia todos os acumuladores de memória e o relógio."""
        self._context.reset()
        self._clock.reset()
        self._chord_stabilizer.reset()
        self._key_stabilizer.reset()

    def set_meter(self, meter_str: str) -> None:
        """Configura nova fórmula de compasso."""
        self._clock.set_meter(meter_str)
        self._context.meter = self._clock.meter
        self._context.time_signature = self._clock.meter

    def update(
        self,
        timestamp: float,
        sample_rate: int,
        pitch_res: Optional[PitchResult] = None,
        chord_res: Optional[Chord] = None,
        key_res: Optional[KeyResult] = None,
        tempo_res: Optional[TempoResult] = None,
        chroma_vector: Optional[np.ndarray] = None,
        raw_chroma_vector: Optional[np.ndarray] = None,
        active_notes: Optional[Dict[str, float]] = None,
        lat_metrics: Optional[LatencyMetrics] = None,
        audio_activity: float = 0.0,
    ) -> MusicalContext:
        """Consolida as informações de todos os analisadores no MusicalContext."""
        ctx = self._context
        ctx.timestamp = timestamp
        ctx.sample_rate = sample_rate
        ctx.audio_activity = max(0.0, float(audio_activity))
        # Sem SongSession não há máquina de performance: o modo livre continua tocando.
        ctx.performance_state = "PLAYING"

        # 1. Atualizar Nota e Pitch Dominante
        if pitch_res is not None and pitch_res.is_voiced and pitch_res.confidence > 0.25:
            ctx.note = pitch_res.full_note
            ctx.frequency = pitch_res.frequency_hz
            ctx.note_confidence = pitch_res.confidence
            ctx.cents_deviation = pitch_res.cents_deviation
        else:
            ctx.note = "--"
            ctx.frequency = 0.0
            ctx.note_confidence = 0.0
            ctx.cents_deviation = 0.0

        ctx.raw_note = pitch_res.full_note if pitch_res and pitch_res.is_voiced else "--"
        ctx.raw_pitch_hz = pitch_res.frequency_hz if pitch_res and pitch_res.is_voiced else 0.0
        ctx.pitch_confidence = pitch_res.confidence if pitch_res else 0.0

        if chroma_vector is not None:
            ctx.chroma_vector = chroma_vector.tolist() if isinstance(chroma_vector, np.ndarray) else list(chroma_vector)
        ctx.raw_chroma_vector = (raw_chroma_vector.tolist()
                                 if isinstance(raw_chroma_vector, np.ndarray)
                                 else list(raw_chroma_vector) if raw_chroma_vector is not None else [0.0] * 12)
        ctx.active_notes = dict(active_notes or {})

        # 2. Estabilização e Histórico de Acordes
        if chord_res is not None:
            stable_chord = self._chord_stabilizer.process(
                raw_symbol=chord_res.symbol,
                confidence=chord_res.confidence,
                timestamp=timestamp,
                root=chord_res.root,
                quality=chord_res.quality,
                bass_note=chord_res.bass_note,
                inversion=chord_res.inversion,
                detected_notes=chord_res.detected_notes,
                chroma_vector=chroma_vector,
                audio_activity=ctx.audio_activity,
            )
            ctx.chord = stable_chord
            ctx.raw_detected_chord = chord_res.symbol
            ctx.raw_chord_confidence = chord_res.confidence
            ctx.smoothed_detected_chord = stable_chord
            ctx.stable_chord_confidence = self._chord_stabilizer.chord_confidence
            ctx.stable_chord_duration = self._chord_stabilizer.chord_duration
            ctx.stable_chord_root = self._chord_stabilizer.chord_root
            ctx.stable_chord_quality = self._chord_stabilizer.chord_quality
            ctx.chord_candidate = self._chord_stabilizer.candidate_chord
            ctx.chord_candidate_root = self._chord_stabilizer.candidate_root
            ctx.chord_candidate_confidence = self._chord_stabilizer.candidate_confidence
            ctx.chord_candidate_age_ms = self._chord_stabilizer.candidate_duration * 1000.0
            ctx.chord_candidate_frames = self._chord_stabilizer.candidate_frame_count
            ctx.current_chord_support_age_ms = (
                self._chord_stabilizer.time_since_current_chord_support * 1000.0)
            ctx.stable_chord_stale = self._chord_stabilizer.is_current_chord_stale
            ctx.previous_chord = self._chord_stabilizer.previous_chord
            ctx.chord_start_time = self._chord_stabilizer.chord_start_time
            ctx.chord_duration = self._chord_stabilizer.chord_duration
            ctx.chord_confidence = self._chord_stabilizer.chord_confidence
            ctx.chord_root = self._chord_stabilizer.chord_root
            ctx.chord_quality = self._chord_stabilizer.chord_quality
            ctx.bass_note = self._chord_stabilizer.bass_note
            ctx.inversion = self._chord_stabilizer.inversion
            ctx.detected_notes = self._chord_stabilizer.detected_notes
        else:
            ctx.chord = "--"
            ctx.previous_chord = "--"
            ctx.chord_duration = 0.0
            ctx.chord_confidence = 0.0
            ctx.raw_detected_chord = "--"
            ctx.raw_chord_confidence = 0.0
            ctx.smoothed_detected_chord = "--"
            ctx.stable_chord_confidence = 0.0
            ctx.stable_chord_duration = 0.0
            ctx.stable_chord_root = "--"
            ctx.stable_chord_quality = "--"
            ctx.chord_candidate = "--"
            ctx.chord_candidate_root = "--"
            ctx.chord_candidate_confidence = 0.0
            ctx.chord_candidate_age_ms = 0.0
            ctx.chord_candidate_frames = 0
            ctx.current_chord_support_age_ms = 0.0
            ctx.stable_chord_stale = False

        # 3. Estabilização e Histórico de Tonalidade
        if key_res is not None:
            recent_chords = self._chord_history.get_recent(8)
            all_key_scores = getattr(key_res, "all_key_scores", None)
            stable_key = self._key_stabilizer.process(
                raw_key=key_res.key_name,
                confidence=key_res.confidence,
                timestamp=timestamp,
                root=key_res.root,
                scale_type=key_res.scale_type,
                recent_chords=recent_chords,
                all_key_scores=all_key_scores
            )
            ctx.key = stable_key
            ctx.previous_key = self._key_stabilizer.previous_key
            ctx.key_start_time = self._key_stabilizer.key_start_time
            ctx.key_duration = self._key_stabilizer.key_duration
            ctx.key_confidence = self._key_stabilizer.key_confidence
            ctx.key_candidate = self._key_stabilizer.candidate_key
            ctx.candidate_confidence = self._key_stabilizer.candidate_confidence
            ctx.candidate_duration = self._key_stabilizer.candidate_duration
            ctx.local_tonal_center = self._key_stabilizer.local_tonal_center
        else:
            ctx.key = "--"
            ctx.previous_key = "--"
            ctx.key_duration = 0.0
            ctx.key_confidence = 0.0
            ctx.key_candidate = "--"
            ctx.candidate_confidence = 0.0
            ctx.candidate_duration = 0.0
            ctx.local_tonal_center = "--"


        # 4. Relógio Musical (MusicalClock) e Posição Rítmica
        bpm_val = tempo_res.bpm if (tempo_res and tempo_res.bpm > 0
                                    and self._clock.tracking_state != "TRACKING") else None
        pulse = tempo_res.beat_timestamp if tempo_res else None
        self._clock.update(timestamp=timestamp, bpm=bpm_val,
                           beat_timestamp=pulse,
                           observation_confidence=tempo_res.confidence if tempo_res else 0.0)

        ctx.bpm = self._clock.bpm
        ctx.meter = self._clock.meter
        # Sem cifra, a posição musical coincide com a referência temporal.
        ctx.clock_bar = self._clock.bar
        ctx.clock_beat = self._clock.beat
        ctx.bar_offset = 0
        ctx.bar = ctx.clock_bar
        ctx.beat = ctx.clock_beat
        ctx.beat_position = self._clock.beat_position
        ctx.is_beat = self._clock.is_beat
        ctx.initial_bpm = self._clock.initial_bpm
        ctx.target_bpm = self._clock.target_bpm
        ctx.tempo_confidence = self._clock.tempo_confidence
        ctx.phase_confidence = self._clock.phase_confidence
        ctx.phase_error_ms = self._clock.phase_error_ms
        ctx.tempo_tracking_state = self._clock.tracking_state

        # 5. Métricas de Latência Quadripartida
        stab_delay = self._chord_stabilizer.stabilization_delay_ms
        ctx.stabilization_delay = stab_delay

        if lat_metrics is not None:
            ctx.processing_latency = lat_metrics.processing_ms
            ctx.analysis_window = lat_metrics.analysis_window_ms
            ctx.capture_latency = lat_metrics.capture_ms
            ctx.analysis_latency = lat_metrics.analysis_ms
            ctx.output_latency = lat_metrics.output_ms
        else:
            ctx.processing_latency = 0.0
            ctx.analysis_window = (4096 / float(sample_rate)) * 1000.0 if sample_rate > 0 else 92.8
            ctx.capture_latency = 0.0
            ctx.analysis_latency = ctx.analysis_window * 0.5
            ctx.output_latency = 0.0
        ctx.refresh_total_latency()

        # 6. Confiança de acompanhamento. Sem cifra, ``chart_alignment`` permanece
        # desconhecida e impede que a banda assuma que pode antecipar uma mudança.
        follow = calculate_follow_confidence(
            tempo=ctx.tempo_confidence, phase=ctx.phase_confidence,
            position=ctx.position_confidence, harmonic=ctx.chord_confidence,
            chart_alignment=ctx.chart_alignment_confidence, stability=ctx.recent_stability)
        ctx.confidence = follow.score
        ctx.follow_confidence_level = follow.level

        # 7. Sincronizar todos os aliases para garantir total retrocompatibilidade
        ctx.sync_aliases()

        return ctx
