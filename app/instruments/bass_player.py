"""Baixista Virtual Autônomo (BassPlayer v0.2).

Orquestrador central do baixista virtual:
1. Consome o MusicalContext (acorde estabilizado, tom, BPM, beat, compasso).
2. Coordena BassDecisionEngine (decisão harmônica de notas).
3. Coordena BassPatternGenerator (estruturação do padrão rítmico).
4. Coordena BassPerformanceEngine (sincronização de timing e articulação).
5. Dispara notas em tempo real no BassSynthesizer para mixagem com o áudio do instrumento real.
6. Mantém histórico de decisões para inspeção em tempo real e modo debug na interface gráfica.
"""

from typing import Optional, List, Dict, Tuple
import collections
import copy
import math
import time

from app.instruments.base import VirtualInstrument
from app.instruments.bass_model import BassNoteEvent, BassDecision, BassPatternType, BassNoteValue, BassHarmonySource
from app.instruments.bass_decision import BassDecisionEngine
from app.instruments.bass_pattern import BassPatternGenerator
from app.instruments.bass_performance import BassPerformanceEngine
from app.instruments.bass_synthesizer import BassSynthesizer
from app.music.musical_context import MusicalContext
from app.music.chord_chart import parse_chord
from app.input.chart_semantic_classifier import ChartSemanticClassifier
from app.music.constants import BASS_DEFAULT_VOLUME


class BassPlayer(VirtualInstrument):
    """Baixista Virtual Inteligente que acompanha a harmonia e o andamento em tempo real."""

    def __init__(
        self,
        sample_rate: int = 44100,
        volume: float = BASS_DEFAULT_VOLUME,
        pattern: BassPatternType = BassPatternType.AUTO,
        note_value: BassNoteValue = BassNoteValue.QUARTER,
        harmony_source: BassHarmonySource = BassHarmonySource.FOLLOW,
    ):
        self._enabled: bool = True
        self._pattern_override: BassPatternType = pattern
        self._note_value = note_value
        self._harmony_source = harmony_source

        # Módulos especializados com responsabilidades separadas
        self._decision_engine = BassDecisionEngine()
        self._pattern_generator = BassPatternGenerator()
        self._performance_engine = BassPerformanceEngine()
        self._synthesizer = BassSynthesizer(sample_rate=sample_rate, volume=volume)
        self._synthesizer.strict_mono = pattern == BassPatternType.FUNDAMENTALS

        # Estado instantâneo de execução
        self._current_decision: Optional[BassDecision] = None
        self._current_event: Optional[BassNoteEvent] = None
        self._last_triggered_beat: Optional[Tuple[int, int]] = None
        self._last_chord: str = "--"
        self._last_scheduled_time: Optional[float] = None
        self._generation_id: Optional[int] = None
        self._last_chart_position_chord: str = "--"
        self._chart_rhythm_anchor_grid: float = 0.0

        # Histórico recente para modo debug e UI (ring-buffer)
        self._recent_events: collections.deque = collections.deque(maxlen=30)
        self._last_context: Optional[MusicalContext] = None

    @property
    def instrument_name(self) -> str:
        return "Virtual Bassist"

    @property
    def synthesizer(self) -> BassSynthesizer:
        return self._synthesizer

    @property
    def decision_engine(self) -> BassDecisionEngine:
        return self._decision_engine

    @property
    def pattern_generator(self) -> BassPatternGenerator:
        return self._pattern_generator

    @property
    def performance_engine(self) -> BassPerformanceEngine:
        return self._performance_engine

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = bool(val)
        self._synthesizer.enabled = self._enabled
        if not self._enabled:
            self._synthesizer.clear()

    @property
    def pattern(self) -> BassPatternType:
        return self._pattern_override

    @pattern.setter
    def pattern(self, p: BassPatternType) -> None:
        if p == BassPatternType.FUNDAMENTALS and self._pattern_override != p:
            self._synthesizer.stop_voices()
        self._pattern_override = p
        self._synthesizer.strict_mono = p == BassPatternType.FUNDAMENTALS
        self._synthesizer.cancel_scheduled()
        self._last_triggered_beat = None

    @property
    def note_value(self) -> BassNoteValue:
        return self._note_value

    @note_value.setter
    def note_value(self, value: BassNoteValue) -> None:
        self._note_value = BassNoteValue(value)
        self._synthesizer.cancel_scheduled()
        self._last_triggered_beat = None

    def prepare_chart_start(self, context: MusicalContext) -> Optional[BassNoteEvent]:
        """Arma o primeiro ataque antes que a reprodução comece em t=0."""
        if (not self._enabled or self._harmony_source != BassHarmonySource.CHART or
                not context.chart_available):
            return None
        self._synthesizer.set_generation(context.position_generation)
        self._generation_id = context.position_generation
        self._last_triggered_beat = None
        self._last_chart_position_chord = context.chord
        self._chart_rhythm_anchor_grid = 0.0
        return self._schedule_root_grid(context, allow_current=True)

    @property
    def harmony_source(self) -> BassHarmonySource:
        return self._harmony_source

    @harmony_source.setter
    def harmony_source(self, value: BassHarmonySource) -> None:
        self._harmony_source = BassHarmonySource(value)
        self._synthesizer.cancel_scheduled()
        self._synthesizer.stop_voices()
        self._last_triggered_beat = None
        self._last_chart_position_chord = "--"
        self._chart_rhythm_anchor_grid = 0.0

    @property
    def volume(self) -> float:
        return self._synthesizer.volume

    @volume.setter
    def volume(self, v: float) -> None:
        self._synthesizer.volume = v

    @property
    def current_decision(self) -> Optional[BassDecision]:
        return self._current_decision

    @property
    def current_event(self) -> Optional[BassNoteEvent]:
        return self._current_event

    @property
    def lookahead(self) -> Dict:
        """Projeção fornecida pela sessão; o baixo não recalcula posição na cifra."""
        ctx = self._last_context
        if ctx is None:
            return {}
        return {
            "current_chord": ctx.chord, "next_chord": ctx.next_expected_chord,
            "current_bar": ctx.bar, "current_beat": ctx.beat,
            "next_change_position": (ctx.next_change_bar, ctx.next_change_beat),
            "current_chord_elapsed_beats": ctx.current_chord_elapsed_beats,
            "expected_chord_duration_beats": ctx.expected_chord_duration_beats,
            "beats_until_change": ctx.beats_until_chord_change,
            "current_section": ctx.current_section,
            "next_section": ctx.next_expected_section,
            "tracking_confidence": ctx.position_confidence,
        }

    def get_recent_events(self) -> List[BassNoteEvent]:
        """Retorna lista dos eventos de baixo recentes."""
        return list(self._recent_events)

    def get_recent_decisions(self) -> List[Dict]:
        """Retorna representação serializada para exibição na UI e log de debug."""
        return [ev.to_dict() for ev in self._recent_events]

    def get_prediction(self, current_bar: int = 1, current_beat: int = 1) -> Tuple[str, str]:
        """Calcula a previsão da próxima nota a ser executada pelo baixista."""
        if not self._enabled or self._current_decision is None:
            return ("--", "Aguardando harmonia")

        dec = self._current_decision
        if dec.pattern_type == BassPatternType.SUSTAINED:
            return (f"{dec.root_note_name} (Sustentada)", f"Sustenta até Compasso {current_bar + 1}")

        # Padrões com múltiplos passos por compasso
        steps = self._pattern_generator.generate_pattern(
            dec, beats=4, chord_duration_beats=4.0, note_value=self._note_value)
        for step in steps:
            b, note_name, _, reason = step
            if b > current_beat:
                return (f"{note_name} ({reason})", f"Tempo {b} do Compasso {current_bar}")

        # Se passou de todos os tempos deste compasso, prevê o tempo 1 do próximo compasso
        if steps:
            first_step = steps[0]
            pred_text = f"{first_step[1]} ({first_step[3]})"
            target_text = f"Tempo {first_step[0]} do Compasso {current_bar + 1}"
            
            # Enriquece a predição se houver mudança de seção prevista com alta certeza
            if (self._last_context and 
                self._last_context.predicted_next_section not in ("UNKNOWN", "--") and 
                self._last_context.bars_until_change <= 1 and 
                self._last_context.prediction_confidence >= 0.70):
                next_sec = self._last_context.predicted_next_section
                target_text += f" [-> {next_sec}]"

            return (pred_text, target_text)

        return ("--", "Aguardando próximo compasso")

    def reset(self) -> None:
        """Reinicia estado do baixista, histórico e sintetizador."""
        self._decision_engine.reset()
        self._synthesizer.clear()
        self._current_decision = None
        self._current_event = None
        self._last_triggered_beat = None
        self._last_chord = "--"
        self._last_scheduled_time = None
        self._generation_id = None
        self._last_chart_position_chord = "--"
        self._chart_rhythm_anchor_grid = 0.0
        self._last_context = None
        self._recent_events.clear()

    def generate_bar_events(self, context: MusicalContext, bar: int, bar_start_time: float) -> List[BassNoteEvent]:
        """Gera antecipadamente todos os eventos do compasso para o acorde do contexto."""
        if context.chord == "--":
            return []

        bpm = context.bpm if context.bpm > 0 else 120.0
        beat_dur = 60.0 / bpm
        beats = 4

        # 1. Decisão harmônica
        decision = self._decision_engine.decide(context, pattern_override=self._pattern_override)
        self._current_decision = decision

        # 2. Geração do padrão
        pattern_steps = self._pattern_generator.generate_pattern(
            decision,
            beats=beats,
            chord_duration_beats=4.0,
            note_value=self._note_value,
        )

        # 3. Performance sincronizada com o MusicalClock
        events = self._performance_engine.create_events_for_bar(
            decision=decision,
            pattern_steps=pattern_steps,
            bar=bar,
            bar_start_time=bar_start_time,
            beat_duration=beat_dur,
            total_beats=beats
        )
        if decision.pattern_type == BassPatternType.FUNDAMENTALS:
            for event in events:
                event.duration = round(beat_dur * self._note_value.beats * .85, 4)
        return events

    def on_musical_context(self, context: MusicalContext) -> Optional[BassNoteEvent]:
        """Processa o contexto musical em tempo real e dispara notas de baixo no instante do tempo musical."""
        if not self._enabled:
            return None

        chart_chord_changed = False
        if self._harmony_source == BassHarmonySource.CHART:
            # O Play Along já publicou a posição da cifra localizada pelo áudio.
            # O relógio fornece o instante, mas nunca escolhe outro acorde.
            if context.chord != self._last_chart_position_chord:
                previous_root = parse_chord(self._last_chart_position_chord).root
                next_root = parse_chord(context.chord).root
                chart_chord_changed = (previous_root not in ("", "--") and
                                       next_root not in ("", "--") and
                                       previous_root != next_root)
                self._synthesizer.cancel_scheduled()
                self._last_chart_position_chord = context.chord
            if not context.chart_available:
                return None
        self._last_context = context
        performance_state = getattr(context, "performance_state", "PLAYING")
        if performance_state in ("HOLDING", "WAITING", "ENDED"):
            self._synthesizer.cancel_scheduled()
            return None
        if (self._harmony_source != BassHarmonySource.CHART and
                getattr(context, "tracking_state", "TRACKING") in ("LOST", "RECOVERING")):
            self._synthesizer.cancel_scheduled()
            return None
        if (self._harmony_source != BassHarmonySource.CHART and
                context.audio_activity >= 0.005 and
                context.smoothed_detected_chord in ("", "--", "UNKNOWN", "N") and
                context.position_confidence < 0.65):
            self._synthesizer.cancel_scheduled()
            return None
        active_chord = context.chord
        if (active_chord == "--" or
                not ChartSemanticClassifier.is_chord_shaped(active_chord)):
            self._synthesizer.cancel_scheduled()
            return None
        if context.tempo_tracking_state != "UNINITIALIZED":
            return self._schedule_next_beat(
                context, chart_chord_changed and performance_state == "PLAYING")

        # Rastreia compasso e tempo atuais
        bar = max(1, context.bar)
        beat = max(1, context.beat)
        beat_key = (bar, beat)

        # Evita disparos duplicados dentro do mesmo tempo musical
        if self._last_triggered_beat == beat_key:
            return None

        self._last_triggered_beat = beat_key

        bpm = context.bpm if context.bpm > 0 else 120.0
        beat_dur = 60.0 / bpm

        # 1. Decisão harmônica
        decision_context = copy.copy(context) if active_chord != context.chord else context
        decision_context.chord = active_chord
        if decision_context is not context:
            symbol = parse_chord(active_chord)
            decision_context.chord_root = symbol.root
            decision_context.chord_quality = symbol.quality
            decision_context.bass_note = symbol.bass_note or symbol.root
            decision_context.inversion = "slash" if symbol.bass_note and symbol.bass_note != symbol.root else "root"
        decision = self._decision_engine.decide(
            decision_context, pattern_override=self._pattern_override)
        self._current_decision = decision

        # 2. Padrão rítmico
        chord_beats = max(1.0, (context.chord_duration / beat_dur)) if context.chord_duration > 0 else 4.0
        pattern_steps = self._pattern_generator.generate_pattern(
            decision,
            beats=4,
            chord_duration_beats=chord_beats
        )

        # 3. Localiza a nota correspondente a este tempo musical específico
        matched_step = None
        for step in pattern_steps:
            if step[0] == beat:
                matched_step = step
                break

        if matched_step is None:
            return None

        _, note_name, midi_note, reason = matched_step

        # Enriquece razão com antecipação estrutural preditiva
        if (beat == 4 and 
            context.prediction_confidence >= 0.75 and 
            context.bars_until_change <= 1 and 
            context.predicted_next_section not in ("UNKNOWN", "--")):
            reason = f"{reason} [Antecipa {context.predicted_next_section}]"

        # Duração e articulação
        is_sustained = (decision.pattern_type == BassPatternType.SUSTAINED)
        duration = (beat_dur * 4.0 * 0.95) if is_sustained else (beat_dur * 0.85)

        # Dinâmica
        velocity = 100 if beat == 1 else 92 if beat == 3 else 85

        ev = BassNoteEvent(
            note=note_name,
            midi_note=midi_note,
            start_time=round(context.timestamp, 4),
            duration=round(duration, 4),
            velocity=velocity,
            beat=beat,
            bar=bar,
            confidence=decision.confidence,
            reason=reason
        )

        # Dispara som no sintetizador local
        self._synthesizer.trigger_note(midi_note, velocity, duration)

        # Registra no histórico recente
        self._recent_events.append(ev)
        self._current_event = ev

        return ev

    def _schedule_next_beat(self, context: MusicalContext,
                            chart_chord_changed: bool = False) -> Optional[BassNoteEvent]:
        """Agenda no sintetizador a próxima batida; nunca recupera batidas antigas."""
        if context.position_generation != self._generation_id:
            self._synthesizer.set_generation(context.position_generation)
            self._generation_id = context.position_generation
            self._last_triggered_beat = None
            self._last_chord = "--"
            self._last_scheduled_time = None
            if self._harmony_source == BassHarmonySource.CHART:
                self._chart_rhythm_anchor_grid = 0.0
        if self._pattern_override == BassPatternType.FUNDAMENTALS:
            return self._schedule_root_grid(context, chord_change=chart_chord_changed)
        recovering = context.performance_state == "RECOVERING"
        bpm = context.bpm if context.bpm > 0 else 120.0
        beat_duration = 60.0 / bpm
        beats = max(1, int(context.meter.split('/')[0]))
        phase = max(0.0, min(1.0, context.beat_position))
        chart_only = self._harmony_source == BassHarmonySource.CHART
        bar = max(1, context.clock_bar if chart_only else context.bar)
        beat = max(1, context.clock_beat if chart_only else context.beat)
        current_start = context.timestamp - phase * beat_duration
        if context.timestamp - current_start > 0.015:
            beat += 1
            start_time = current_start + beat_duration
        else:
            start_time = current_start
        if beat > beats:
            beat = 1
            bar += 1
        # A recuperação prepara somente uma entrada em downbeat; não entra no meio da frase.
        if recovering and beat != 1:
            start_time += (beats - beat + 1) * beat_duration
            bar += 1
            beat = 1
        key = (bar, beat)
        if chart_only:
            next_chord = context.chord
            source = "chart-only"
            if (next_chord == "--" or
                    not ChartSemanticClassifier.is_chord_shaped(next_chord)):
                self._synthesizer.cancel_scheduled()
                return None
        else:
            chart_changes_here = (bar > context.bar and context.next_expected_chord != "--" and
                                  (context.next_change_bar == 0 or bar >= context.next_change_bar))
            live_harmony_holds = (context.audio_activity >= 0.005 and
                                  context.harmonic_event_chord != context.next_expected_chord)
            if live_harmony_holds:
                chart_changes_here = False
            next_chord = context.next_expected_chord if chart_changes_here else context.chord
            source = "chart" if context.chart_available or context.next_expected_chord != "--" else "audio"
            beats_to_target = max(0.0, (start_time - context.timestamp) / beat_duration)
            rhythm_changes_here = (
                context.rhythmic_next_chord != "--" and
                context.duration_confidence >= 0.55 and context.pattern_confidence >= 0.55 and
                context.beats_until_chord_change <= beats_to_target + 0.15)
            if (context.audio_activity >= 0.005 and
                    context.harmonic_event_chord != context.rhythmic_next_chord):
                rhythm_changes_here = False
            if rhythm_changes_here:
                next_chord = context.rhythmic_next_chord
                source = "chart-pattern"
            if (context.confirmed_variation_chord != "--" and
                    (bar == context.bar or
                     (context.tracking_state in ("UNCERTAIN", "LOST") and
                      context.position_confidence < 0.65))):
                next_chord = context.confirmed_variation_chord
                source = "audio-confirmed"
        follow_level = getattr(context, "follow_confidence_level", "MEDIUM")
        # Baixa confiança mantém a banda presente, mas só em um ponto seguro e
        # sem preencher os tempos intermediários. Em média, AUTO evita notas de
        # passagem e fica em fundamental/quinta.
        if follow_level == "LOW" and beat != 1 and source != "audio-confirmed":
            return None
        effective_pattern = self._pattern_override
        if self._pattern_override == BassPatternType.AUTO:
            if follow_level == "LOW":
                # Uma variação já confirmada é evidência forte suficiente para
                # uma fundamental segura no beat atual; ainda não há antecipação.
                effective_pattern = (BassPatternType.ROOT if source == "audio-confirmed"
                                     else BassPatternType.SUSTAINED)
            elif follow_level == "MEDIUM":
                effective_pattern = BassPatternType.ROOT_FIFTH
        same_key = key == self._last_triggered_beat
        # O callback remove o evento da fila ao renderizar. Uma correção de fase
        # posterior não pode transformar o mesmo beat em um segundo ataque.
        if same_key and self._synthesizer.was_rendered(key):
            return None
        if (same_key and next_chord == self._last_chord and
                self._last_scheduled_time is not None and
                abs(start_time - self._last_scheduled_time) < 0.005):
            return None

        decision_context = context
        if next_chord != context.chord:
            decision_context = copy.copy(context)
            decision_context.chord = next_chord
            symbol = parse_chord(next_chord)
            decision_context.chord_root = symbol.root
            decision_context.chord_quality = symbol.quality
            decision_context.bass_note = symbol.bass_note or symbol.root
            decision_context.inversion = "slash" if symbol.bass_note and symbol.bass_note != symbol.root else "root"
        decision_started = time.perf_counter()
        decision = self._decision_engine.decide(decision_context, pattern_override=effective_pattern)
        context.decision_latency = (time.perf_counter() - decision_started) * 1000.0
        self._current_decision = decision
        steps = self._pattern_generator.generate_pattern(decision, beats=beats,
                                                         chord_duration_beats=float(beats))
        step = next((item for item in steps if item[0] == beat), None)
        self._last_triggered_beat = key
        self._last_chord = next_chord
        self._last_scheduled_time = start_time
        if step is None:
            return None
        _, note_name, midi_note, reason = step
        duration = beat_duration * (beats * 0.95 if decision.pattern_type == BassPatternType.SUSTAINED else 0.85)
        velocity = 100 if beat == 1 else 92 if beat == 3 else 85
        event = BassNoteEvent(note=note_name, midi_note=midi_note,
                              start_time=round(start_time, 4), duration=round(duration, 4),
                              velocity=velocity, beat=beat, bar=bar,
                              confidence=decision.confidence, reason=reason)
        execution_time = start_time
        can_compensate = (follow_level == "HIGH" and source in ("chart", "chart-pattern") and
                          context.chart_available and context.position_confidence >= 0.75)
        if can_compensate:
            compensation = max(0.0, context.total_estimated_latency) / 1000.0
            candidate = start_time - compensation
            # Se a análise chegou tarde, não transforma uma previsão em ataque atrasado.
            if candidate >= context.timestamp + 0.010:
                execution_time = candidate
                reason = f"{reason} [Compensa {compensation * 1000.0:.0f} ms]"
                event.reason = reason
        scheduler_started = time.perf_counter()
        self._synthesizer.schedule_note(key, execution_time, midi_note, velocity, duration,
                                        source=source, confidence=context.position_confidence,
                                        generation_id=context.position_generation, note=note_name,
                                        musical_time=start_time)
        context.scheduling_latency = (time.perf_counter() - scheduler_started) * 1000.0
        context.refresh_total_latency()
        if same_key and self._recent_events and (self._recent_events[-1].bar, self._recent_events[-1].beat) == key:
            self._recent_events.pop()
        self._recent_events.append(event)
        self._current_event = event
        return event

    def _schedule_root_grid(self, context: MusicalContext,
                            allow_current: bool = False,
                            chord_change: bool = False) -> Optional[BassNoteEvent]:
        """Agenda só a tônica na grade do relógio, com antecedência ao áudio de saída."""
        bpm = context.bpm if context.bpm > 0 else 120.0
        beat_duration = 60.0 / bpm
        beats_per_bar = max(1, int(context.meter.split("/")[0]))
        # A cifra pode aguardar um acorde por vários compassos. O ritmo usa
        # sempre o relógio, enquanto a escolha da nota usa a cifra localizada.
        use_clock = bool(context.chart_available)
        clock_bar = max(1, context.clock_bar if use_clock else context.bar)
        clock_beat = max(1, context.clock_beat if use_clock else context.beat)
        phase = max(0.0, min(.999999, context.beat_position))
        current_grid = ((clock_bar - 1) * beats_per_bar + clock_beat - 1 + phase)
        interval = self._note_value.beats
        chart_only = self._harmony_source == BassHarmonySource.CHART
        if chord_change and chart_only:
            # Troca harmônica é um novo ataque, mesmo entre os pulsos 1/3 da
            # mínima. A grade seguinte passa a contar da batida dessa troca.
            self._chart_rhythm_anchor_grid = math.floor(current_grid)
            # A análise pode terminar depois que o callback já gerou o buffer
            # atual. Deixar um buffer conhecido à frente evita perder a troca.
            output_horizon = max(0.0, context.output_latency) / 1000.0
            delay = max(.020, output_horizon + .010)
            start_time = context.timestamp + delay
            target_grid = current_grid + delay / beat_duration
            target_bar = int(target_grid // beats_per_bar) + 1
            beat_in_bar = target_grid % beats_per_bar
            target_beat = int(beat_in_bar) + 1
            fraction = round(beat_in_bar - int(beat_in_bar), 6)
            key = (target_bar, target_beat, "chord-change")
        else:
            # Uma leitura que chegou em cima do ataque não gera nota atrasada.
            # O próximo ponto é preparado antes do callback de áudio alcançá-lo.
            lead_beats = (-1e-9 if allow_current and current_grid <= 1e-9
                          else .015 / beat_duration)
            anchor = self._chart_rhythm_anchor_grid if chart_only else 0.0
            grid_index = math.ceil((current_grid + lead_beats - anchor - 1e-9) / interval)
            target_grid = max(0.0, anchor + grid_index * interval)
            if context.performance_state == "RECOVERING":
                target_grid = math.ceil((current_grid + lead_beats) / beats_per_bar) * beats_per_bar
            start_time = context.timestamp + (target_grid - current_grid) * beat_duration
            target_bar = int(target_grid // beats_per_bar) + 1
            beat_in_bar = target_grid % beats_per_bar
            target_beat = int(beat_in_bar) + 1
            fraction = round(beat_in_bar - int(beat_in_bar), 6)
            key = ((target_bar, target_beat) if fraction == 0.0 else
                   (target_bar, target_beat, round(fraction, 3)))
        if (not chart_only and context.follow_confidence_level == "LOW" and
                context.audio_activity >= .005 and
                (target_beat != 1 or fraction != 0.0)):
            chart_root = parse_chord(context.chord).root
            stable_harmony = (context.chart_available and
                              context.position_confidence >= .75 and
                              context.harmonic_event_root == chart_root and
                              context.harmonic_event_confidence >= .65)
            if not stable_harmony:
                self._synthesizer.cancel_scheduled()
                return None

        if chart_only:
            target_chord = context.chord
            if (target_chord == "--" or
                    not ChartSemanticClassifier.is_chord_shaped(target_chord)):
                self._synthesizer.cancel_scheduled()
                return None
            source = "chart-only"
        else:
            selected = self._choose_follow_chord(context, clock_bar, target_bar)
            if selected is None:
                self._synthesizer.cancel_scheduled()
                return None
            target_chord, source = selected
        same_key = key == self._last_triggered_beat
        if same_key and self._synthesizer.was_rendered(key):
            return None
        if (same_key and target_chord == self._last_chord and
                self._last_scheduled_time is not None and
                abs(start_time - self._last_scheduled_time) < .005):
            return None

        decision_context = copy.copy(context)
        decision_context.chord = target_chord
        symbol = parse_chord(target_chord)
        decision_context.chord_root = symbol.root
        decision_context.chord_quality = symbol.quality
        decision_context.bass_note = symbol.bass_note or symbol.root
        decision_context.inversion = "slash" if symbol.bass_note and symbol.bass_note != symbol.root else "root"
        started = time.perf_counter()
        decision = self._decision_engine.decide(
            decision_context, pattern_override=BassPatternType.FUNDAMENTALS)
        context.decision_latency = (time.perf_counter() - started) * 1000.0
        self._current_decision = decision
        duration = interval * beat_duration * .85
        velocity = 100 if target_beat == 1 and fraction == 0.0 else 88 if fraction == 0 else 78
        event = BassNoteEvent(
            note=decision.root_note, midi_note=decision.root_midi,
            start_time=round(start_time, 4), duration=round(duration, 4),
            velocity=velocity, beat=target_beat, bar=target_bar,
            confidence=decision.confidence, reason="Fundamental da cifra" if chart_only else "Fundamental")
        self._last_triggered_beat = key
        self._last_chord = target_chord
        self._last_scheduled_time = start_time
        scheduled = time.perf_counter()
        self._synthesizer.schedule_note(
            key, start_time, decision.root_midi, velocity, duration,
            source=source, confidence=context.position_confidence,
            generation_id=context.position_generation,
            note=decision.root_note, musical_time=start_time)
        context.scheduling_latency = (time.perf_counter() - scheduled) * 1000.0
        context.refresh_total_latency()
        if same_key and self._recent_events and self._recent_events[-1].start_time == event.start_time:
            self._recent_events.pop()
        self._recent_events.append(event)
        self._current_event = event
        return event

    def _choose_follow_chord(self, context: MusicalContext, clock_bar: int,
                             target_bar: int) -> Optional[Tuple[str, str]]:
        """Modo original: áudio pode confirmar/corrigir a harmonia prevista."""
        target_chord = context.chord
        source = "chart" if context.chart_available else "audio"
        observed_root = context.harmonic_event_root
        chart_root = parse_chord(context.chord).root
        variation_root = parse_chord(context.confirmed_variation_chord).root
        stable_root = parse_chord(context.smoothed_detected_chord).root
        next_root = parse_chord(context.next_expected_chord).root
        stable_audible = (context.audio_activity >= .005 and
                          context.stable_chord_confidence >= .70 and
                          context.stable_chord_duration >= .20 and
                          stable_root not in ("", "--"))
        audio_confirms_next = (stable_audible and
                               context.next_expected_chord != "--" and
                               stable_root == next_root)
        if (stable_audible and stable_root not in (chart_root, variation_root) and
                not audio_confirms_next):
            return None
        if (context.audio_activity >= .005 and
                context.harmonic_event_confidence >= .65 and
                context.current_chord_elapsed_beats >= .5 and
                observed_root not in ("--", chart_root, variation_root, next_root)):
            return None
        chart_change = (target_bar > clock_bar and context.next_expected_chord != "--" and
                        context.next_change_bar == context.bar + 1)
        if chart_change and context.audio_activity >= .005:
            current_root = parse_chord(context.chord).root
            chart_change = (
                context.follow_confidence_level == "HIGH" and
                context.position_confidence >= .80 and
                context.phase_confidence >= .45 and
                context.stable_chord_confidence >= .60 and
                context.harmonic_event_root == current_root
            )
        if chart_change:
            target_chord = context.next_expected_chord
        if audio_confirms_next:
            target_chord = context.next_expected_chord
            source = "audio-chart-confirmed"
        if (context.confirmed_variation_chord != "--" and
                context.position_confidence >= .65):
            target_chord = context.confirmed_variation_chord
            source = "audio-confirmed"
        return target_chord, source
