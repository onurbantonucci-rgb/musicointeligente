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
from app.analysis.chart_harmony_aligner import HarmonyTimeline, MIN_CONFIDENCE
from app.instruments.bass_arranger import BassArranger, BassArrangement, BarPlan
from app.instruments.bass_styles import STYLES, DEFAULT_STYLE
from app.instruments.bass_intent import (BarContext, BassMusicalIntent, BassMusicalIntentEngine,
                                         format_brain_report)
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
        self._synthesizer.strict_mono = pattern in (BassPatternType.FUNDAMENTALS, BassPatternType.STYLE)

        # Estado instantâneo de execução
        self._current_decision: Optional[BassDecision] = None
        self._current_event: Optional[BassNoteEvent] = None
        self._last_triggered_beat: Optional[Tuple[int, int]] = None
        self._last_chord: str = "--"
        self._last_scheduled_time: Optional[float] = None
        self._generation_id: Optional[int] = None
        self._last_chart_position_chord: str = "--"
        # Cifra alinhada ao arquivo de áudio (pré-análise). Quando presente, o modo
        # cifra agenda cada nota na batida real do áudio com o acorde da cifra.
        self._harmony_timeline: Optional[HarmonyTimeline] = None
        self._timeline_chords: Dict[tuple, str] = {}
        # Modo ESTILO: arranjo criativo da música inteira (BassArranger).
        self._style_name: str = DEFAULT_STYLE
        self._creativity: int = 3
        self._variation_seed: int = 0
        self._arrangement: Optional[BassArrangement] = None
        self._arrangement_key: Optional[tuple] = None
        self._arrangement_id: int = 0
        # Intenção musical AO VIVO (sem pré-análise): uma decisão por compasso.
        self._intent_engine = BassMusicalIntentEngine(self._style_name, self._creativity)
        self._intent_bar: Optional[Tuple[int, int]] = None
        self._live_intent: Optional[BassMusicalIntent] = None

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
        self._synthesizer.strict_mono = p in (BassPatternType.FUNDAMENTALS, BassPatternType.STYLE)
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
        self._last_chart_position_chord = self._published_chart_chord(context)
        timeline = self._active_timeline(chart_only=True)
        if timeline is not None:
            if self._pattern_override == BassPatternType.STYLE:
                return self._schedule_from_arrangement(context, timeline)
            return self._schedule_from_timeline(context, timeline)
        return self._schedule_root_grid(context, allow_current=True)

    @property
    def harmony_timeline(self) -> Optional[HarmonyTimeline]:
        return self._harmony_timeline

    def set_harmony_timeline(self, timeline: Optional[HarmonyTimeline]) -> None:
        """Instala (ou remove) a cifra alinhada ao áudio do arquivo atual."""
        if timeline is self._harmony_timeline:
            return
        self._harmony_timeline = timeline
        self._timeline_chords.clear()
        self._synthesizer.cancel_scheduled()
        self._last_triggered_beat = None

    # ------------------------------------------------------------ modo ESTILO
    @property
    def style_name(self) -> str:
        return self._style_name

    @style_name.setter
    def style_name(self, name: str) -> None:
        if name not in STYLES:
            raise ValueError(f"Estilo desconhecido: {name}")
        self._style_name = name
        self._intent_engine = BassMusicalIntentEngine(name, self._creativity)
        self._invalidate_arrangement()

    @property
    def creativity(self) -> int:
        return self._creativity

    @creativity.setter
    def creativity(self, level: int) -> None:
        self._creativity = max(1, min(5, int(level)))
        self._intent_engine.creativity = self._creativity
        self._invalidate_arrangement()

    @property
    def variation_seed(self) -> int:
        return self._variation_seed

    def new_variation(self) -> int:
        """Gera outra versão do arranjo (mesmo estilo e criatividade)."""
        self._variation_seed += 1
        self._invalidate_arrangement()
        return self._variation_seed

    @property
    def arrangement(self) -> Optional[BassArrangement]:
        return self._arrangement

    def style_status(self, time: float) -> str:
        """Resumo do que o arranjo está tocando no instante ``time`` (para a interface)."""
        arrangement = self._arrangement
        if arrangement is None:
            return f"{STYLES[self._style_name].label} · aguardando análise da música"
        plan: Optional[BarPlan] = arrangement.bar_at(time)
        if plan is None:
            return f"{arrangement.style_label} · aguardando a entrada"
        extra = plan.fill or (f"variação: {plan.variation}" if plan.variation else "")
        if plan.source:
            # "Artista\Música.3.mid" -> "Artista – Música"
            artist, _, song = plan.source.replace("/", "\\").partition("\\")
            title = song.rsplit(".mid", 1)[0].rsplit(".", 1)[0] if song else ""
            extra = f"inspirada em {artist} – {title}" if title else extra
        text = plan.groove
        return f"{text} · {extra}" if extra else text

    def _invalidate_arrangement(self) -> None:
        self._arrangement_key = None
        self._synthesizer.cancel_scheduled()

    def _arrangement_for(self, timeline: HarmonyTimeline) -> BassArrangement:
        key = (id(timeline), self._style_name, self._creativity, self._variation_seed)
        if self._arrangement is None or self._arrangement_key != key:
            self._arrangement = BassArranger(STYLES[self._style_name]).arrange(
                timeline, self._creativity, self._variation_seed)
            self._arrangement_key = key
            self._arrangement_id += 1
            self._synthesizer.cancel_scheduled()
        return self._arrangement

    def _schedule_from_arrangement(self, context: MusicalContext,
                                   timeline: HarmonyTimeline) -> Optional[BassNoteEvent]:
        """Agenda as próximas notas do arranjo nos instantes exatos do áudio."""
        arrangement = self._arrangement_for(timeline)
        now = max(0.0, context.timestamp)
        earliest = now - 0.001
        rendered = self._synthesizer.rendered_until_time
        if rendered is not None and now - 0.001 < rendered <= now + 0.5:
            earliest = rendered + 0.001
        first_event: Optional[BassNoteEvent] = None
        start = arrangement.first_index_at(earliest)
        end = arrangement.first_index_at(now + self._TIMELINE_HORIZON)
        for index in range(start, end):
            note = arrangement.notes[index]
            key = ("style", self._arrangement_id, index)
            if self._synthesizer.was_rendered(key) or self._synthesizer.scheduled_note(key) is not None:
                continue
            self._synthesizer.schedule_note(
                key, note.time, note.midi, note.velocity, note.duration,
                source="style", confidence=timeline.confidence,
                generation_id=context.position_generation, note=note.note, musical_time=note.time)
            event = BassNoteEvent(
                note=note.note, midi_note=note.midi, start_time=round(note.time, 4),
                duration=round(note.duration, 4), velocity=note.velocity,
                beat=note.beat_in_bar, bar=note.bar, confidence=timeline.confidence,
                reason=note.reason)
            self._recent_events.append(event)
            self._current_event = event
            if first_event is None:
                first_event = event
        return first_event

    def _fallback_pattern(self) -> BassPatternType:
        """Sem a música analisada, o modo ESTILO toca como AUTO."""
        return (BassPatternType.AUTO if self._pattern_override == BassPatternType.STYLE
                else self._pattern_override)

    def _active_timeline(self, chart_only: bool) -> Optional[HarmonyTimeline]:
        timeline = self._harmony_timeline
        if (not chart_only or timeline is None or timeline.beat_count == 0 or
                timeline.confidence < MIN_CONFIDENCE):
            return None
        return timeline

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
        chart_only = self._harmony_source == BassHarmonySource.CHART
        current_chord = self._published_chart_chord(ctx) if chart_only else ctx.chord
        next_chord = ctx.next_expected_chord if chart_only else ctx.next_expected_chord
        return {
            "current_chord": current_chord, "next_chord": next_chord,
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
        self._last_context = None
        self._recent_events.clear()
        self._intent_engine.reset()
        self._intent_bar = None
        self._live_intent = None

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

        chart_only = self._harmony_source == BassHarmonySource.CHART
        timeline = self._active_timeline(chart_only) if context.chart_available else None
        # A geração identifica uma mudança de transporte (seek, restart,
        # transposição ou nova sessão). Faça esta sincronização antes de
        # comparar a harmonia atual: assim uma posição nova nunca herda uma
        # nota, uma grade ou um acorde da posição anterior.
        self._sync_position_generation(context, chart_only, keep_voices=timeline is not None)
        if timeline is not None:
            # Cifra já alinhada ao arquivo: harmonia da cifra, batidas do áudio.
            # A posição publicada pela sessão (que só confirma trocas depois de
            # ouvi-las) não participa do agendamento.
            self._last_context = context
            if self._pattern_override == BassPatternType.STYLE:
                return self._schedule_from_arrangement(context, timeline)
            return self._schedule_from_timeline(context, timeline)
        if chart_only:
            # Sem a pré-análise (entrada ao vivo, cifra que não bate com o áudio),
            # a harmonia vem do cursor publicado da cifra e o tempo, do relógio.
            if not context.chart_available:
                return None
            active_chord = self._chart_chord_for_target(
                context, max(1, context.clock_bar), max(1, context.clock_bar))
        else:
            active_chord = context.chord
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
        if (active_chord == "--" or
                not ChartSemanticClassifier.is_chord_shaped(active_chord)):
            self._synthesizer.cancel_scheduled()
            return None
        if chart_only:
            # Uma troca confirmada no meio de um pulso vale a partir do PRÓXIMO
            # pulso da grade: nunca criamos ataques fora da subdivisão escolhida.
            self._last_chart_position_chord = active_chord
        if context.tempo_tracking_state != "UNINITIALIZED":
            return self._schedule_next_beat(context)

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
            decision_context, pattern_override=self._fallback_pattern())
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
        chart_only = self._harmony_source == BassHarmonySource.CHART
        if self._pattern_override == BassPatternType.FUNDAMENTALS:
            return self._schedule_root_grid(context, chord_change=chart_chord_changed)
        recovering = context.performance_state == "RECOVERING"
        bpm = context.bpm if context.bpm > 0 else 120.0
        beat_duration = 60.0 / bpm
        beats = max(1, int(context.meter.split('/')[0]))
        phase = max(0.0, min(1.0, context.beat_position))
        clock_bar = max(1, context.clock_bar if chart_only else context.bar)
        bar = clock_bar
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
            # Harmonia da cifra para o compasso do pulso, na grade do áudio (ver
            # _chart_chord_for_target): sem esperar o cursor confirmar a troca.
            next_chord = self._chart_chord_for_target(context, clock_bar, bar)
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
        if (not chart_only and follow_level == "LOW" and beat != 1 and
                source != "audio-confirmed"):
            return None
        effective_pattern = self._fallback_pattern()
        if not chart_only and effective_pattern == BassPatternType.AUTO:
            if follow_level == "LOW":
                # Uma variação já confirmada é evidência forte suficiente para
                # uma fundamental segura no beat atual; ainda não há antecipação.
                effective_pattern = (BassPatternType.ROOT if source == "audio-confirmed"
                                     else BassPatternType.SUSTAINED)
            elif follow_level == "MEDIUM":
                effective_pattern = BassPatternType.ROOT_FIFTH
        intent = None
        pattern_note_value = BassNoteValue.QUARTER
        if self._pattern_override in (BassPatternType.AUTO, BassPatternType.STYLE):
            # A intenção (seção + groove + confiança) decide quanto tocar; a
            # segurança (confiança baixa) nunca é afrouxada por ela.
            intent = self._intent_for(context, beats)
            effective_pattern, pattern_note_value = self._pattern_for_intent(intent, effective_pattern)
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
        if chart_only:
            decision_context = copy.copy(decision_context)
            decision_context.chord_confidence = 1.0
        decision_started = time.perf_counter()
        decision = self._decision_engine.decide(decision_context, pattern_override=effective_pattern)
        context.decision_latency = (time.perf_counter() - decision_started) * 1000.0
        self._current_decision = decision
        steps = self._pattern_generator.generate_pattern(decision, beats=beats,
                                                         chord_duration_beats=float(beats),
                                                         note_value=pattern_note_value)
        step = next((item for item in steps if item[0] == beat), None)
        self._last_triggered_beat = key
        self._last_chord = next_chord
        self._last_scheduled_time = start_time
        if step is None:
            return None
        _, note_name, midi_note, reason = step
        sustained = decision.pattern_type == BassPatternType.SUSTAINED
        if intent is not None:
            span = beats if sustained else pattern_note_value.beats
            duration = beat_duration * span * (0.95 if sustained else intent.note_length_factor)
            velocity = int(min(127, intent.velocity_base + (intent.accent if beat == 1 else 0)))
            reason = f"{reason} [{intent.density_target}]"
        else:
            duration = beat_duration * (beats * 0.95 if sustained else 0.85)
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

    # ------------------------------------------------------------ intenção ao vivo
    _PATTERN_RISK = {BassPatternType.SUSTAINED: 0, BassPatternType.ROOT: 1,
                     BassPatternType.ROOT_FIFTH: 2, BassPatternType.AUTO: 2}

    def _intent_for(self, context: MusicalContext, beats: int) -> BassMusicalIntent:
        """Intenção do compasso atual (recalculada só quando o compasso muda)."""
        bar_key = (context.position_generation, max(1, context.clock_bar))
        if self._live_intent is not None and self._intent_bar == bar_key:
            return self._live_intent
        chord = self._published_chart_chord(context) if context.chart_available else context.chord
        duration = (context.expected_chord_duration_beats
                    if context.expected_chord_duration_beats > 0 else float(beats))
        position_conf = context.position_confidence if context.chart_available else context.confidence
        self._live_intent = self._intent_engine.plan_bar(BarContext(
            bar_number=max(1, context.clock_bar), beats_per_bar=beats,
            section_key=context.current_section if context.current_section not in ("", "--") else "",
            section_type=getattr(context, "current_section_type", context.current_section),
            section_occurrence=getattr(context, "section_occurrence", 1),
            section_progress=context.section_progress,
            bars_until_section_change=getattr(context, "bars_until_section_change", 99),
            chords=[chord] * beats, next_chord=context.next_expected_chord,
            chord_duration_beats=duration, current_key=context.key,
            next_key=getattr(context, "next_expected_key", "--"),
            beats_until_key_change=getattr(context, "beats_until_key_change", -1.0),
            groove=getattr(context, "groove_profile", None),
            position_confidence=position_conf,
            chord_confidence=1.0 if context.chart_available else context.chord_confidence))
        self._intent_bar = bar_key
        return self._live_intent

    def _pattern_for_intent(self, intent: BassMusicalIntent,
                            safety: BassPatternType) -> Tuple[BassPatternType, BassNoteValue]:
        """Padrão que a intenção pede, nunca mais arriscado que a segurança permite."""
        if intent.fallback_pattern == "SUSTAINED" or intent.sustain_preferred or intent.attacks_cap <= 1.6:
            wanted, value = BassPatternType.SUSTAINED, BassNoteValue.WHOLE
        elif intent.fallback_pattern == "ROOT" or intent.attacks_cap <= 3.2:
            wanted, value = BassPatternType.ROOT, BassNoteValue.HALF
        else:
            wanted, value = BassPatternType.ROOT_FIFTH, BassNoteValue.QUARTER
        if self._PATTERN_RISK.get(safety, 2) < self._PATTERN_RISK[wanted]:
            value = BassNoteValue.HALF if safety == BassPatternType.ROOT else BassNoteValue.WHOLE
            return safety, value
        return wanted, value

    @property
    def live_intent(self) -> Optional[BassMusicalIntent]:
        return self._live_intent

    def brain_report(self, time: float) -> str:
        """Painel 'BASS BRAIN': o que o baixo está decidindo e por quê (dados objetivos)."""
        arrangement = self._arrangement
        if (self._pattern_override == BassPatternType.STYLE and arrangement is not None and
                self._active_timeline(chart_only=True) is not None):
            plan = arrangement.bar_at(time)
            if plan is not None and plan.intent:
                intent = BassMusicalIntent(**plan.intent)
                return format_brain_report(intent, plan.phrase or plan.groove,
                                           plan.decision or plan.groove,
                                           intent.bars_left_in_phrase)
        intent = self._live_intent
        decision = self._current_event.reason if self._current_event is not None else "--"
        return format_brain_report(intent, intent.groove_label if intent else "--", decision,
                                   intent.bars_left_in_phrase if intent else 0)

    # Janela de agendamento à frente da leitura: cobre alguns callbacks de áudio.
    _TIMELINE_HORIZON = 0.45

    def _schedule_from_timeline(self, context: MusicalContext,
                                timeline: HarmonyTimeline) -> Optional[BassNoteEvent]:
        """Agenda os próximos pulsos nas batidas REAIS do áudio com o acorde da cifra.

        A grade rítmica é a escolhida pelo usuário (subdivisão), contada a partir
        dos tempos 1 localizados no áudio; nenhum ataque extra é criado quando o
        acorde muda. Chamado a cada tick: pulsos já agendados e iguais são
        mantidos, pulsos já tocados nunca voltam.
        """
        now = max(0.0, context.timestamp)
        # Um pulso em ``now`` ainda pode entrar se o áudio não foi renderizado além
        # dele (ex.: a primeira batida em t=0); o que já saiu nunca é reagendado.
        earliest = now - 0.001
        rendered = self._synthesizer.rendered_until_time
        if rendered is not None and now - 0.001 < rendered <= now + 0.5:
            earliest = rendered + 0.001
        fundamentals = self._pattern_override == BassPatternType.FUNDAMENTALS
        interval = self._note_value.beats if fundamentals else 1.0
        generation = context.position_generation
        first_event: Optional[BassNoteEvent] = None
        for pulse in timeline.pulses(earliest, now + self._TIMELINE_HORIZON, interval):
            key = ("timeline", pulse.beat_index, pulse.subdivision)
            if self._synthesizer.was_rendered(key):
                continue
            chord = timeline.chord_at_beat(pulse.beat_index)
            pending = self._synthesizer.scheduled_note(key)
            if chord is None or not ChartSemanticClassifier.is_chord_shaped(chord):
                if pending is not None:
                    self._synthesizer.cancel_scheduled(key)
                continue
            if (pending is not None and self._timeline_chords.get(key) == chord and
                    abs(pending.scheduled_time - pulse.time) < 1e-4):
                continue
            decision_context = self._decision_context_for(context, chord)
            if fundamentals:
                decision = self._decision_engine.decide(
                    decision_context, pattern_override=BassPatternType.FUNDAMENTALS)
                note_name, midi_note, reason = decision.root_note, decision.root_midi, "Fundamental da cifra"
                duration = interval * pulse.period * .85
            else:
                decision = self._decision_engine.decide(
                    decision_context, pattern_override=self._pattern_override)
                beats = timeline.beats_per_bar
                steps = self._pattern_generator.generate_pattern(
                    decision, beats=beats, chord_duration_beats=float(beats))
                step = next((item for item in steps if item[0] == pulse.beat_in_bar), None)
                if step is None:
                    if pending is not None:
                        self._synthesizer.cancel_scheduled(key)
                    continue
                _, note_name, midi_note, reason = step
                sustained = decision.pattern_type == BassPatternType.SUSTAINED
                duration = pulse.period * (beats * .95 if sustained else .85)
            self._current_decision = decision
            on_beat = pulse.subdivision == 0
            velocity = (100 if on_beat and pulse.beat_in_bar == 1 else
                        92 if on_beat and pulse.beat_in_bar == 3 else 85 if on_beat else 78)
            self._synthesizer.schedule_note(
                key, pulse.time, midi_note, velocity, duration,
                source="chart-timeline", confidence=timeline.confidence,
                generation_id=generation, note=note_name, musical_time=pulse.time)
            self._timeline_chords[key] = chord
            event = BassNoteEvent(
                note=note_name, midi_note=midi_note, start_time=round(pulse.time, 4),
                duration=round(duration, 4), velocity=velocity,
                beat=pulse.beat_in_bar, bar=pulse.bar,
                confidence=decision.confidence, reason=reason)
            if self._recent_events and self._recent_events[-1].start_time == event.start_time:
                self._recent_events.pop()
            self._recent_events.append(event)
            self._current_event = event
            if first_event is None:
                first_event = event
        if len(self._timeline_chords) > 64:
            keep_from = timeline.beat_index_at(now) - 2
            self._timeline_chords = {key: chord for key, chord in self._timeline_chords.items()
                                     if key[1] >= keep_from}
        return first_event

    @staticmethod
    def _decision_context_for(context: MusicalContext, chord: str) -> MusicalContext:
        decision_context = copy.copy(context)
        symbol = parse_chord(chord)
        decision_context.chord = chord
        decision_context.chord_root = symbol.root
        decision_context.chord_quality = symbol.quality
        decision_context.bass_note = symbol.bass_note or symbol.root
        decision_context.inversion = ("slash" if symbol.bass_note and symbol.bass_note != symbol.root
                                      else "root")
        decision_context.chord_confidence = 1.0
        return decision_context

    @staticmethod
    def _chart_chord_for_target(context: MusicalContext, clock_bar: int,
                                target_bar: int) -> str:
        """Acorde da cifra para um pulso quando NÃO há cifra alinhada ao áudio.

        Sem a pré-análise não sabemos quantos compassos cada acorde dura (a cifra
        do Cifra Club não informa), então projetar a cifra no relógio erra com
        frequência. Usamos o cursor publicado, que só avança quando o áudio
        confirma a troca: correto, ainda que com a latência da confirmação.
        """
        for chord in (getattr(context, "chart_published_chord", "--"), context.chord,
                      context.chart_clock_chord):
            if ChartSemanticClassifier.is_chord_shaped(chord):
                return chord
        return "--"

    @staticmethod
    def _published_chart_chord(context: MusicalContext) -> str:
        """Retorna o acorde da ChartPosition visível, sem consultar o áudio."""
        chord = getattr(context, "chart_published_chord", "--")
        if ChartSemanticClassifier.is_chord_shaped(chord):
            return chord
        # Compatibilidade com contexts de chamadas antigas e testes isolados.
        if ChartSemanticClassifier.is_chord_shaped(context.chord):
            return context.chord
        return context.chart_clock_chord

    def _sync_position_generation(self, context: MusicalContext,
                                  chart_only: bool, keep_voices: bool = False) -> None:
        """Invalida decisões de uma posição musical anterior uma única vez.

        Com a cifra alinhada ao áudio (``keep_voices``), uma relocalização da
        sessão não muda nada do que o baixo toca: só a fila é refeita, sem
        cortar a nota que está soando.
        """
        if context.position_generation == self._generation_id:
            return

        was_initialized = self._generation_id is not None
        self._synthesizer.set_generation(context.position_generation)
        self._generation_id = context.position_generation
        self._last_triggered_beat = None
        self._last_chord = "--"
        self._last_scheduled_time = None
        self._timeline_chords.clear()

        if chart_only and was_initialized and not keep_voices:
            # Nunca deixe uma voz ou previsão da posição anterior atravessar
            # um seek/restart. A chamada vem antes da comparação de acordes,
            # permitindo que o novo acorde da cifra seja atacado já neste tick.
            self._synthesizer.cancel_scheduled()
            self._synthesizer.stop_voices()
            self._last_chart_position_chord = "--"

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
        # Uma leitura que chegou em cima do ataque não gera nota atrasada.
        # O próximo ponto é preparado antes do callback de áudio alcançá-lo.
        lead_beats = (-1e-9 if allow_current and current_grid <= 1e-9
                      else .015 / beat_duration)
        grid_index = math.ceil((current_grid + lead_beats - 1e-9) / interval)
        target_grid = max(0.0, grid_index * interval)
        if not chart_only and context.performance_state == "RECOVERING":
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
            # Harmonia da cifra para o compasso ONDE o pulso vai soar, na grade
            # do áudio: o tempo 1 de um novo acorde já sai com a tônica nova.
            target_chord = self._chart_chord_for_target(context, clock_bar, target_bar)
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
        if chart_only:
            decision_context.chord_confidence = 1.0
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
            source=source, confidence=(1.0 if chart_only else context.position_confidence),
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
