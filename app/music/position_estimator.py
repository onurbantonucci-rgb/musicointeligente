"""Estimador Contínuo de Posição Musical na Cifra (PositionEstimator v0.4).

Combina múltiplas fontes de percepção e expectativas:
1. Relógio Musical (MusicalClock - tempo, BPM, compasso, beat)
2. Análise de Áudio (Pitch, Chroma, Acorde Detectado, Confiança)
3. Cifra Estruturada (ChordChart, ChartAlignment, Linhas Físicas)
4. Memória de Padrões e Estrutura (MusicStructureAnalyzer)
5. Motor de Predição (PredictionEngine)
6. Histórico Recente de Posições

Princípio Fundamental:
A ausência ou divergência de detecção de acordes NUNCA congela o avanço da posição.
O relógio musical e a estrutura garantem continuidade temporal monotônica com
recuperação contextual por janela local e correspondência de progressão.
"""

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any, Set

from app.music.chart_alignment import ChartAlignment, ChartPosition
from app.music.chord_chart import parse_chord
from app.music.theory import note_to_pc, get_chord_notes, PITCH_CLASSES
from app.music.musical_clock import MusicalClock
from app.analysis.music_structure_analyzer import MusicStructureAnalyzer
from app.music.prediction_engine import PredictionEngine
from app.music.transposition_tracker import TranspositionTracker
from app.music.harmonic_rhythm import section_key
from app.input.chart_semantic_classifier import ChartSemanticClassifier


class TrackingState(str, Enum):
    """Estados operacionais do rastreamento da posição musical."""
    TRACKING = "TRACKING"       # Posição plenamente confirmada e confiável
    UNCERTAIN = "UNCERTAIN"     # Dúvida momentânea (falha breve de áudio ou variação)
    LOST = "LOST"               # Ausência prolongada de confirmação (avanço 100% temporal)
    RECOVERING = "RECOVERING"   # Candidato harmônico encontrado na vizinhança local


@dataclass
class ProgressionMatch:
    """Resultado da correspondência de uma sequência de acordes na cifra."""
    matched_chords: List[str]
    expected_chords: List[str]
    start_bar: int
    end_bar: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched_chords": list(self.matched_chords),
            "expected_chords": list(self.expected_chords),
            "start_bar": self.start_bar,
            "end_bar": self.end_bar,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class EstimatedPosition:
    """Posição estimada consolidada pelo PositionEstimator."""
    bar: int
    beat: float
    line_index: int
    current_chord: str                  # Acorde efetivo utilizado
    expected_chord: str                 # Acorde nominal previsto pela cifra
    detected_chord: str                 # Acorde percebido no áudio
    predicted_chord: str                # Previsão harmônica estrutural
    section_name: str
    section_progress: float
    confidence: float                   # Confiança na posição [0.0 a 1.0]
    chord_confidence: float             # Confiança no acorde detectado [0.0 a 1.0]
    tracking_state: TrackingState
    action_note: str = ""
    timestamp: float = 0.0


class PositionEstimator:
    """Estimador contínuo de posição musical na cifra com recuperação inteligente."""

    def __init__(
        self,
        alignment: ChartAlignment,
        clock: MusicalClock,
        structure_analyzer: Optional[MusicStructureAnalyzer] = None,
        prediction_engine: Optional[PredictionEngine] = None,
        local_window_bars: int = 4
    ):
        self._alignment = alignment
        self._clock = clock
        self._structure_analyzer = structure_analyzer
        self._prediction_engine = prediction_engine
        self._local_window = local_window_bars

        # Estado instantâneo do estimador
        self._tracking_state: TrackingState = TrackingState.TRACKING
        self._position_confidence: float = 0.95
        self._last_confirmed_bar: int = 1
        self._last_confirmed_time: float = 0.0
        self._last_detected_chord: str = "--"
        self._consecutive_unknown_seconds: float = 0.0
        self._consecutive_mismatch_seconds: float = 0.0

        # Histórico recente de acordes detectados e esperados (últimos 8)
        self._recent_detected: deque = deque(maxlen=8)
        self._recent_expected: deque = deque(maxlen=8)
        self._recent_notes: deque = deque(maxlen=5)
        self._recent_harmonic_rhythm: deque = deque(maxlen=4)

        # Detector de transposição/capotraste global entre a cifra e a execução
        self._transposition = TranspositionTracker()
        # Capotraste CONHECIDO (semitons): o áudio soa este tanto acima dos shapes escritos
        self._capo_semitones: int = 0
        self._capo_domain = "WRITTEN"
        self._capo_domain_candidate = "--"
        self._capo_domain_observations = 0

        # LOCALIZAÇÃO POR ÁUDIO: o áudio decide ONDE na cifra o músico está.
        # O relógio dá o avanço suave; este deslocamento (offset) mapeia o compasso do
        # relógio para o compasso da CIFRA. A progressão detectada reancora o offset.
        self._follow_audio: bool = True
        self._bar_offset: int = 0  # chart_bar = clock_bar - offset

        # Cursor de consumo da timeline da cifra. Ele pertence ao estimador
        # existente: não é uma segunda posição nem um segundo relógio. O cursor
        # deixa explícito qual evento está ativo e quais já ficaram para trás.
        self._chart_cursor_index: int = 0
        self._consumed_chart_events: Set[int] = set()
        self._completed_sections: Set[Tuple[int, int]] = set()
        self._seen_rhythm_markers: Set[Tuple[str, float]] = set()
        self._global_recovery_confirmation_pending: bool = False
        self._pending_next_chord: str = "--"
        self._pending_next_observations: int = 0
        self._pending_next_last_beat: float = -1.0
        self._event_start_beat: float = 0.0
        self._last_position_change_reason: str = "session reset"
        self._position_transition_log: deque = deque(maxlen=64)
        self._has_seen_harmonic_observation: bool = False
        self._last_search_mode: str = "LOCAL"
        self._last_note_evidence: Dict[str, Any] = {}
        self._startup_lock: bool = True
        self._start_anchor_confirmed: bool = False

        # Última posição consolidada
        self._current_estimated_position: Optional[EstimatedPosition] = None

    def reset(self) -> None:
        """Reinicia o estado interno do estimador para o início da música."""
        self._tracking_state = TrackingState.TRACKING
        self._position_confidence = 0.95
        start = self._alignment.get_start_position()
        self._last_confirmed_bar = start.current_bar
        self._last_confirmed_time = 0.0
        self._last_detected_chord = "--"
        self._consecutive_unknown_seconds = 0.0
        self._consecutive_mismatch_seconds = 0.0
        self._recent_detected.clear()
        self._recent_expected.clear()
        self._recent_notes.clear()
        self._recent_harmonic_rhythm.clear()
        self._transposition.reset()
        self._capo_domain = "UNRESOLVED" if self._capo_semitones else "WRITTEN"
        self._capo_domain_candidate = "--"
        self._capo_domain_observations = 0
        self._bar_offset = self._clock.bar - start.current_bar
        anchor = self._alignment.start_anchor
        self._chart_cursor_index = anchor.event_index if anchor else 0
        self._consumed_chart_events.clear()
        self._completed_sections.clear()
        self._seen_rhythm_markers.clear()
        self._global_recovery_confirmation_pending = False
        self._pending_next_chord = "--"
        self._pending_next_observations = 0
        self._pending_next_last_beat = -1.0
        self._event_start_beat = 0.0
        self._last_position_change_reason = "session reset"
        self._position_transition_log.clear()
        self._has_seen_harmonic_observation = False
        self._last_search_mode = "LOCAL"
        self._last_note_evidence = {}
        self._startup_lock = True
        self._start_anchor_confirmed = False
        self._current_estimated_position = None

    @property
    def tracking_state(self) -> TrackingState:
        return self._tracking_state

    @property
    def position_confidence(self) -> float:
        return self._position_confidence

    @property
    def current_estimated_position(self) -> Optional[EstimatedPosition]:
        return self._current_estimated_position

    @property
    def transposition_semitones(self) -> int:
        """Deslocamento global detectado entre a cifra e a execução (capotraste/tom)."""
        return self._transposition.semitones

    @property
    def bar_offset(self) -> int:
        """Deslocamento conhecido: musical_bar = clock_bar - bar_offset."""
        return self._bar_offset

    @property
    def chart_cursor_index(self) -> int:
        """Evento atualmente ativo na timeline expandida da cifra."""
        return self._chart_cursor_index

    @property
    def start_anchor_confirmed(self) -> bool:
        return self._start_anchor_confirmed

    @property
    def consumed_chart_events(self) -> Set[int]:
        """Snapshot dos eventos já consumidos, útil para diagnóstico e testes."""
        return set(self._consumed_chart_events)

    @property
    def last_position_change_reason(self) -> str:
        return self._last_position_change_reason

    @property
    def current_event_elapsed_beats(self) -> float:
        return self._event_elapsed_beats()

    @property
    def position_transition_log(self) -> List[Dict[str, Any]]:
        return list(self._position_transition_log)

    @property
    def last_search_mode(self) -> str:
        return self._last_search_mode

    @property
    def last_note_evidence(self) -> Dict[str, Any]:
        return dict(self._last_note_evidence)

    @property
    def position_stability_metrics(self) -> Dict[str, int]:
        entries = list(self._position_transition_log)
        return {
            "cursor_changes": len(entries),
            "global_relocations": sum(1 for item in entries if item.get("reason") == "global progression recovery"),
            "backward_jumps": sum(1 for item in entries if item.get("distance", 0) < 0),
            "large_forward_jumps": sum(1 for item in entries if item.get("distance", 0) > 1),
        }

    def _event_elapsed_beats(self, absolute_beat: Optional[float] = None) -> float:
        beat = self._clock.total_beats if absolute_beat is None else absolute_beat
        return max(0.0, float(beat) - self._event_start_beat)

    def _section_key_for_position(self, position: ChartPosition) -> Tuple[int, int]:
        return position.section_index, position.section_occurrence

    def _cursor_position(self, beat: float, timestamp: float,
                         nominal_bar: Optional[int] = None) -> ChartPosition:
        """Posição do evento ativo, preservando o compasso dentro de acordes longos."""
        position = self._alignment.get_position_for_event(
            self._chart_cursor_index, beat=beat, absolute_time=timestamp)
        if nominal_bar is None or self._alignment.event_count <= 0:
            return position
        if self._chart_cursor_index + 1 < self._alignment.event_count:
            next_bar = self._alignment.get_position_for_event(
                self._chart_cursor_index + 1).current_bar
            event_end_bar = max(position.current_bar, next_bar - 1)
        else:
            event_end_bar = self._alignment.total_bars
        bar = max(position.current_bar, min(int(nominal_bar), event_end_bar))
        return self._alignment.get_position_at(bar, beat=beat, absolute_time=timestamp)

    def _set_chart_cursor(self, event_index: int, reason: str = "position correction",
                          absolute_beat: Optional[float] = None,
                          evidence: str = "") -> None:
        """Reposiciona o cursor e atualiza o consumo entre eventos ordenados."""
        if self._alignment.event_count <= 0:
            self._chart_cursor_index = 0
            return
        target = max(0, min(self._alignment.event_count - 1, int(event_index)))
        previous = self._chart_cursor_index
        if target > self._chart_cursor_index:
            for index in range(self._chart_cursor_index, target):
                old = self._alignment.get_position_for_event(index)
                self._consumed_chart_events.add(index)
                new = self._alignment.get_position_for_event(index + 1)
                if self._section_key_for_position(old) != self._section_key_for_position(new):
                    self._completed_sections.add(self._section_key_for_position(old))
        elif target < self._chart_cursor_index:
            # Reabrir o passado só é permitido após uma relocalização forte; o
            # chamador é responsável por validar essa evidência.
            self._consumed_chart_events = {i for i in self._consumed_chart_events if i < target}
            self._completed_sections = {
                key for key in self._completed_sections
                if key[0] < self._alignment.get_position_for_event(target).section_index
            }
        self._chart_cursor_index = target
        if target != previous:
            beat = self._clock.total_beats if absolute_beat is None else absolute_beat
            before = self._alignment.get_position_for_event(previous)
            after = self._alignment.get_position_for_event(target)
            self._last_position_change_reason = reason
            self._position_transition_log.append({
                "time": round(self._clock.elapsed_time, 3),
                "beat": round(float(beat), 3),
                "bar": after.current_bar,
                "from_event": previous,
                "to_event": target,
                "from": f"{before.section_name}/{before.section_event_index + 1}/{before.current_chord}",
                "to": f"{after.section_name}/{after.section_event_index + 1}/{after.current_chord}",
                "distance": target - previous,
                "state": self._tracking_state.value,
                "reason": reason,
                "evidence": evidence or reason,
            })
            self._event_start_beat = float(beat)
            self._pending_next_chord = "--"
            self._pending_next_observations = 0
            self._pending_next_last_beat = -1.0
            if (reason == "manual seek" or
                    (after.section_index > 0 and reason != "global progression recovery")):
                self._startup_lock = False
            anchor = self._alignment.start_anchor
            if anchor is not None and target > anchor.event_index:
                self._start_anchor_confirmed = True
                self._startup_lock = False

    def _cursor_state(self, position: ChartPosition) -> ChartPosition:
        key = self._section_key_for_position(position)
        return replace(
            position,
            event_index=position.element_index,
            next_event_index=(position.element_index + 1
                              if position.element_index + 1 < self._alignment.event_count else -1),
            event_state=("CONSUMED" if position.element_index in self._consumed_chart_events else "ACTIVE"),
            section_state=("CONSUMED" if key in self._completed_sections else "ACTIVE"),
            next_event_state=("PENDING" if position.element_index + 1 < self._alignment.event_count else "--"),
        )

    def _event_matches(self, event_index: int, detected_chord: str) -> bool:
        if not detected_chord or detected_chord in ("--", "UNKNOWN", "N"):
            return False
        expected = self._cursor_position(1.0, 0.0) if event_index == self._chart_cursor_index else self._alignment.get_position_for_event(event_index)
        expected_symbol = parse_chord(self._expected_as_sounding(expected.current_chord))
        detected_symbol = parse_chord(detected_chord)
        return self._chords_equal(expected_symbol, detected_symbol) or self._roots_equal(expected_symbol, detected_symbol)

    # Mínimo de um acorde da cifra (há cifras com um acorde por tempo): abaixo disso
    # é a voz ou um ornamento confundindo o detector.
    _MIN_EVENT_BEATS = 0.6

    def _consume_from_audio(self, detected_chord: str, detected_confidence: float = 0.0,
                            expected_duration_beats: float = 0.0,
                            duration_confidence: float = 0.0,
                            absolute_beat: Optional[float] = None) -> bool:
        """Avança apenas na janela à frente; candidatos consumidos não competem."""
        if not detected_chord or detected_chord in ("--", "UNKNOWN", "N"):
            return False
        # Uma observação isolada só pode confirmar o evento ativo ou o próximo.
        # Saltos maiores exigem a recuperação por progressão em LOST.
        last = min(self._alignment.event_count - 1, self._chart_cursor_index + 1)
        for index in range(self._chart_cursor_index, last + 1):
            if self._event_matches(index, detected_chord):
                anchor = self._alignment.start_anchor
                if (anchor is not None and index == anchor.event_index and
                        self._chart_cursor_index == anchor.event_index):
                    self._start_anchor_confirmed = True
                    self._startup_lock = False
                if index > self._chart_cursor_index:
                    if detected_chord == self._pending_next_chord:
                        beat = self._clock.total_beats if absolute_beat is None else absolute_beat
                        if beat - self._pending_next_last_beat >= 0.15:
                            self._pending_next_observations += 1
                            self._pending_next_last_beat = beat
                    else:
                        self._pending_next_chord = detected_chord
                        self._pending_next_observations = 1
                        self._pending_next_last_beat = (self._clock.total_beats if absolute_beat is None
                                                        else absolute_beat)
                    elapsed = self._event_elapsed_beats(absolute_beat)
                    duration_ready = (expected_duration_beats > 0.0 and duration_confidence >= 0.55
                                      and elapsed >= expected_duration_beats * 0.65)
                    # Um detector já estabilizado pode confirmar a troca esperada
                    # perto do fim plausível do evento. Nos demais casos, duas
                    # observações consistentes evitam que um frame mude a posição.
                    recovery_confirmation = (
                        self._global_recovery_confirmation_pending
                        and detected_confidence >= 0.75
                        and len(self._recent_detected) >= 3
                    )
                    startup_required = 3 if (self._startup_lock and
                                              not self._start_anchor_confirmed) else 2
                    confirmed = (self._pending_next_observations >= startup_required or
                                 recovery_confirmation or
                                 (not self._startup_lock and detected_confidence >= 0.85 and
                                  (duration_ready or elapsed >= 0.75)))
                    if (self._startup_lock and not self._start_anchor_confirmed and
                            self._clock.total_beats < 2.0):
                        confirmed = False
                    # Uma "troca" logo depois de entrar no acorde costuma ser a voz
                    # ou um ornamento confundindo o detector, não a banda mudando.
                    if (confirmed and not recovery_confirmation and
                            elapsed < self._MIN_EVENT_BEATS):
                        confirmed = False
                    if confirmed:
                        self._set_chart_cursor(index, "confirmed expected next chord", absolute_beat)
                        self._global_recovery_confirmation_pending = False
                else:
                    self._pending_next_chord = "--"
                    self._pending_next_observations = 0
                    self._pending_next_last_beat = -1.0
                return True
        if self._global_recovery_confirmation_pending:
            # A primeira transição após uma relocalização precisa continuar a
            # progressão encontrada. Uma divergência encerra essa autorização.
            self._global_recovery_confirmation_pending = False
        return False

    def _consume_closed_rhythm_events(self, events, successor_chord: str = "--",
                                      successor_confidence: float = 0.0,
                                      require_successor: bool = False) -> bool:
        """Consome evento fechado somente se o novo acorde confirmar o sucessor."""
        advanced = False
        for event in events or ():
            marker = (event.symbol, round(event.start_beat, 3))
            if marker in self._seen_rhythm_markers:
                continue
            self._seen_rhythm_markers.add(marker)
            if getattr(event, "end_reason", "CHANGE") != "CHANGE":
                continue
            next_index = self._chart_cursor_index + 1
            successor_confirmed = (not require_successor or
                                   (self._capo_domain != "UNRESOLVED" and
                                    successor_confidence >= 0.35 and
                                    next_index < self._alignment.event_count and
                                    self._event_matches(next_index, successor_chord)))
            # O acorde que está terminando precisa ter durado o mínimo de um acorde
            # da cifra; um "acorde" de frações de tempo é a voz/ornamento.
            lasted = min(float(getattr(event, "raw_duration_beats", self._MIN_EVENT_BEATS)),
                         self._event_elapsed_beats(event.start_beat + event.raw_duration_beats))
            if (self._event_matches(self._chart_cursor_index, event.symbol) and
                    successor_confirmed and lasted >= self._MIN_EVENT_BEATS):
                self._set_chart_cursor(self._chart_cursor_index + 1,
                                       "confirmed harmonic transition", event.start_beat + event.raw_duration_beats)
                advanced = True
        return advanced

    def _consume_temporal_progress(self, temporal: ChartPosition,
                                   elapsed_beats: float = 0.0,
                                   expected_duration_beats: float = 0.0,
                                   duration_confidence: float = 0.0) -> bool:
        """Fallback estrutural somente quando não há acorde ativo confiável.

        A posição temporal pode recuperar um trecho sem áudio, mas nunca deve
        competir com um evento harmônico que ainda está sendo confirmado.
        """
        if self._startup_lock and not self._start_anchor_confirmed:
            # Protege apenas a janela realmente inicial. Depois de dois beats,
            # holdover temporal volta a valer para a música não congelar caso o
            # instrumento entre em silêncio ou o detector permaneça UNKNOWN.
            if self._clock.total_beats < 2.0:
                return False
            self._startup_lock = False
        if temporal.element_index <= self._chart_cursor_index:
            return False
        duration_is_holding = (expected_duration_beats > 0.0 and duration_confidence >= 0.55
                               and elapsed_beats < expected_duration_beats * 0.85)
        if duration_is_holding:
            return False
        self._set_chart_cursor(temporal.element_index, "temporal fallback")
        return True

    def refresh_position(self) -> ChartPosition:
        """Reconsulta a cifra usando o offset conhecido, sem nova evidência de áudio."""
        nominal = self._alignment.get_position_at(
            max(1, self._clock.bar - self._bar_offset),
            self._clock.beat + self._clock.beat_position,
            self._clock.elapsed_time,
        )
        if self._alignment.event_count:
            self._consume_temporal_progress(nominal)
            nominal = self._cursor_state(self._cursor_position(
                self._clock.beat + self._clock.beat_position, self._clock.elapsed_time))
        return self._consolidate_position(nominal, "--", 0.0, "Posição atualizada", nominal.next_chord)

    def seek_to_bar(self, bar: int) -> ChartPosition:
        """Reancora a cifra manualmente sem alterar o tempo bruto do áudio."""
        target = max(1, min(self._alignment.total_bars, int(bar)))
        anchor = self._alignment.start_anchor
        if anchor is not None and target == self._alignment.get_start_position().current_bar:
            return self.seek_to_start()
        self.reset()
        self._bar_offset = self._clock.bar - target
        self._set_chart_cursor(self._alignment.get_position_at(target).element_index,
                               "manual seek", self._clock.total_beats)
        return self.refresh_position()

    def seek_to_start(self) -> ChartPosition:
        """Reinicia no StartAnchor mantendo a busca inicial estritamente local."""
        self.reset()
        start = self._alignment.get_start_position(
            self._clock.beat + self._clock.beat_position, self._clock.elapsed_time)
        self._bar_offset = self._clock.bar - start.current_bar
        return self.refresh_position()

    def _consolidate_position(self, nominal: ChartPosition, detected_chord: str,
                              detected_confidence: float, action_note: str,
                              predicted_chord: str) -> ChartPosition:
        position = replace(nominal, confidence=self._position_confidence,
                           tracking_state=self._tracking_state.value)
        self._current_estimated_position = EstimatedPosition(
            bar=position.current_bar, beat=position.current_beat,
            line_index=position.line_index, current_chord=position.current_chord,
            expected_chord=position.current_chord, detected_chord=detected_chord,
            predicted_chord=predicted_chord, section_name=position.section_name,
            section_progress=position.section_progress, confidence=position.confidence,
            chord_confidence=detected_confidence, tracking_state=self._tracking_state,
            action_note=action_note, timestamp=position.absolute_time,
        )
        return position

    def set_alignment(self, alignment: ChartAlignment) -> None:
        """Atualiza o alinhamento estruturado da cifra."""
        self._alignment = alignment
        self._recent_notes.clear()
        anchor = alignment.start_anchor
        self._chart_cursor_index = anchor.event_index if anchor else 0
        self._consumed_chart_events.clear()
        self._completed_sections.clear()
        self._global_recovery_confirmation_pending = False
        self._pending_next_chord = "--"
        self._pending_next_observations = 0
        self._pending_next_last_beat = -1.0
        self._event_start_beat = self._clock.total_beats
        self._has_seen_harmonic_observation = False
        self._startup_lock = True
        self._start_anchor_confirmed = False

    @property
    def capo_semitones(self) -> int:
        return self._capo_semitones

    def relocate_to_bar(self, bar: int, reason: str, note: str = "") -> None:
        """Reposiciona a cifra por uma evidência externa forte (ex.: a letra cantada)."""
        position = self._alignment.get_position_at(max(1, int(bar)))
        self._set_chart_cursor(position.element_index, reason, self._clock.total_beats, note)
        self._bar_offset = self._clock.bar - position.current_bar
        self._tracking_state = TrackingState.TRACKING
        self._position_confidence = max(self._position_confidence, 0.9)
        self._consecutive_mismatch_seconds = 0.0

    def relocate_to_event(self, event_index: int, reason: str, note: str = "") -> None:
        """Como ``relocate_to_bar``, mas por acorde da cifra (vários por compasso)."""
        if self._alignment.event_count <= 0:
            return
        event_index = max(0, min(self._alignment.event_count - 1, int(event_index)))
        position = self._alignment.get_position_for_event(event_index)
        self._set_chart_cursor(event_index, reason, self._clock.total_beats, note)
        self._bar_offset = self._clock.bar - position.current_bar
        self._tracking_state = TrackingState.TRACKING
        self._position_confidence = max(self._position_confidence, 0.85)
        self._consecutive_mismatch_seconds = 0.0

    def observe_start_chord(self, chord: str, confidence: float) -> None:
        """Observação harmônica feita durante a confirmação do início da música.

        Antes do início a cifra não se move, mas a observação ainda decide se o
        áudio soa com ou sem o capotraste (forma escrita x som real).
        """
        self._resolve_capo_domain(chord, confidence)

    def set_capo(self, semitones: int) -> None:
        """Informa o capotraste (semitons) para comparar o áudio soante com os shapes escritos."""
        self._capo_semitones = max(0, int(semitones))
        self._capo_domain = "UNRESOLVED" if self._capo_semitones else "WRITTEN"
        self._capo_domain_candidate = "--"
        self._capo_domain_observations = 0

    def _resolve_capo_domain(self, detected_chord: str, confidence: float) -> None:
        """Decide se os símbolos da cifra já estão no tom soante anunciado."""
        if (self._capo_domain != "UNRESOLVED" or confidence < 0.65 or
                detected_chord in ("", "--", "UNKNOWN", "N") or
                self._alignment.event_count == 0):
            return
        written = self._alignment.get_position_for_event(self._chart_cursor_index).current_chord
        sounding = self._transpose_for_capo(written)
        detected = parse_chord(detected_chord)
        raw_match = self._roots_equal(parse_chord(written), detected)
        transposed_match = self._roots_equal(parse_chord(sounding), detected)
        candidate = ("WRITTEN" if raw_match and not transposed_match else
                     "SOUNDING" if transposed_match and not raw_match else "--")
        if candidate == "--":
            return
        if candidate == self._capo_domain_candidate:
            self._capo_domain_observations += 1
        else:
            self._capo_domain_candidate = candidate
            self._capo_domain_observations = 1
        if self._capo_domain_observations >= 2:
            self._capo_domain = candidate

    def _transpose_for_capo(self, chord: str) -> str:
        from app.music.transposition import transpose_chord_symbol
        return transpose_chord_symbol(chord, self._capo_semitones)

    def _expected_as_sounding(self, expected_chord: str) -> str:
        """Converte o acorde escrito (shape) para o som real, aplicando o capotraste."""
        if (self._capo_semitones == 0 or self._capo_domain == "WRITTEN" or
                not expected_chord or expected_chord == "--"):
            return expected_chord
        return self._transpose_for_capo(expected_chord)

    def update(
        self,
        timestamp: float,
        detected_chord: str = "--",
        detected_confidence: float = 0.0,
        detected_key: str = "--",
        detected_note: str = "--",
        note_confidence: float = 0.0,
        harmonic_rhythm_events=None,
        current_chord_elapsed_beats: float = 0.0,
        expected_chord_duration_beats: float = 0.0,
        duration_confidence: float = 0.0,
        audio_observable: bool = False,
    ) -> ChartPosition:
        """Atualiza a estimativa de posição: o ÁUDIO localiza onde estamos na cifra."""
        cursor_before = self._chart_cursor_index
        # 1. Tempo contínuo (o relógio dá o avanço suave; o áudio decide o LUGAR)
        self._last_search_mode = "LOCAL"
        clock_bar = max(1, self._clock.bar)
        current_beat = self._clock.beat + self._clock.beat_position
        dt = max(0.0, timestamp - self._last_confirmed_time) if self._last_confirmed_time > 0 else 0.0
        self._last_confirmed_time = timestamp

        # Entradas externas que não provam ser acordes são lixo semântico: não
        # são UNKNOWN temporal e não podem acionar consumo, busca ou fallback.
        invalid_input = (bool(detected_chord) and detected_chord not in ("--", "UNKNOWN", "N") and
                         not ChartSemanticClassifier.is_chord_shaped(detected_chord))
        if invalid_input:
            nominal = (self._cursor_state(self._cursor_position(current_beat, timestamp))
                       if self._alignment.event_count else
                       self._alignment.get_position_at(self._last_confirmed_bar, current_beat, timestamp))
            return self._consolidate_position(
                nominal, "--", 0.0, "Ignored non-musical input", nominal.next_chord)

        confident_audio = bool(detected_chord) and detected_chord not in ("--", "UNKNOWN", "N") and detected_confidence >= 0.25
        if harmonic_rhythm_events:
            for event in harmonic_rhythm_events:
                marker = (event.symbol, round(event.start_beat, 3))
                if marker not in self._seen_rhythm_markers:
                    self._recent_harmonic_rhythm.append((marker, event))

        # O relógio continua sendo a referência temporal. A diferença é que ele
        # agora consome a timeline ordenada, em vez de voltar a procurar acordes
        # iguais no documento inteiro a cada frame.
        temporal_pos = self._alignment.get_position_at(
            max(1, clock_bar - self._bar_offset), current_beat, timestamp)
        self._resolve_capo_domain(detected_chord, detected_confidence)
        transition_consumed = self._consume_closed_rhythm_events(
            harmonic_rhythm_events, detected_chord, detected_confidence,
            require_successor=audio_observable)
        # Com uma fonte de áudio ativa, o relógio não confirma sozinho uma
        # mudança harmônica. O fallback temporal só serve para chamadas sem
        # observações de áudio. Mesmo acordes repetidos não são prova de troca.
        repeated_chart_chord = (not audio_observable and confident_audio and
                                self._chart_cursor_index + 1 < self._alignment.event_count and
                                self._event_matches(self._chart_cursor_index, detected_chord) and
                                self._event_matches(self._chart_cursor_index + 1, detected_chord))
        if (repeated_chart_chord or
                (not audio_observable and
                 (not confident_audio or not self._has_seen_harmonic_observation or
                  self._tracking_state == TrackingState.LOST))):
            self._consume_temporal_progress(
                temporal_pos, current_chord_elapsed_beats,
                expected_chord_duration_beats, duration_confidence)

        # Registra o acorde detectado (buffer para localização por progressão)
        chord_changed = confident_audio and (not self._recent_detected or self._recent_detected[-1] != detected_chord)
        if chord_changed:
            self._recent_detected.append(detected_chord)
        # O acorde já passou pelo estabilizador. Dois frames espaçados podem
        # confirmar o próximo evento; limitar isto a chord_changed contava só
        # o primeiro frame e deixava o cursor preso até o relógio avançar.
        if (confident_audio and self._follow_audio and not transition_consumed and
                self._capo_domain != "UNRESOLVED"):
            self._consume_from_audio(
                detected_chord, detected_confidence,
                expected_chord_duration_beats, duration_confidence,
                self._clock.total_beats)
        if confident_audio:
            self._has_seen_harmonic_observation = True

        # 1.5. LOCALIZAÇÃO POR ÁUDIO — encaixa a posição onde a progressão recente casa
        # na cifra, reancorando o deslocamento relógio→cifra. É isto que faz o programa
        # SEGUIR o músico em vez de apenas percorrer a cifra em ordem pelo tempo.
        localize_note = ""
        if self._follow_audio and chord_changed:
            tentative_bar = self._cursor_position(
                current_beat, timestamp, max(1, clock_bar - self._bar_offset)).current_bar
            # Busca global é uma recuperação excepcional. Uma progressão longa
            # sem candidato local declara perda real da posição antes de buscar
            # no documento inteiro; a operação normal permanece local e
            # monotônica pelo cursor.
            # A observação já foi comparada uma única vez com o evento atual e
            # o próximo. Reprocessar o mesmo frame aqui contaria duas
            # confirmações e poderia avançar o cursor artificialmente.
            recovery_chords = self._confirmed_recovery_sequence()
            loc = (self._global_localize(
                near_bar=tentative_bar, observed_chords=recovery_chords)
                if (self._tracking_state == TrackingState.LOST
                    and not self._startup_lock
                    and len(recovery_chords) >= 3) else None)
            if self._tracking_state == TrackingState.LOST and not self._startup_lock:
                self._last_search_mode = "GLOBAL"
            if loc is not None:
                loc_bar, loc_score, loc_len = loc
                distance = abs(loc_bar - tentative_bar)
                # Quanto maior o salto, mais forte a evidência exigida (evita saltos falsos).
                # 0.78 no salto longo permite RE-ACHAR a música por uma progressão de
                # tônicas (cada acerto só de raiz vale 0.8), mesmo com qualidades erradas.
                strong_enough = (
                    (distance <= 2 and loc_len >= 2 and loc_score >= 0.75)
                    or (distance > 2 and loc_len >= 3 and loc_score >= 0.78)
                )
                if strong_enough and distance >= 1:
                    target_event = self._alignment.get_position_at(loc_bar).element_index
                    # Nunca volta para "repetir" compassos perdidos: só segue para frente.
                    # A única volta aceita é para o INÍCIO de uma seção (a banda repetindo
                    # o refrão, por exemplo), e com uma progressão forte e longa.
                    first_event = target_event - (loc_len - 1)
                    backward_ok = (loc_len >= 3 and loc_score >= 0.90 and any(
                        first_event <= start <= target_event
                        for start in self._alignment.section_start_events()))
                    if target_event >= self._chart_cursor_index or backward_ok:
                        self._set_chart_cursor(
                            target_event, "global progression recovery", self._clock.total_beats,
                            f"matched_events={loc_len}; confidence={loc_score:.2f}")
                        self._bar_offset = clock_bar - loc_bar
                        self._recent_notes.clear()
                        self._global_recovery_confirmation_pending = True
                        localize_note = (
                            f"Localizado pelo áudio no Comp. {loc_bar} "
                            f"({loc_len} acordes, {int(loc_score * 100)}%)"
                        )

        raw_chart_bar = max(1, clock_bar - self._bar_offset)
        current_bar = (self._cursor_position(current_beat, timestamp, raw_chart_bar).current_bar
                       if self._alignment.event_count else raw_chart_bar)
        note_match = self._locate_note_sequence(timestamp, detected_note, note_confidence, current_bar)
        if note_match is not None:
            # Nota melódica/arpejo é evidência auxiliar. Nunca possui autoridade
            # para alterar cursor, offset, compasso ou seção.
            self._last_note_evidence = {
                "candidate_bar": note_match,
                "near_bar": current_bar,
                "tracking_state": self._tracking_state.value,
                "action": "CONFIDENCE_ONLY",
            }
        if audio_observable and self._follow_audio:
            self._follow_local_notes(timestamp, detected_note, note_confidence,
                                     detected_chord, detected_confidence)

        # Consulta posição nominal na cifra pelo compasso localizado
        nominal_pos = (self._cursor_state(self._cursor_position(current_beat, timestamp, raw_chart_bar))
                       if self._alignment.event_count else
                       self._alignment.get_position_at(current_bar, current_beat, timestamp))
        expected_chord = nominal_pos.current_chord

        # Registra no buffer de histórico
        if expected_chord and expected_chord != "--":
            if not self._recent_expected or self._recent_expected[-1] != expected_chord:
                self._recent_expected.append(expected_chord)

        # Acorde previsto estruturalmente (Prediction fallback). Guardamos a substituição
        # do motor de previsão separada do valor derivado de nominal_pos: se uma recuperação
        # mais adiante reancorar a posição, nominal_pos.next_chord é reconsultado no retorno
        # em vez de usar um valor obsoleto calculado para a posição anterior.
        # BUG corrigido: MusicPrediction expõe a lista ``predicted_chords`` (plural);
        # o atributo singular ``predicted_next_chord`` nunca existiu, então este fallback
        # sempre resultava em None e nunca disparava. Usamos o primeiro item da lista.
        predicted_chord_override = None
        pred = getattr(self._prediction_engine, "latest_prediction", None)
        pred_chords = getattr(pred, "predicted_chords", None) if pred else None
        if pred_chords and pred_chords[0] and pred_chords[0] != "--":
            predicted_chord_override = pred_chords[0]

        # 2. Avaliação Harmônica do Áudio
        is_unknown_audio = not detected_chord or detected_chord in ("--", "UNKNOWN", "N") or detected_confidence < 0.25

        if is_unknown_audio:
            # ----------------------------------------------------
            # CASO A: ÁUDIO DESCONHECIDO / INDETERMINADO
            # Sem harmonia confiável, o relógio segue; a cifra aguarda notas ou
            # acordes que confirmem uma mudança quando há áudio observável.
            # ----------------------------------------------------
            self._consecutive_unknown_seconds += dt
            self._consecutive_mismatch_seconds = 0.0
            if self._consecutive_unknown_seconds > 2.5:
                # Uma sequência interrompida não é evidência para reancorar após a perda.
                self._recent_detected.clear()
                self._recent_harmonic_rhythm.clear()

            # Degradação suave da confiança temporal
            if self._startup_lock and not self._start_anchor_confirmed:
                self._tracking_state = TrackingState.UNCERTAIN
                self._position_confidence = max(0.70, self._position_confidence - (dt * 0.03))
                action_note = "STARTING: aguardando confirmação do primeiro evento"
            elif self._consecutive_unknown_seconds > 8.0:
                self._tracking_state = TrackingState.LOST
                self._position_confidence = max(0.40, self._position_confidence - (dt * 0.03))
                action_note = "Aguardando evidência harmônica (LOST)" if audio_observable else "Avanço temporal (Detecção ausente há > 8s - LOST)"
            elif self._consecutive_unknown_seconds > 2.5:
                self._tracking_state = TrackingState.UNCERTAIN
                self._position_confidence = max(0.65, self._position_confidence - (dt * 0.04))
                action_note = "Aguardando evidência harmônica (UNCERTAIN)" if audio_observable else "Avanço temporal (Detecção incerta - UNCERTAIN)"
            else:
                self._tracking_state = TrackingState.TRACKING
                self._position_confidence = max(0.85, 0.95 - (self._consecutive_unknown_seconds * 0.04))
                action_note = "Aguardando evidência harmônica" if audio_observable else "Avanço temporal contínuo"

        else:
            # Áudio com acorde detectado convincente
            self._consecutive_unknown_seconds = 0.0
            self._last_detected_chord = detected_chord

            # Comparação harmônica com o esperado
            is_match = False
            is_transposed_match = False
            root_only_match = False
            if expected_chord and expected_chord != "--":
                det_sym = parse_chord(detected_chord)

                def _same(a_sym):
                    return a_sym.is_synonym_of(det_sym) or (a_sym.root == det_sym.root and a_sym.quality == det_sym.quality)

                # O áudio pode ser reportado no domínio SOANTE (com capotraste) ou como o
                # SHAPE escrito. Aceitamos qualquer um dos dois para robustez.
                exp_sounding = self._expected_as_sounding(expected_chord)
                is_match = _same(parse_chord(exp_sounding))
                if not is_match and exp_sounding != expected_chord:
                    is_match = _same(parse_chord(expected_chord))

                # Aprende um deslocamento global RESIDUAL (afinação diferente etc.)
                self._transposition.observe(exp_sounding, detected_chord)
                if not is_match and self._transposition.matches_under_transposition(exp_sounding, detected_chord):
                    # A execução está transposta de forma coerente: confirma a posição
                    # (a cifra continua soberana; apenas deixamos de reportar falsa divergência).
                    is_match = True
                    is_transposed_match = True

                # Fallback pela TÔNICA: se a raiz bate (mas a qualidade detectada divergiu),
                # ainda consideramos a posição confirmada — o detector erra a qualidade, não
                # o baixo/tônica. Evita que a recuperação brigue com a localização por áudio.
                if not is_match and self._roots_equal(parse_chord(exp_sounding), det_sym):
                    is_match = True
                    root_only_match = True

            if is_match:
                # ----------------------------------------------------
                # CASO B: CONCORDÂNCIA PLENA (ÁUDIO CONFIRMA A CIFRA)
                # ----------------------------------------------------
                # A observação que produziu uma relocalização global escolhe o
                # candidato; uma observação posterior ainda precisa confirmá-lo.
                self._tracking_state = (TrackingState.RECOVERING
                                        if localize_note else TrackingState.TRACKING)
                # Uma confirmação harmônica forte deve restaurar a confiança de imediato
                # (ponderada pela confiança do áudio), em vez de subir apenas +0.08 por frame.
                # Assim o sistema recupera rapidamente após uma perda prolongada (LOST).
                recovery_target = 0.60 + 0.35 * max(0.0, min(1.0, detected_confidence))
                self._position_confidence = min(
                    0.98,
                    max(self._position_confidence + 0.08, recovery_target)
                )
                self._last_confirmed_bar = current_bar
                self._consecutive_mismatch_seconds = 0.0
                if is_transposed_match:
                    action_note = f"Confirmada com transposição de {self._transposition.semitones:+d} semitons (capotraste/tom)"
                elif root_only_match:
                    action_note = f"Confirmada pela tônica ({detected_chord} ~ {expected_chord})"
                else:
                    action_note = "Posição harmônica confirmada"

            else:
                # ----------------------------------------------------
                # CASO C: DIVERGÊNCIA (ÁUDIO DETECTOU ACORDE DIFERENTE)
                # ----------------------------------------------------
                self._consecutive_mismatch_seconds += dt

                # Se a divergência for muito recente (< 2s), pode ser variação de performance ou antecipação
                if self._consecutive_mismatch_seconds < 2.0:
                    action_note = f"Variação momentânea ({detected_chord} vs {expected_chord})"
                    self._position_confidence = max(0.70, self._position_confidence - (dt * 0.05))
                else:
                    self._tracking_state = TrackingState.UNCERTAIN
                    self._position_confidence = max(0.50, self._position_confidence - (dt * 0.08))
                    confirmed_events = [item[1] for item in self._recent_harmonic_rhythm
                                        if getattr(item[1], "confidence", 0.0) >= 0.65]
                    if self._consecutive_mismatch_seconds >= 6.0 and len(confirmed_events) >= 3:
                        self._tracking_state = TrackingState.LOST
                        action_note = "Posição perdida após divergência harmônica confirmada"

                    # ------------------------------------------------
                    # RECUPERAÇÃO INTELIGENTE POR JANELA LOCAL
                    # ------------------------------------------------
                    local_match = self._probe_local_search_window(
                        center_bar=current_bar,
                        detected_chord=detected_chord,
                        window_bars=self._local_window
                    )

                    if local_match:
                        matched_bar, matched_chord = local_match
                        # Só para frente: perdeu um compasso, alcança a música; não volta.
                        bar_diff = matched_bar - current_bar
                        if 0 <= bar_diff <= 2:
                            self._tracking_state = TrackingState.RECOVERING
                            action_note = f"Recuperado na janela local (Comp. {matched_bar} - {matched_chord})"
                            self._position_confidence = min(0.88, self._position_confidence + 0.10)
                            # Se for uma pequena diferença de 1 compasso, atualiza o compasso
                            current_bar = matched_bar
                            self._set_chart_cursor(self._alignment.get_position_at(current_bar).element_index)
                            nominal_pos = self._cursor_state(self._cursor_position(
                                current_beat, timestamp, current_bar))
                        else:
                            action_note = f"Candidato local descartado (Comp. {matched_bar} fora da tolerância)"
                    else:
                        # RECUPERAÇÃO POR PROGRESSÃO: um único acorde diverge, mas a
                        # SEQUÊNCIA recente de acordes ainda casa com um trecho da cifra.
                        # É mais robusto que o casamento de acorde isolado.
                        prog = self.find_progression_match(
                            list(self._recent_detected),
                            start_bar_hint=current_bar,
                            window_bars=self._local_window * 2
                        )
                        if prog and prog.confidence >= 0.75 and 0 <= (prog.end_bar - current_bar) <= 3:
                            self._tracking_state = TrackingState.RECOVERING
                            current_bar = prog.end_bar
                            self._set_chart_cursor(self._alignment.get_position_at(current_bar).element_index)
                            nominal_pos = self._cursor_state(self._cursor_position(
                                current_beat, timestamp, current_bar))
                            self._position_confidence = min(0.90, self._position_confidence + 0.15)
                            action_note = (
                                f"Reancorado por progressão "
                                f"({' → '.join(prog.matched_chords)}) no Comp. {current_bar}"
                            )
                        else:
                            action_note = "Avançando temporalmente (Divergência harmônica sustentada)"

        # Se a localização por áudio reancorou a posição, esse é o evento mais relevante
        if localize_note:
            action_note = localize_note
            if self._tracking_state == TrackingState.LOST:
                self._tracking_state = TrackingState.RECOVERING

        # Todas as recuperações, inclusive as locais, persistem na mesma transformação.
        current_bar = nominal_pos.current_bar
        # Uma leitura sustentada não altera a transformação relógio→cifra.
        # Corrigimos o offset apenas quando houve transição harmônica confirmada
        # ou reancoragem real; o fallback temporal já usa o relógio existente.
        if (self._chart_cursor_index != cursor_before and
                self._last_position_change_reason != "temporal fallback"):
            self._bar_offset = clock_bar - current_bar
        return self._consolidate_position(
            nominal_pos, detected_chord if not is_unknown_audio else "--",
            detected_confidence, action_note,
            predicted_chord_override or nominal_pos.next_chord,
        )

    @staticmethod
    def _chords_equal(a_sym, b_sym) -> bool:
        """Igualdade harmônica estrita (sinônimos ou mesma tônica + qualidade)."""
        return a_sym.is_synonym_of(b_sym) or (a_sym.root == b_sym.root and a_sym.quality == b_sym.quality)

    @staticmethod
    def _roots_equal(a_sym, b_sym) -> bool:
        """Igualdade apenas pela TÔNICA (raiz), tolerante a erro de qualidade na detecção.

        O detector de áudio às vezes erra a qualidade (maior/menor/7ª), mas acerta a
        tônica. Casar pela raiz torna a localização muito mais robusta na prática.
        """
        pa, pb = note_to_pc(a_sym.root), note_to_pc(b_sym.root)
        return pa >= 0 and pa == pb

    def _match_score(self, a_sym, b_sym) -> float:
        """Pontuação de casamento: 1.0 exato, 0.8 só pela tônica, 0.0 sem relação."""
        if self._chords_equal(a_sym, b_sym):
            return 1.0
        if self._roots_equal(a_sym, b_sym):
            return 0.8
        return 0.0

    def _chart_chord_sequence(self):
        """Sequência (bar, símbolo, parse) dos acordes da cifra no domínio SOANTE (capo).

        Preserva cada evento expandido, inclusive acordes consecutivos iguais e
        repetições declaradas. A duração pertence ao ritmo harmônico; a ordem
        pertence à cifra e não pode ser colapsada durante uma relocalização.
        """
        seq = []
        for event_index in range(self._alignment.event_count):
            pos = self._alignment.get_position_for_event(event_index)
            ch = pos.current_chord
            if ch and ch != "--":
                seq.append((pos.current_bar, ch, parse_chord(self._expected_as_sounding(ch)),
                            pos.section_name, pos.section_event_index))
        return seq

    @staticmethod
    def _note_score(note_pc: int, chord: str) -> float:
        """Compatibilidade da nota com o acorde; notas de passagem não são prova de erro."""
        symbol = parse_chord(chord)
        root_pc = note_to_pc(symbol.root)
        if root_pc < 0:
            return 0.1
        if note_pc == root_pc:
            return 1.0
        quality = symbol.quality
        tones = {note_to_pc(n) for n in get_chord_notes(PITCH_CLASSES[root_pc], quality)}
        if symbol.extension and "7" in symbol.extension:
            tones.add((root_pc + (11 if "maj7" in symbol.extension.lower() else 10)) % 12)
        if note_pc in tones:
            return 0.88
        return 0.15

    def _follow_local_notes(self, timestamp: float, note: str, confidence: float,
                            detected_chord: str, detected_confidence: float) -> None:
        """Confirma o evento atual ou o próximo com uma frase de notas distinta.

        Notas isoladas, tons compartilhados e candidatos distantes não movem o
        cursor. O acorde detectado, quando seguro, continua tendo prioridade.
        """
        if confidence < 0.7 or note_to_pc(note) < 0 or len(self._recent_notes) < 3:
            return
        notes = [pc for pc, _ in list(self._recent_notes)[-3:]]
        if len(set(notes)) < 3:
            return
        current = self._expected_as_sounding(
            self._alignment.get_position_for_event(self._chart_cursor_index).current_chord)
        current_scores = [self._note_score(pc, current) for pc in notes]
        current_mean = sum(current_scores) / len(notes)
        if current_mean >= 0.84:
            anchor = self._alignment.start_anchor
            if anchor is not None and self._chart_cursor_index == anchor.event_index:
                self._start_anchor_confirmed = True
                self._startup_lock = False
            self._last_note_evidence = {
                "event_index": self._chart_cursor_index,
                "score": round(current_mean, 3), "action": "CONFIRMED_CURRENT",
            }
            return
        next_index = self._chart_cursor_index + 1
        if next_index >= self._alignment.event_count:
            return
        if (detected_chord not in ("", "--", "UNKNOWN", "N") and
                detected_confidence >= 0.7 and
                self._event_matches(self._chart_cursor_index, detected_chord)):
            return
        next_chord = self._expected_as_sounding(
            self._alignment.get_position_for_event(next_index).current_chord)
        next_scores = [self._note_score(pc, next_chord) for pc in notes]
        next_mean = sum(next_scores) / len(notes)
        distinctive = sum(1 for now, nxt in zip(current_scores, next_scores)
                          if now <= 0.4 and nxt >= 0.8)
        next_root = note_to_pc(parse_chord(next_chord).root)
        # Acordes relativos podem compartilhar duas das três notas. Uma
        # tríade tocada a partir da nova fundamental também é evidência forte,
        # desde que a nota exclusiva abra a frase (não uma nota de passagem).
        root_led_triad = (distinctive >= 1 and notes[0] == next_root and
                          all(score >= 0.85 for score in next_scores) and
                          current_mean <= 0.70 and
                          self._event_elapsed_beats() >= 0.5)
        clear_next = (distinctive >= 2 and next_mean >= 0.82 and
                      next_mean - current_mean >= 0.25)
        if (not (clear_next or root_led_triad) or
                (self._startup_lock and not self._start_anchor_confirmed and
                 self._clock.total_beats < 2.0)):
            return
        previous = self._chart_cursor_index
        self._set_chart_cursor(next_index, "confirmed local note sequence",
                               evidence=f"notes={notes}; score={next_mean:.2f}")
        self._recent_notes.clear()
        self._last_note_evidence = {
            "from_event": previous, "event_index": next_index,
            "score": round(next_mean, 3), "action": "ADVANCED_LOCAL",
        }

    def _locate_note_sequence(self, timestamp: float, note: str, confidence: float,
                              near_bar: int) -> Optional[int]:
        """Compara notas distintas com candidatos em toda a cifra, sem saltar em empates."""
        pc = note_to_pc(note) if confidence >= 0.7 else -1
        if pc < 0:
            return None
        if self._recent_notes and timestamp - self._recent_notes[-1][1] > 3.0:
            self._recent_notes.clear()
        if self._recent_notes and (pc == self._recent_notes[-1][0]
                                   or timestamp - self._recent_notes[-1][1] < 0.08):
            return None
        self._recent_notes.append((pc, timestamp))
        if len(self._recent_notes) < 3:
            return None
        notes = [item[0] for item in list(self._recent_notes)[-4:]]
        bars = [self._expected_as_sounding(self._alignment.get_position_at(b).current_chord)
                for b in range(1, self._alignment.total_bars + 1)]
        if not bars:
            return None
        # Cada nota pode pertencer ao mesmo acorde ou ao compasso seguinte.
        # O melhor caminho é calculado para cada possível compasso final.
        scores = [self._note_score(notes[0], chord) for chord in bars]
        for note_pc in notes[1:]:
            scores = [self._note_score(note_pc, chord) +
                      max(scores[i], scores[i - 1] if i else -1.0)
                      for i, chord in enumerate(bars)]
        ranked = sorted(enumerate((s / len(notes) for s in scores), 1),
                        key=lambda item: (-item[1], abs(item[0] - near_bar)))
        best_bar, best_score = ranked[0]
        if best_score < 0.82 or abs(best_bar - near_bar) <= 2:
            return None
        # Empates em refrões/versos repetidos não autorizam salto remoto.
        alternatives = [score for bar, score in ranked[1:] if abs(bar - best_bar) > 2]
        if alternatives and best_score - max(alternatives) < 0.12:
            return None
        current_score = scores[min(near_bar, len(scores)) - 1] / len(notes)
        if best_score - current_score < 0.18:
            return None
        return best_bar

    def _confirmed_recovery_sequence(self) -> List[str]:
        """Acordes fechados e confiáveis que podem sustentar uma recuperação global."""
        events = [entry[1] for entry in self._recent_harmonic_rhythm
                  if getattr(entry[1], "confidence", 0.0) >= 0.65
                  and getattr(entry[1], "raw_duration_beats", 0.0) > 0.0
                  and ChartSemanticClassifier.is_chord_shaped(
                      getattr(entry[1], "symbol", ""))]
        return [event.symbol for event in events[-4:]]

    def _global_localize(self, near_bar: int = 1,
                         observed_chords: Optional[List[str]] = None):
        """Procura NO DOCUMENTO INTEIRO onde a progressão detectada recentemente encaixa.

        Retorna (compasso_atual, score, qtd_acordes) do melhor encaixe, preferindo o mais
        próximo de ``near_bar`` em caso de empate (evita saltar para um refrão idêntico
        distante). É o coração do "seguir o áudio": o programa acha ONDE o músico está.
        """
        source = observed_chords if observed_chords is not None else self._recent_detected
        recent = [c for c in source if c and c not in ("--", "UNKNOWN", "N")]
        if len(recent) < 2:
            return None
        recent = recent[-4:]  # os últimos acordes distintos bastam para localizar
        recent_syms = [parse_chord(c) for c in recent]
        L = len(recent_syms)

        seq = self._chart_chord_sequence()
        if len(seq) < L:
            return None

        best = None  # ((score, -distância), end_bar, score, L)
        for i in range(0, len(seq) - L + 1):
            window = seq[i:i + L]
            # Pontuação tolerante à tônica (1.0 exato, 0.8 só pela raiz)
            score_sum = sum(self._match_score(window[j][2], recent_syms[j]) for j in range(L))
            ratio = score_sum / L
            duration_score = None
            # Duração é uma segunda evidência independente, não substitui os
            # acordes. Só entra após um padrão já ter sido observado na seção.
            if self._structure_analyzer is not None and self._recent_harmonic_rhythm:
                observed = [entry[1] for entry in self._recent_harmonic_rhythm]
                count = min(len(observed), L)
                first = window[:count]
                expected_durations = []
                for candidate in first:
                    pattern = self._structure_analyzer.pattern_memory.find_harmonic_rhythm(
                        section_key(candidate[3]), candidate[1], candidate[4])
                    if pattern is None or pattern.confidence < .55 or candidate[4] >= len(pattern.duration_sequence):
                        expected_durations = []
                        break
                    expected_durations.append(pattern.duration_sequence[candidate[4]])
                if expected_durations:
                    scores = [max(0.0, 1.0 - abs(obs.quantized_duration_beats - exp) / max(1.0, exp))
                              for obs, exp in zip(observed[-count:], expected_durations)]
                    duration_score = sum(scores) / len(scores)
            combined = ratio if duration_score is None else (0.65 * ratio + 0.35 * duration_score)
            if combined >= 0.75:
                end_bar = window[-1][0]
                dist = abs(end_bar - near_bar)
                key = (combined, -dist)
                if best is None or key > best[0]:
                    best = (key, end_bar, combined, L)
        if best is None:
            return None
        return best[1], best[2], best[3]

    def localize(self, near_bar: int = 1):
        """API pública: melhor encaixe da progressão detectada na cifra (ou None)."""
        return self._global_localize(near_bar=near_bar)

    def _probe_local_search_window(
        self,
        center_bar: int,
        detected_chord: str,
        window_bars: int = 4
    ) -> Optional[Tuple[int, str]]:
        """Pesquisa um acorde detectado exclusivamente dentro de uma janela local de compassos.
        
        NUNCA pesquisa no documento inteiro para evitar saltos falsos para outras seções.
        """
        det_sym = parse_chord(detected_chord)
        start_b = max(1, center_bar - window_bars)
        end_b = min(self._alignment.total_bars, center_bar + window_bars)

        # Escolhe o candidato mais próximo, preferindo o futuro em empate.
        matches = []
        for b in range(start_b, end_b + 1):
            pos = self._alignment.get_position_at(b)
            if pos.current_chord and pos.current_chord != "--":
                cand_sym = parse_chord(pos.current_chord)
                if cand_sym.is_synonym_of(det_sym) or (cand_sym.root == det_sym.root and cand_sym.quality == det_sym.quality):
                    matches.append((b, pos.current_chord))

        return (min(matches, key=lambda item: (abs(item[0] - center_bar),
                                              item[0] < center_bar)) if matches else None)

    def find_progression_match(
        self,
        recent_chords: List[str],
        start_bar_hint: int = 1,
        window_bars: int = 8
    ) -> Optional[ProgressionMatch]:
        """Verifica se uma sequência de múltiplos acordes detectados coincide com a progressão da cifra."""
        if len(recent_chords) < 2:
            return None

        clean_recent = [parse_chord(c) for c in recent_chords if c and c not in ("--", "UNKNOWN")]
        if len(clean_recent) < 2:
            return None

        start_b = max(1, start_bar_hint - window_bars)
        end_b = min(self._alignment.total_bars, start_bar_hint + window_bars)

        # Extrai sequência esperada na vizinhança
        expected_seq = []
        for b in range(start_b, end_b + 1):
            pos = self._alignment.get_position_at(b)
            if pos.current_chord and pos.current_chord != "--":
                if not expected_seq or expected_seq[-1][1] != pos.current_chord:
                    expected_seq.append((b, pos.current_chord, parse_chord(pos.current_chord)))

        # Compara subsequências
        req_len = min(len(clean_recent), len(expected_seq))
        if req_len < 2:
            return None

        recent_tails = clean_recent[-req_len:]
        for i in range(len(expected_seq) - req_len + 1):
            window = expected_seq[i:i + req_len]
            matches = sum(
                1 for j in range(req_len)
                if window[j][2].is_synonym_of(recent_tails[j]) or (window[j][2].root == recent_tails[j].root and window[j][2].quality == recent_tails[j].quality)
            )
            match_ratio = matches / req_len
            if match_ratio >= 0.75:
                return ProgressionMatch(
                    matched_chords=[w[1] for w in window],
                    expected_chords=[c.original_symbol for c in recent_tails],
                    start_bar=window[0][0],
                    end_bar=window[-1][0],
                    confidence=match_ratio
                )

        return None
