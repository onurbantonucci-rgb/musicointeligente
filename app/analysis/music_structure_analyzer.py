"""Analisador de Estrutura Musical e Seções (MusicStructureAnalyzer v0.3).

Executa:
1. Segmentação da linha do tempo em frases e compassos a partir do MusicalClock e ChordHistory.
2. Reconhecimento de repetições harmônicas e padrões recorrentes via PatternMemory.
3. Classificação probabilística de seções (INTRO, VERSE, PRE_CHORUS, CHORUS, BRIDGE, OUTRO, UNKNOWN).
4. Mapeamento da forma concreta e abstrata da composição (ex: A - A - B - A - B - C - B).
5. Suporte duplo:
   - ONLINE: Atualização contínua e incremental durante reprodução ao vivo sem latência.
   - OFFLINE: Análise estrutural completa da faixa inteira para planejamento prévio.
"""

from typing import List, Optional, Tuple, Any
import numpy as np

from app.music.music_structure import (
    MusicSection,
    MusicalPattern,
    MusicStructure,
)
from app.music.harmonic_normalization import normalize_chord_sequence
from app.music.pattern_memory import PatternMemory
from app.music.prediction_engine import PredictionEngine, MusicPrediction
from app.analysis.music_position_estimator import MusicPositionEstimator
from app.analysis.chord_history import ChordHistory
from app.analysis.key_history import KeyHistory
from app.music.musical_clock import MusicalClock
from app.music.musical_context import MusicalContext
from app.music.chart_alignment import ChartPosition


class MusicStructureAnalyzer:
    """Analisador central de forma, frases, seções e repetições musicais."""

    def __init__(self):
        self._pattern_memory = PatternMemory()
        self._position_estimator = MusicPositionEstimator()
        self._prediction_engine = PredictionEngine()

        self._structure = MusicStructure()
        self._sections: List[MusicSection] = []
        self._section_counter: int = 0
        self._active_section: Optional[MusicSection] = None
        self._active_pattern: Optional[MusicalPattern] = None

        # Rastreamento incremental de compassos processados
        self._last_processed_bar: int = 0
        self._last_bar_offset: Optional[int] = None
        self._current_phrase_chords: List[str] = []
        self._phrase_start_bar: int = 1
        self._phrase_start_time: float = 0.0

    @property
    def pattern_memory(self) -> PatternMemory:
        return self._pattern_memory

    @property
    def position_estimator(self) -> MusicPositionEstimator:
        return self._position_estimator

    @property
    def prediction_engine(self) -> PredictionEngine:
        return self._prediction_engine

    @property
    def current_structure(self) -> MusicStructure:
        return self._structure

    @property
    def active_section(self) -> Optional[MusicSection]:
        return self._active_section

    @property
    def active_pattern(self) -> Optional[MusicalPattern]:
        return self._active_pattern

    @property
    def sections(self) -> List[MusicSection]:
        """Retorna lista das seções detectadas até agora."""
        return list(self._sections)

    @property
    def structure(self) -> MusicStructure:
        """Alias para current_structure."""
        return self._structure

    def reset(self) -> None:
        """Reinicia todos os motores de aprendizado estrutural para uma nova faixa."""
        self._pattern_memory.reset()
        self._position_estimator.reset()
        self._prediction_engine.reset()
        self._structure = MusicStructure()
        self._sections.clear()
        self._section_counter = 0
        self._active_section = None
        self._active_pattern = None
        self._last_processed_bar = 0
        self._last_bar_offset = None
        self._current_phrase_chords.clear()
        self._phrase_start_bar = 1
        self._phrase_start_time = 0.0

    # -------------------------------------------------------------
    # 1. Classificação Probabilística de Seção
    # -------------------------------------------------------------
    def _classify_section(
        self,
        start_bar: int,
        duration_bars: int,
        occurrence_count: int,
        total_song_bars: int = 64,
        is_contrasting: bool = False,
        precedes_high_rep: bool = False
    ) -> Tuple[str, float]:
        """Avalia características observáveis para estimar o tipo de seção (probabilístico).
        
        Se não houver evidências consistentes, retorna ('UNKNOWN', conf moderada) sem inventar.
        """
        # Seção com duração mínima insuficiente para ser verso/seção estrutural
        if duration_bars < 2:
            return "UNKNOWN", 0.45

        # Início da composição
        if start_bar <= 4 and occurrence_count <= 1:
            return "INTRO", 0.88
        
        # Pré-refrão: seção curta (geralmente 2 a 4 compassos) que prepara um padrão recorrente
        if precedes_high_rep and duration_bars in (2, 4):
            return "PRE_CHORUS", 0.78

        # Refrão: padrão com alta frequência de repetição e retorno
        if occurrence_count >= 3:
            return "CHORUS", min(0.95, 0.75 + (occurrence_count * 0.05))

        # Verso: padrão recorrente fundamental (ocorrências 1 e 2)
        if occurrence_count in (1, 2) and not is_contrasting:
            # Se for bem no início (após intro), forte tendência de Verso
            if start_bar <= 16:
                return "VERSE", 0.85
            return "VERSE", 0.76

        # Ponte (Bridge): quebra contrastante após versos e refrãos conhecidos
        if is_contrasting and start_bar > 20 and start_bar < (total_song_bars - 12):
            return "BRIDGE", 0.80

        # Outro: final da música
        if total_song_bars > 24 and start_bar >= (total_song_bars - 8):
            return "OUTRO", 0.82

        # Sem evidência conclusiva
        return "UNKNOWN", 0.50


    # -------------------------------------------------------------
    # 2. Atualização Online Incremental (Em tempo de reprodução)
    # -------------------------------------------------------------
    def update_online(
        self,
        context: MusicalContext,
        chord_history: ChordHistory,
        key_history: KeyHistory,
        clock: MusicalClock,
        chart_position: Optional[ChartPosition] = None,
    ) -> MusicStructure:
        """Executa atualização incremental leve em tempo real (chamada em cada ciclo da UI)."""
        current_bar = max(1, context.bar)
        timestamp = context.timestamp

        # Acumula acorde vigente na frase se o compasso avançou
        offset_changed = self._last_bar_offset is not None and context.bar_offset != self._last_bar_offset
        self._last_bar_offset = context.bar_offset
        if current_bar != self._last_processed_bar or offset_changed:
            if offset_changed or self._last_processed_bar == 0 or current_bar != self._last_processed_bar + 1:
                # Não misture frases separadas por reancoragem ou seek.
                self._current_phrase_chords.clear()
                self._phrase_start_bar = current_bar
                self._phrase_start_time = timestamp
                self._active_section = None
                self._active_pattern = None
            self._last_processed_bar = current_bar
            curr_chord = context.chord if context.chord != "--" else "C"
            self._current_phrase_chords.append(curr_chord)

            # A cada 4 compassos (ou múltiplo de frase comum), avalia e registra o padrão na memória
            phrase_len = len(self._current_phrase_chords)
            if phrase_len >= 4:
                key_str = context.key if context.key != "--" else "C Major"
                normalized_seq = normalize_chord_sequence(
                    chords=self._current_phrase_chords,
                    key_str=key_str,
                    confidence=context.chord_confidence if context.chord_confidence > 0 else 0.80
                )

                start_b = self._phrase_start_bar
                end_b = current_bar
                dur_bars = max(1, end_b - start_b + 1)

                # Registra na memória de padrões
                pat, is_new, occ = self._pattern_memory.register_pattern(
                    normalized_seq=normalized_seq,
                    start_bar=start_b,
                    end_bar=end_b,
                    start_time=self._phrase_start_time,
                    end_time=timestamp,
                    key=key_str
                )
                self._active_pattern = pat

                # Classifica seção
                sec_type, sec_conf = self._classify_section(
                    start_bar=start_b,
                    duration_bars=dur_bars,
                    occurrence_count=pat.occurrence_count
                )
                pat.associated_section_type = sec_type

                # Registra transição de seção se houve mudança
                old_sec_type = self._active_section.section_type if self._active_section else "UNKNOWN"
                old_pat_id = self._active_pattern.id if self._active_pattern else "--"
                if old_sec_type != sec_type or old_pat_id != pat.id:
                    self._pattern_memory.record_transition(
                        from_pattern_id=old_pat_id,
                        to_pattern_id=pat.id,
                        from_section=old_sec_type,
                        to_section=sec_type,
                        duration=timestamp - self._phrase_start_time
                    )

                # Cria nova seção no histórico de seções
                self._section_counter += 1
                sec_id = f"sec_{self._section_counter:02d}"
                label_char = chr(ord('A') + (int(pat.id[1:]) - 1) % 26)

                new_section = MusicSection(
                    id=sec_id,
                    section_type=sec_type,
                    start_time=self._phrase_start_time,
                    end_time=timestamp,
                    start_bar=start_b,
                    end_bar=end_b,
                    duration_bars=dur_bars,
                    chord_sequence=list(self._current_phrase_chords),
                    pattern_id=pat.id,
                    confidence=sec_conf,
                    occurrence_index=pat.occurrence_count,
                    occurrences=pat.occurrence_count,
                    label=label_char
                )
                self._sections.append(new_section)
                self._active_section = new_section

                # Prepara próxima frase
                self._current_phrase_chords = []
                self._phrase_start_bar = current_bar + 1
                self._phrase_start_time = timestamp

        # Estima posição atual na música
        pos = self._position_estimator.estimate_position(
            timestamp=timestamp,
            clock=clock,
            current_section=self._active_section,
            current_pattern=self._active_pattern,
            occurrence_index=self._active_pattern.occurrence_count if self._active_pattern else 1,
            musical_context=context,
            chart_position=chart_position,
        )

        # Produz predição antecipada
        pred = self._prediction_engine.predict_next(
            current_position=pos,
            current_pattern=self._active_pattern,
            pattern_memory=self._pattern_memory,
            current_section_type=self._active_section.section_type if self._active_section else "UNKNOWN",
            harmonic_rhythm=context
        )
        if pred is None:
            pred = MusicPrediction()

        # Monta sequências globais de estrutura
        struct_seq = [s.section_type for s in self._sections]
        abstract_seq = [s.label for s in self._sections]

        self._structure = MusicStructure(
            sections=list(self._sections),
            patterns=dict(self._pattern_memory._patterns),
            transitions=self._pattern_memory.get_transitions(),
            current_position=pos,
            structure_sequence=struct_seq,
            abstract_sequence=abstract_seq,
            analysis_confidence=round(float(np.mean([s.confidence for s in self._sections])) if self._sections else 0.50, 2)
        )

        # Atualiza campos preditivos no MusicalContext
        context.current_section = pos.current_section
        context.current_pattern = pos.pattern_id
        context.section_progress = pos.section_progress
        context.structure_confidence = pos.confidence
        context.predicted_next_section = pred.predicted_section
        context.predicted_next_chords = list(pred.predicted_chords)
        context.bars_until_change = pred.bars_until_change
        context.beats_until_change = pred.beats_until_change
        context.prediction_confidence = pred.confidence
        context.prediction_reason = pred.prediction_reason

        return self._structure

    # -------------------------------------------------------------
    # 3. Análise Estrutural Offline (Música Inteira)
    # -------------------------------------------------------------
    def analyze_full_song_from_events(
        self,
        chord_events: Any,
        total_duration: float = 0.0,
        bpm: float = 120.0,
        key_str: str = "C Major"
    ) -> MusicStructure:
        """Processa a progressão completa de acordes gravada para mapear toda a composição."""
        self.reset()
        if not chord_events:
            return self._structure

        bpm_val = bpm if bpm > 0 else 120.0
        seconds_per_beat = 60.0 / bpm_val
        seconds_per_bar = seconds_per_beat * 4.0

        # Normaliza eventos para formato [(start_time, end_time, chord)]
        norm_events: List[Tuple[float, float, str]] = []
        for i, ev in enumerate(chord_events):
            if hasattr(ev, 'start_time') and hasattr(ev, 'end_time') and hasattr(ev, 'chord'):
                norm_events.append((ev.start_time, ev.end_time, ev.chord))
            elif isinstance(ev, (tuple, list)):
                t_start = float(ev[0])
                ch = str(ev[1])
                t_next = float(chord_events[i+1][0]) if (i + 1) < len(chord_events) else (t_start + seconds_per_bar)
                norm_events.append((t_start, t_next, ch))

        if total_duration <= 0.0 and norm_events:
            total_duration = max(ev[1] for ev in norm_events)

        total_estimated_bars = max(1, int(round(total_duration / seconds_per_bar)))

        # 1. Agrupar eventos de acordes em blocos de compassos (frases de 4 compassos)
        bar_chords: List[str] = []
        for b in range(1, total_estimated_bars + 1):
            t_bar = (b - 1) * seconds_per_bar
            matching = [ch for t_start, t_end, ch in norm_events if t_start <= (t_bar + seconds_per_bar * 0.5) <= t_end]
            if matching:
                bar_chords.append(matching[0])
            elif norm_events:
                closest = min(norm_events, key=lambda e: abs(e[0] - t_bar))
                bar_chords.append(closest[2])
            else:
                bar_chords.append("C")


        # 2. Segmentar em frases típicas de 4 compassos
        phrase_size = 4
        prev_pat_id = None
        prev_sec_type = None

        for idx in range(0, len(bar_chords), phrase_size):
            chunk = bar_chords[idx:idx + phrase_size]
            if not chunk:
                continue

            start_b = idx + 1
            end_b = idx + len(chunk)
            start_t = (start_b - 1) * seconds_per_bar
            end_t = min(total_duration, end_b * seconds_per_bar)
            dur_bars = len(chunk)

            norm_seq = normalize_chord_sequence(chunk, key_str=key_str, confidence=0.90)

            pat, is_new, occ = self._pattern_memory.register_pattern(
                normalized_seq=norm_seq,
                start_bar=start_b,
                end_bar=end_b,
                start_time=start_t,
                end_time=end_t,
                key=key_str
            )

            # Classifica o tipo de seção
            sec_type, sec_conf = self._classify_section(
                start_bar=start_b,
                duration_bars=dur_bars,
                occurrence_count=pat.occurrence_count,
                total_song_bars=total_estimated_bars
            )
            pat.associated_section_type = sec_type

            # Registra transição
            if prev_pat_id is not None and prev_sec_type is not None:
                self._pattern_memory.record_transition(
                    from_pattern_id=prev_pat_id,
                    to_pattern_id=pat.id,
                    from_section=prev_sec_type,
                    to_section=sec_type,
                    duration=end_t - start_t
                )
            prev_pat_id = pat.id
            prev_sec_type = sec_type

            self._section_counter += 1
            sec_id = f"sec_{self._section_counter:02d}"
            label_char = chr(ord('A') + (int(pat.id[1:]) - 1) % 26)

            new_section = MusicSection(
                id=sec_id,
                section_type=sec_type,
                start_time=start_t,
                end_time=end_t,
                start_bar=start_b,
                end_bar=end_b,
                duration_bars=dur_bars,
                chord_sequence=list(chunk),
                pattern_id=pat.id,
                confidence=sec_conf,
                occurrence_index=pat.occurrence_count,
                occurrences=pat.occurrence_count,
                label=label_char
            )
            self._sections.append(new_section)

        # Monta a estrutura final
        struct_seq = [s.section_type for s in self._sections]
        abstract_seq = [s.label for s in self._sections]

        self._structure = MusicStructure(
            sections=list(self._sections),
            patterns=dict(self._pattern_memory._patterns),
            transitions=self._pattern_memory.get_transitions(),
            structure_sequence=struct_seq,
            abstract_sequence=abstract_seq,
            analysis_confidence=round(float(np.mean([s.confidence for s in self._sections])) if self._sections else 0.50, 2)
        )

        return self._structure
