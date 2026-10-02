"""Alinhamento entre Tempo Musical e Posição na Cifra (ChartAlignment v0.4).

Permite navegar, projetar e correlacionar a posição contínua do MusicalClock
com os compassos, seções, acordes, letras e linhas físicas do ChordChart.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from app.music.chord_chart import ChordChart, ChartSection, ChartChord
from app.music.musical_clock import MusicalClock
from app.input.chart_semantic_classifier import ChartSemanticClassifier


@dataclass
class ChartPosition:
    """Posição musical consolidada: compasso, beat fracionário, linha e seção.

    Os consumidores usam este snapshot. O relógio bruto permanece separado.
    """
    current_bar: int = 1
    current_beat: float = 1.0
    section_id: str = "--"
    section_name: str = "--"
    section_type: str = "UNKNOWN"
    section_index: int = 0
    current_chord: str = "--"
    next_chord: str = "--"
    next_section_name: str = "--"
    chord_index_in_section: int = 0
    chord_index: int = 0
    element_index: int = 0
    # Índices explícitos do evento na timeline expandida. ``element_index`` é
    # mantido por compatibilidade; os novos nomes tornam o contrato de cursor
    # consumível pelos seguidores musicais mais claro.
    event_index: int = 0
    next_event_index: int = -1
    section_event_index: int = 0
    section_event_count: int = 0
    section_occurrence: int = 1
    event_state: str = "ACTIVE"       # PENDING, ACTIVE ou CONSUMED
    section_state: str = "ACTIVE"     # PENDING, ACTIVE ou CONSUMED
    next_event_state: str = "PENDING"
    line_index: int = 1                 # Número da linha 1-indexada no texto original da cifra
    section_progress: float = 0.0
    bars_until_chord_change: int = 0
    bars_until_section_change: int = 0
    current_lyric: str = ""
    is_last_chord: bool = False
    absolute_time: float = 0.0
    confidence: float = 0.90
    tracking_state: str = "TRACKING"    # "TRACKING", "UNCERTAIN", "LOST", "RECOVERING"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_bar": self.current_bar,
            "current_beat": round(self.current_beat, 2),
            "section_id": self.section_id,
            "section_name": self.section_name,
            "section_type": self.section_type,
            "section_index": self.section_index,
            "current_chord": self.current_chord,
            "next_chord": self.next_chord,
            "next_section_name": self.next_section_name,
            "chord_index_in_section": self.chord_index_in_section,
            "chord_index": self.chord_index,
            "element_index": self.element_index,
            "event_index": self.event_index,
            "next_event_index": self.next_event_index,
            "section_event_index": self.section_event_index,
            "section_event_count": self.section_event_count,
            "section_occurrence": self.section_occurrence,
            "event_state": self.event_state,
            "section_state": self.section_state,
            "next_event_state": self.next_event_state,
            "line_index": self.line_index,
            "section_progress": round(self.section_progress, 3),
            "bars_until_chord_change": self.bars_until_chord_change,
            "bars_until_section_change": self.bars_until_section_change,
            "current_lyric": self.current_lyric,
            "is_last_chord": self.is_last_chord,
            "absolute_time": round(self.absolute_time, 3),
            "confidence": round(self.confidence, 2),
            "tracking_state": self.tracking_state,
        }


@dataclass(frozen=True)
class StartAnchor:
    """Primeiro evento musical tocável da primeira seção não vazia."""

    section_index: int
    section_occurrence: int
    event_index: int
    line_index: int
    chord: str
    section_id: str
    section_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_index": self.section_index,
            "section_occurrence": self.section_occurrence,
            "event_index": self.event_index,
            "line_index": self.line_index,
            "chord": self.chord,
            "section_id": self.section_id,
            "section_name": self.section_name,
        }


class ChartAlignment:
    """Motor de alinhamento temporal entre a reprodução musical e o ChordChart."""

    def __init__(self, chart: Optional[ChordChart] = None):
        self._chart = chart
        self._timeline_chords: List[Tuple[int, ChartChord, ChartSection, int, int, int]] = []
        # (bar, chord, section, chord_global_idx, repetition, index_in_occurrence)
        self._total_bars: int = 1
        self._start_anchor: Optional[StartAnchor] = None
        if chart:
            self._build_timeline()

    def set_chart(self, chart: ChordChart) -> None:
        """Atualiza a cifra associada e reconstrói a linha temporal de compassos."""
        self._chart = chart
        self._build_timeline()

    def _build_timeline(self) -> None:
        """Constrói mapa linear ordenado (bar, ChartChord, ChartSection) com expansão de repetições."""
        self._timeline_chords.clear()
        self._start_anchor = None
        if not self._chart or not self._chart.sections:
            self._total_bars = 1
            return

        bar_counter = 1
        global_chord_idx = 0
        occurrence_counts: Dict[Tuple[str, str], int] = {}
        for sec in self._chart.sections:
            playable_chords = [
                chord for chord in sec.chords
                if ChartSemanticClassifier.is_chord_shaped(chord.symbol.original_symbol)
            ]
            if not playable_chords:
                continue
            for rep in range(sec.repeat_count):
                occurrence_key = (sec.section_type.upper(), sec.name.strip().lower())
                occurrence_counts[occurrence_key] = occurrence_counts.get(occurrence_key, 0) + 1
                occurrence = occurrence_counts[occurrence_key]
                for chord_index, c in enumerate(playable_chords):
                    # A timeline do motor é uma allowlist: ChartChord criado por
                    # edição/importação manual ainda precisa provar que é acorde.
                    self._timeline_chords.append(
                        (bar_counter, c, sec, global_chord_idx, occurrence, chord_index)
                    )
                    bar_counter += int(max(1, c.duration_bars))
                    global_chord_idx += 1

        # Considera também a duração das linhas no line_map se houver
        max_line_bar = 1
        if self._chart.line_map:
            for lm in self._chart.line_map:
                if lm.end_bar > max_line_bar:
                    max_line_bar = lm.end_bar

        self._total_bars = max(1, max(bar_counter - 1, max_line_bar))
        if self._timeline_chords:
            position = self.get_position_for_event(0)
            self._start_anchor = StartAnchor(
                section_index=position.section_index,
                section_occurrence=position.section_occurrence,
                event_index=position.event_index,
                line_index=position.line_index,
                chord=position.current_chord,
                section_id=position.section_id,
                section_name=position.section_name,
            )

    @property
    def total_bars(self) -> int:
        return self._total_bars

    @property
    def start_anchor(self) -> Optional[StartAnchor]:
        return self._start_anchor

    def get_start_position(self, beat: float = 1.0,
                           absolute_time: float = 0.0) -> ChartPosition:
        """Retorna o início musical real, ignorando cabeçalhos e seções vazias."""
        if self._start_anchor is None:
            return self.get_position_at(1, beat, absolute_time)
        return self.get_position_for_event(
            self._start_anchor.event_index, beat=beat, absolute_time=absolute_time)

    def get_line_index_at(self, bar: int) -> int:
        """Mapeia um compasso diretamente para a linha física no editor de texto."""
        if not self._chart or not self._chart.line_map:
            return 1

        # 1. Prioriza linhas de acordes
        for lm in self._chart.line_map:
            if lm.line_type == "CHORD" and lm.start_bar <= bar <= lm.end_bar:
                return lm.line_number

        # 2. Em seguida, linhas de letra
        for lm in self._chart.line_map:
            if lm.line_type == "LYRIC" and lm.start_bar <= bar <= lm.end_bar:
                return lm.line_number

        # Se passou do último, retorna a última linha relevante
        content_lines = [lm for lm in self._chart.line_map if lm.line_type in ("CHORD", "LYRIC")]
        if content_lines:
            if bar >= content_lines[-1].end_bar:
                return content_lines[-1].line_number
            # Caso esteja antes da primeira
            if bar <= content_lines[0].start_bar:
                return content_lines[0].line_number

        return 1

    @property
    def event_count(self) -> int:
        """Quantidade de eventos de acorde após expandir as repetições declaradas."""
        return len(self._timeline_chords)

    def event_chords(self) -> List[str]:
        """Símbolos dos acordes na ordem real da cifra (repetições expandidas)."""
        return [entry[1].symbol.original_symbol for entry in self._timeline_chords]

    def event_section_types(self) -> List[str]:
        """Tipo de seção (INTRO, VERSE, CHORUS...) de cada evento da cifra."""
        return [str(entry[2].section_type or "UNKNOWN").upper() for entry in self._timeline_chords]

    def event_section_labels(self) -> List[str]:
        """"Nome da seção#ocorrência" de cada evento (ex.: "Refrão#2")."""
        return [f"{entry[2].name}#{entry[4]}" for entry in self._timeline_chords]

    def event_keys(self) -> List[str]:
        """Tom vigente (soante, como escrito em "Tom:") em cada evento da cifra."""
        if not self._chart or not self._timeline_chords:
            return []
        cache_key = (self._chart.key, tuple((e.anchor_bar, e.anchor_beat, e.to_key)
                                            for e in self._chart.key_changes))
        if getattr(self, "_event_keys_cache", (None, None))[0] != cache_key:
            if not self._chart.key_changes:
                keys = [self._chart.key] * len(self._timeline_chords)
            else:
                timeline = self._chart.key_timeline
                keys = [timeline.key_at(int(entry[1].bar), float(entry[1].beat))
                        for entry in self._timeline_chords]
            self._event_keys_cache = (cache_key, keys)
        return self._event_keys_cache[1]

    def key_at_event(self, event_index: int) -> str:
        keys = self.event_keys()
        if not keys:
            return self._chart.key if self._chart else "--"
        return keys[max(0, min(len(keys) - 1, int(event_index)))]

    def next_key_change(self, event_index: int) -> Optional[Tuple[int, str]]:
        """(evento onde o tom muda, novo tom) depois de ``event_index``, ou None."""
        keys = self.event_keys()
        if not keys or not self._chart.key_changes:
            return None
        index = max(0, min(len(keys) - 1, int(event_index)))
        for later in range(index + 1, len(keys)):
            if keys[later] != keys[index]:
                return later, keys[later]
        return None

    def event_bar(self, event_index: int) -> int:
        """Compasso da timeline (repetições expandidas) em que o evento começa."""
        if not self._timeline_chords:
            return 1
        return self._timeline_chords[max(0, min(len(self._timeline_chords) - 1, int(event_index)))][0]

    def event_entry(self, event_index: int):
        """(acorde da cifra, seção) do evento da timeline, ou None."""
        if not self._timeline_chords:
            return None
        entry = self._timeline_chords[max(0, min(len(self._timeline_chords) - 1, int(event_index)))]
        return entry[1], entry[2]

    def bars_for_chart_bar(self, chart_bar: int) -> List[int]:
        """Compassos da timeline (com repetições expandidas) do compasso escrito ``chart_bar``.

        A letra guarda o compasso ESCRITO (sem repetições); uma seção "2x" faz o
        mesmo compasso escrito aparecer duas vezes na execução.
        """
        bars = [entry[0] for entry in self._timeline_chords if int(entry[1].bar) == int(chart_bar)]
        return sorted(set(bars))

    def section_start_events(self) -> List[int]:
        """Índices de evento onde começa cada ocorrência de seção."""
        return [index for index, entry in enumerate(self._timeline_chords) if entry[5] == 0]

    def get_position_for_event(self, event_index: int, beat: float = 1.0,
                               absolute_time: float = 0.0) -> ChartPosition:
        """Consulta um evento pela sua ordem real na cifra, incluindo repetições."""
        if not self._timeline_chords:
            return self.get_position_at(1, beat, absolute_time)
        index = max(0, min(len(self._timeline_chords) - 1, int(event_index)))
        return self.get_position_at(self._timeline_chords[index][0], beat, absolute_time)

    def get_position_at(self, bar: int, beat: float = 1.0, absolute_time: float = 0.0) -> ChartPosition:
        """Consulta a posição correspondente na cifra para um dado compasso e tempo."""
        line_idx = self.get_line_index_at(bar)

        if not self._timeline_chords:
            return ChartPosition(
                current_bar=bar,
                current_beat=beat,
                line_index=line_idx,
                absolute_time=absolute_time
            )

        current_idx = -1
        # Busca o acorde ativo para este compasso
        for idx, (b, ch, sec, g_idx, occurrence, section_event_idx) in enumerate(self._timeline_chords):
            if b <= bar:
                current_idx = idx
            else:
                break

        if current_idx == -1:
            current_idx = 0

        (bar_start, chord_obj, section_obj, chord_global_idx,
         occurrence, section_event_idx) = self._timeline_chords[current_idx]

        # Próximo acorde
        next_chord_str = "--"
        bars_to_chord_change = max(0, int(chord_obj.duration_bars) - (bar - bar_start))
        if current_idx + 1 < len(self._timeline_chords):
            next_chord_str = self._timeline_chords[current_idx + 1][1].symbol.original_symbol
            bars_to_chord_change = max(1, self._timeline_chords[current_idx + 1][0] - bar)

        # Seção e progresso
        sec_chords = [item for item in self._timeline_chords
                      if item[2].id == section_obj.id and item[4] == occurrence]
        first_bar_sec = sec_chords[0][0] if sec_chords else bar_start
        last_bar_sec = (sec_chords[-1][0] + int(sec_chords[-1][1].duration_bars)
                        if sec_chords else bar_start + 1)
        sec_duration = max(1, last_bar_sec - first_bar_sec)
        sec_progress = max(0.0, min(1.0, (bar - first_bar_sec) / sec_duration))
        bars_to_sec_change = max(0, last_bar_sec - bar)

        # Próxima seção
        next_sec_name = "--"
        for item in self._timeline_chords[current_idx + 1:]:
            if item[2].id != section_obj.id or item[4] != occurrence:
                next_sec_name = item[2].name
                break

        # Letra alinhada
        lyric_text = chord_obj.lyric

        # Índice da seção no chart
        sec_idx = 0
        if self._chart and self._chart.sections:
            for s_i, s in enumerate(self._chart.sections):
                if s.id == section_obj.id:
                    sec_idx = s_i
                    break

        # Linha física preferencial:
        # line_idx considera o mapeamento exato linha a linha (incluindo letras)
        final_line_idx = line_idx if line_idx > 0 else (chord_obj.line_number if chord_obj.line_number > 0 else 1)

        return ChartPosition(
            current_bar=bar,
            current_beat=beat,
            section_id=section_obj.id,
            section_name=section_obj.name,
            section_type=section_obj.section_type,
            section_index=sec_idx,
            current_chord=chord_obj.symbol.original_symbol,
            next_chord=next_chord_str,
            next_section_name=next_sec_name,
            chord_index_in_section=section_event_idx,
            chord_index=chord_global_idx,
            element_index=current_idx,
            event_index=current_idx,
            next_event_index=(current_idx + 1 if current_idx + 1 < len(self._timeline_chords) else -1),
            section_event_index=section_event_idx,
            section_event_count=len(sec_chords),
            section_occurrence=occurrence,
            line_index=final_line_idx,
            section_progress=sec_progress,
            bars_until_chord_change=bars_to_chord_change,
            bars_until_section_change=bars_to_sec_change,
            current_lyric=lyric_text,
            is_last_chord=(current_idx == len(self._timeline_chords) - 1),
            absolute_time=absolute_time,
            confidence=0.90,
            tracking_state="TRACKING"
        )

    def align_from_clock(self, clock: MusicalClock) -> ChartPosition:
        """Alinhamento temporal sem localização por áudio (sem offset).

        Sessões com rastreamento devem consumir SongSession.chart_position.
        """
        return self.get_position_at(
            bar=clock.bar,
            beat=clock.beat + clock.beat_position,
            absolute_time=clock.elapsed_time
        )
