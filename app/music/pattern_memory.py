"""Memória Dinâmica de Padrões Musicais (PatternMemory v0.3).

Armazena, agrupa e aprende os padrões harmônicos e rítmicos que emergem durante
a audição da música, mantendo registro de ocorrências, durações, transições probabilísticas
e contexto tonal.
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from app.music.music_structure import MusicalPattern, PatternOccurrence, SectionTransition
from app.music.harmonic_normalization import (
    NormalizedChordSequence,
    sequence_similarity,
    normalize_chord_sequence
)
from app.music.harmonic_rhythm import HarmonicRhythmPattern


class PatternMemory:
    """Repositório inteligente de aprendizado e memória de padrões musicais."""

    def __init__(self, similarity_threshold: float = 0.82):
        self._similarity_threshold = similarity_threshold
        self._patterns: Dict[str, MusicalPattern] = {}
        self._pattern_counter: int = 0
        
        # Histórico ordenado de ocorrências na música
        self._occurrences_timeline: List[PatternOccurrence] = []

        # Tabela de transições entre padrões: {from_pattern: {to_pattern: count}}
        self._pattern_transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._pattern_transition_durations: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        # Tabela de transições entre seções: {(from_sec, to_sec): SectionTransition}
        self._section_transitions: Dict[Tuple[str, str], SectionTransition] = {}
        self._harmonic_rhythms: Dict[str, HarmonicRhythmPattern] = {}
        self._harmonic_rhythm_counter: int = 0

    def reset(self) -> None:
        """Limpa toda a memória de padrões e transições para uma nova análise."""
        self._patterns.clear()
        self._pattern_counter = 0
        self._occurrences_timeline.clear()
        self._pattern_transition_counts.clear()
        self._pattern_transition_durations.clear()
        self._section_transitions.clear()
        self._harmonic_rhythms.clear()
        self._harmonic_rhythm_counter = 0

    def register_harmonic_rhythm(self, section_key: str, chord_sequence: List[str],
                                 duration_sequence: List[float], confidence: float) -> HarmonicRhythmPattern:
        """Aprende uma seção inteira e rejeita um outlier isolado por média robusta."""
        key = f"{section_key}|{'|'.join(chord_sequence)}"
        pattern = self._harmonic_rhythms.get(key)
        if pattern is None:
            self._harmonic_rhythm_counter += 1
            pattern = HarmonicRhythmPattern(
                id=f"{section_key}_RHYTHM_{self._harmonic_rhythm_counter:02d}",
                section_key=section_key, chord_sequence=list(chord_sequence),
                duration_sequence=list(duration_sequence), confidence=max(0.0, min(1.0, confidence)),
                variance=[0.0] * len(duration_sequence))
            self._harmonic_rhythms[key] = pattern
            return pattern
        if len(pattern.duration_sequence) == len(duration_sequence):
            for i, observed in enumerate(duration_sequence):
                prior = pattern.duration_sequence[i]
                deviation = abs(observed - prior)
                # Um desvio grande precisa reaparecer: não ensina uma execução errada.
                gain = 0.25 if deviation <= max(0.5, prior * .30) else 0.05
                pattern.duration_sequence[i] = prior + (observed - prior) * gain
                pattern.variance[i] = pattern.variance[i] * .75 + deviation * .25
        pattern.observations += 1
        pattern.confidence = min(.98, pattern.confidence * .80 + max(0.0, min(1.0, confidence)) * .20 + .04)
        return pattern

    def find_harmonic_rhythm(self, section_key: str, chord: str, chord_index: int = 0) -> Optional[HarmonicRhythmPattern]:
        candidates = [pattern for pattern in self._harmonic_rhythms.values()
                      if pattern.section_key == section_key and pattern.chord_sequence]
        exact = [pattern for pattern in candidates
                 if chord_index < len(pattern.chord_sequence) and pattern.chord_sequence[chord_index] == chord]
        if exact:
            return max(exact, key=lambda pattern: (pattern.observations, pattern.confidence))
        matching = [pattern for pattern in candidates if pattern.chord_sequence[0] == chord]
        return max(matching, key=lambda pattern: (pattern.observations, pattern.confidence)) if matching else None

    def get_harmonic_rhythm_patterns(self) -> List[HarmonicRhythmPattern]:
        return list(self._harmonic_rhythms.values())

    def get_all_patterns(self) -> List[MusicalPattern]:
        """Retorna todos os padrões harmônicos atualmente aprendidos."""
        return list(self._patterns.values())

    def get_known_patterns(self) -> List[MusicalPattern]:
        """Alias para get_all_patterns."""
        return list(self._patterns.values())

    def get_pattern(self, pattern_id: str) -> Optional[MusicalPattern]:
        """Recupera um padrão específico por seu ID."""
        return self._patterns.get(pattern_id)

    def get_occurrences(self, pattern_id: str) -> List[PatternOccurrence]:
        """Retorna todas as ocorrências cronológicas de um padrão."""
        pat = self._patterns.get(pattern_id)
        return list(pat.occurrences) if pat else []

    def find_similar_pattern(
        self,
        normalized_seq: NormalizedChordSequence,
        threshold: Optional[float] = None
    ) -> Optional[Tuple[MusicalPattern, float]]:
        """Procura se a sequência harmônica já corresponde a um padrão conhecido na memória."""
        th = threshold if threshold is not None else self._similarity_threshold
        best_match: Optional[MusicalPattern] = None
        best_score: float = 0.0

        for pat in self._patterns.values():
            # Reconstrói NormalizedChordSequence do padrão para comparação
            pat_seq = NormalizedChordSequence(
                absolute_chords=pat.canonical_chords,
                roman_numerals=pat.roman_numerals,
                relative_intervals=pat.relative_intervals,
                key_context="C Major"
            )
            sim = sequence_similarity(normalized_seq, pat_seq)
            if sim >= th and sim > best_score:
                best_score = sim
                best_match = pat

        if best_match is not None:
            return best_match, best_score
        return None

    def register_pattern(
        self,
        normalized_seq: Any = None,
        start_bar: int = 1,
        end_bar: Optional[int] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        key: str = "C Major",
        section_type: str = "UNKNOWN",
        chord_sequence: Optional[List[str]] = None,
        duration_bars: Optional[int] = None,
        tonal_center: Optional[str] = None
    ) -> Tuple[MusicalPattern, bool, PatternOccurrence]:
        """Registra uma sequência na memória.
        
        Se já for similar a um padrão existente, adiciona uma nova ocorrência (#2, #3...).
        Caso contrário, cria um novo padrão (ex: P01, P02).
        
        Retorna: (MusicalPattern, is_new: bool, PatternOccurrence)
        """
        if chord_sequence is not None and normalized_seq is None:
            normalized_seq = chord_sequence
        if tonal_center is not None and key == "C Major":
            key = tonal_center
        if isinstance(normalized_seq, (list, tuple)):
            normalized_seq = normalize_chord_sequence(list(normalized_seq), tonal_center=key)

        dur_bars = duration_bars if duration_bars is not None else (end_bar - start_bar + 1 if end_bar is not None else 4)
        if end_bar is None:
            end_bar = start_bar + dur_bars - 1
        if end_time is None:
            end_time = start_time + dur_bars * 2.0

        duration_bars = max(1, end_bar - start_bar + 1)
        match = self.find_similar_pattern(normalized_seq)

        if match is not None:
            pat, sim_score = match
            occ = PatternOccurrence(
                pattern_id=pat.id,
                start_time=start_time,
                end_time=end_time,
                start_bar=start_bar,
                end_bar=end_bar,
                key_context=key,
                confidence=round(min(0.99, pat.confidence * 0.5 + sim_score * 0.5), 2),
                chord_sequence=list(normalized_seq.absolute_chords),
                duration_bars=duration_bars
            )
            pat.add_occurrence(occ)
            self._occurrences_timeline.append(occ)
            return pat, False, occ


        # Novo padrão não visto anteriormente
        self._pattern_counter += 1
        new_id = f"P{self._pattern_counter:02d}"

        new_pat = MusicalPattern(
            id=new_id,
            roman_numerals=list(normalized_seq.roman_numerals),
            relative_intervals=list(normalized_seq.relative_intervals),
            canonical_chords=list(normalized_seq.absolute_chords),
            duration_bars=duration_bars,
            associated_section_type=section_type,
            confidence=normalized_seq.confidence
        )

        occ = PatternOccurrence(
            pattern_id=new_id,
            start_time=start_time,
            end_time=end_time,
            start_bar=start_bar,
            end_bar=end_bar,
            key_context=key,
            confidence=normalized_seq.confidence,
            chord_sequence=list(normalized_seq.absolute_chords),
            duration_bars=duration_bars
        )
        new_pat.add_occurrence(occ)

        self._patterns[new_id] = new_pat
        self._occurrences_timeline.append(occ)

        return new_pat, True, occ

    def record_transition(
        self,
        from_pattern_id: str,
        to_pattern_id: str,
        from_section: str = "UNKNOWN",
        to_section: str = "UNKNOWN",
        duration: float = 0.0
    ) -> None:
        """Registra uma transição sequencial entre padrões e entre seções."""
        if not from_pattern_id or not to_pattern_id or from_pattern_id == "--" or to_pattern_id == "--":
            return

        # 1. Transição de padrões
        self._pattern_transition_counts[from_pattern_id][to_pattern_id] += 1
        if duration > 0.0:
            self._pattern_transition_durations[from_pattern_id][to_pattern_id].append(duration)

        # 2. Transição de seções
        sec_key = (from_section, to_section)
        if sec_key in self._section_transitions:
            trans = self._section_transitions[sec_key]
            trans.count += 1
            trans.confidence = min(0.99, trans.confidence + 0.05)
        else:
            self._section_transitions[sec_key] = SectionTransition(
                from_section=from_section,
                to_section=to_section,
                from_pattern=from_pattern_id,
                to_pattern=to_pattern_id,
                count=1,
                probability=1.0,
                average_duration=duration,
                confidence=0.75
            )

        # Recalcula probabilidades relativas de seções saindo de from_section
        total_out = sum(t.count for (src, _), t in self._section_transitions.items() if src == from_section)
        if total_out > 0:
            for (src, _), t in self._section_transitions.items():
                if src == from_section:
                    t.probability = round(t.count / total_out, 3)

    def get_next_patterns(self, pattern_id: str) -> List[Tuple[str, float]]:
        """Retorna lista ordenada de (next_pattern_id, probability) após pattern_id."""
        sub_dict = self._pattern_transition_counts.get(pattern_id, {})
        total = sum(sub_dict.values())
        if total == 0:
            return []

        res = []
        for next_id, cnt in sub_dict.items():
            prob = cnt / total
            res.append((next_id, round(prob, 3)))
        res.sort(key=lambda x: x[1], reverse=True)
        return res

    def get_previous_patterns(self, pattern_id: str) -> List[Tuple[str, float]]:
        """Retorna lista ordenada de (prev_pattern_id, probability) que antecedem pattern_id."""
        incoming = []
        total = 0
        for src_id, targets in self._pattern_transition_counts.items():
            if pattern_id in targets:
                cnt = targets[pattern_id]
                incoming.append((src_id, cnt))
                total += cnt

        if total == 0:
            return []

        res = [(src, round(cnt / total, 3)) for src, cnt in incoming]
        res.sort(key=lambda x: x[1], reverse=True)
        return res

    def get_pattern_probability(self, next_pattern_id: str, current_pattern_id: str) -> float:
        """Retorna a probabilidade condicional P(next_pattern | current_pattern)."""
        sub_dict = self._pattern_transition_counts.get(current_pattern_id, {})
        total = sum(sub_dict.values())
        if total == 0:
            return 0.0
        return round(sub_dict.get(next_pattern_id, 0) / total, 3)

    def get_transitions(self) -> List[SectionTransition]:
        """Retorna todas as transições entre seções registradas."""
        return list(self._section_transitions.values())

    def get_section_transition_probability(self, from_section: str, to_section: str) -> float:
        """Retorna a probabilidade condicional P(to_section | from_section)."""
        trans = self._section_transitions.get((from_section, to_section))
        return trans.probability if trans else 0.0

    def get_transition_probability(self, from_section: str, to_section: str) -> float:
        """Alias para get_section_transition_probability."""
        return self.get_section_transition_probability(from_section, to_section)
