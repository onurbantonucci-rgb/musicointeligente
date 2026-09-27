"""Motor de Previsão Musical e Antecipação (PredictionEngine v0.3).

Capaz de:
1. Prever a próxima seção musical com base no aprendizado de repetições e matriz de transições.
2. Prever os próximos acordes dentro do padrão atual (Harmonic Look-Ahead).
3. Calcular tempo e quantidade de compassos até a próxima transição (bars_until_change).
4. Fornecer justificativa musical e grau probabilístico de confiança (sem falsas certezas).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from app.music.music_structure import MusicPosition, MusicalPattern
from app.music.pattern_memory import PatternMemory


@dataclass
class PredictedChordSequence:
    """Sequência de acordes previstos para os próximos tempos/compassos."""
    chords: List[str]
    beats_until: int
    confidence: float
    source_pattern: str = "--"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chords": list(self.chords),
            "beats_until": self.beats_until,
            "confidence": round(self.confidence, 2),
            "source_pattern": self.source_pattern,
        }


@dataclass
class MusicPrediction:
    """Estrutura consolidada de previsão musical para consumo da UI e dos músicos autônomos."""
    predicted_section: str = "UNKNOWN"
    predicted_pattern: str = "--"
    predicted_chords: List[str] = field(default_factory=list)
    bars_until_change: int = 0
    beats_until_change: int = 0
    confidence: float = 0.40
    prediction_reason: str = "Aguardando evidências de repetição"
    source_bar: int = 1
    source_beat: int = 1
    predicted_sequence: List[Tuple[int, str]] = field(default_factory=list) # [(bar_relativo, acorde)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_bar": self.source_bar,
            "source_beat": self.source_beat,
            "predicted_section": self.predicted_section,
            "predicted_pattern": self.predicted_pattern,
            "predicted_chords": list(self.predicted_chords),
            "bars_until_change": self.bars_until_change,
            "beats_until_change": self.beats_until_change,
            "confidence": round(self.confidence, 2),
            "prediction_reason": self.prediction_reason,
            "predicted_sequence": [
                {"bar_offset": b, "chord": c} for b, c in self.predicted_sequence
            ],
        }


class PredictionEngine:
    """Motor que antecipa seções, padrões e acordes a partir da estrutura aprendida."""

    def __init__(self, default_beats_per_bar: int = 4):
        self._beats_per_bar = default_beats_per_bar
        self._latest_prediction: Optional[MusicPrediction] = None

    @property
    def latest_prediction(self) -> Optional[MusicPrediction]:
        return self._latest_prediction

    def reset(self) -> None:
        self._latest_prediction = None

    def predict_next(self, current_position: MusicPosition,
                     current_pattern: Optional[MusicalPattern], pattern_memory: PatternMemory,
                     current_section_type: str = "UNKNOWN", bars_ahead: int = 1,
                     harmonic_rhythm=None) -> MusicPrediction:
        """Consome a posição musical fornecida; não consulta relógios ou offsets."""
        self._latest_prediction = self._predict_next(
            current_position, current_pattern, pattern_memory, current_section_type, bars_ahead,
            harmonic_rhythm)
        self._latest_prediction.source_bar = current_position.current_bar
        self._latest_prediction.source_beat = current_position.current_beat
        return self._latest_prediction

    def _predict_next(
        self,
        current_position: MusicPosition,
        current_pattern: Optional[MusicalPattern],
        pattern_memory: PatternMemory,
        current_section_type: str = "UNKNOWN",
        bars_ahead: int = 1,
        harmonic_rhythm=None,
    ) -> MusicPrediction:
        """Gera a predição da próxima seção, padrão e acordes vindouros."""
        # Perfil observado em beats tem prioridade sobre a suposição antiga de
        # um acorde por compasso, mas só após aprendizagem confiável da seção.
        if harmonic_rhythm is not None:
            expected = float(getattr(harmonic_rhythm, "expected_chord_duration_beats", 0.0))
            remaining = float(getattr(harmonic_rhythm, "beats_until_change", 0.0) or
                              getattr(harmonic_rhythm, "beats_until_chord_change", 0.0))
            next_chord = str(getattr(harmonic_rhythm, "next_chord",
                                     getattr(harmonic_rhythm, "rhythmic_next_chord", "--")))
            confidence = min(float(getattr(harmonic_rhythm, "duration_confidence", 0.0)),
                             float(getattr(harmonic_rhythm, "pattern_confidence", 0.0)))
            pattern_id = str(getattr(harmonic_rhythm, "harmonic_rhythm_pattern", "--"))
            if expected > 0.0 and next_chord != "--" and confidence >= 0.55:
                return MusicPrediction(
                    predicted_section=current_section_type,
                    predicted_pattern=pattern_id,
                    predicted_chords=[next_chord],
                    bars_until_change=max(0, int(remaining // self._beats_per_bar)),
                    beats_until_change=remaining,
                    confidence=confidence,
                    prediction_reason=(f"Ritmo harmônico aprendido: mudança após "
                                       f"{expected:.2f} beats")
                )
        if current_pattern is None:
            # Sem padrão ativo identificado
            return MusicPrediction(
                predicted_section="UNKNOWN",
                predicted_pattern="--",
                predicted_chords=[],
                bars_until_change=0,
                beats_until_change=0,
                confidence=0.35,
                prediction_reason="Padrão atual ainda não identificado na memória"
            )

        duration_bars = max(1, current_pattern.duration_bars)
        
        # 1. Posição dentro do padrão atual (0-indexada em compassos)
        curr_bar_offset = (current_position.current_bar - 1) % duration_bars
        bars_remaining_in_pattern = max(0, duration_bars - curr_bar_offset - 1)
        
        beats_in_bar = self._beats_per_bar
        curr_beat = max(1, min(beats_in_bar, current_position.current_beat))
        beats_remaining_in_bar = beats_in_bar - curr_beat
        total_beats_until_change = (bars_remaining_in_pattern * beats_in_bar) + beats_remaining_in_bar

        canonical = current_pattern.canonical_chords

        # 2. Se o padrão atual ainda possui compassos restantes antes de mudar:
        if bars_remaining_in_pattern > 0 and len(canonical) > 0:
            # Os próximos acordes imediatos são os acordes restantes dentro do padrão
            chords_per_bar = max(1, len(canonical) // duration_bars)
            next_chord_idx = min(len(canonical) - 1, (curr_bar_offset + 1) * chords_per_bar)
            remaining_chords = canonical[next_chord_idx:]

            # Previsão da próxima seção quando este padrão terminar
            next_patterns = pattern_memory.get_next_patterns(current_pattern.id)
            if next_patterns:
                next_pat_id, prob = next_patterns[0]
                target_pat = pattern_memory.get_pattern(next_pat_id)
                next_sec = target_pat.associated_section_type if target_pat else "UNKNOWN"
                reason = (f"Padrão {next_pat_id} ({next_sec}) sucede {current_pattern.id} "
                          f"com prob {int(prob * 100)}% após {bars_remaining_in_pattern} compassos")
                pred_confidence = round(min(0.95, prob * current_pattern.confidence + 0.1), 2)
            else:
                next_pat_id = current_pattern.id # Hipótese de repetição do mesmo padrão
                next_sec = current_section_type
                reason = f"Concluindo padrão {current_pattern.id} ({bars_remaining_in_pattern} compassos restantes)"
                pred_confidence = round(current_pattern.confidence * 0.8, 2)

            # Gera projeção sequencial (look-ahead)
            seq = []
            for b in range(1, min(5, bars_remaining_in_pattern + 2)):
                idx = (curr_bar_offset + b) % len(canonical) if canonical else 0
                c = canonical[idx] if canonical else "--"
                seq.append((b, c))

            return MusicPrediction(
                predicted_section=next_sec,
                predicted_pattern=next_pat_id,
                predicted_chords=remaining_chords if remaining_chords else canonical[:2],
                bars_until_change=bars_remaining_in_pattern,
                beats_until_change=total_beats_until_change,
                confidence=pred_confidence,
                prediction_reason=reason,
                predicted_sequence=seq
            )

        # 3. Estamos no último compasso do padrão atual: Transição Iminente!
        next_patterns = pattern_memory.get_next_patterns(current_pattern.id)

        if next_patterns:
            sorted_next = sorted(next_patterns, key=lambda x: x[1], reverse=True)
            next_pat_id, prob = sorted_next[0]
            target_pat = pattern_memory.get_pattern(next_pat_id)
            predicted_sec = target_pat.associated_section_type if (target_pat and target_pat.associated_section_type != "UNKNOWN") else None

            if not predicted_sec or predicted_sec == "UNKNOWN":
                sec_trans = pattern_memory.get_transitions()
                matching_tr = [tr for tr in sec_trans if tr.to_pattern == next_pat_id or tr.from_section == current_section_type]
                if matching_tr:
                    matching_tr.sort(key=lambda t: t.probability, reverse=True)
                    predicted_sec = matching_tr[0].to_section

            if not predicted_sec and current_pattern.expected_next_section != "UNKNOWN":
                predicted_sec = current_pattern.expected_next_section

            if not predicted_sec:
                predicted_sec = "UNKNOWN"

            pred_chords = target_pat.canonical_chords if target_pat else []
            
            # Razão com contagem de evidências
            reason = (f"Probabilidade de {int(prob * 100)}% de transição para {predicted_sec} ({next_pat_id}) "
                      f"após {current_pattern.id}")
            
            # Confiança proporcional à probabilidade aprendida e ao número de ocorrências
            count_bonus = min(0.20, len(pattern_memory.get_occurrences(next_pat_id)) * 0.05)
            pred_conf = round(min(0.96, (prob * 0.75) + count_bonus), 2)

            seq = [(b + 1, pred_chords[b % len(pred_chords)]) for b in range(min(4, len(pred_chords)))] if pred_chords else []

            return MusicPrediction(
                predicted_section=predicted_sec,
                predicted_pattern=next_pat_id,
                predicted_chords=pred_chords[:4],
                bars_until_change=1,
                beats_until_change=beats_remaining_in_bar,
                confidence=pred_conf,
                prediction_reason=reason,
                predicted_sequence=seq
            )

        # 3.b Consulta transições por tipo de seção se não houver por ID de padrão
        sec_trans = [tr for tr in pattern_memory.get_transitions() if tr.from_section == current_section_type]
        if sec_trans:
            sec_trans.sort(key=lambda t: t.probability, reverse=True)
            best_tr = sec_trans[0]
            return MusicPrediction(
                predicted_section=best_tr.to_section,
                predicted_pattern=best_tr.to_pattern,
                predicted_chords=[],
                bars_until_change=1,
                beats_until_change=beats_remaining_in_bar,
                confidence=round(min(0.95, best_tr.probability * 0.85), 2),
                prediction_reason=f"Probabilidade de {int(best_tr.probability * 100)}% de transição {best_tr.from_section} -> {best_tr.to_section}"
            )

        # 4. Sem histórico de transições anteriores para este padrão:
        # Se for um padrão de alta repetição, prevê a repetição de si próprio (A -> A)
        if current_pattern.occurrence_count >= 2:
            return MusicPrediction(
                predicted_section=current_section_type,
                predicted_pattern=current_pattern.id,
                predicted_chords=canonical[:4],
                bars_until_change=1,
                beats_until_change=beats_remaining_in_bar,
                confidence=0.72,
                prediction_reason=f"Repetição do padrão ativo {current_pattern.id} (Padrão estável {current_pattern.occurrence_count}x)"
            )

        # 5. Fallback: primeira ocorrência sem transições mapeadas ainda
        return MusicPrediction(
            predicted_section=current_section_type if current_section_type != "UNKNOWN" else "UNKNOWN",
            predicted_pattern=current_pattern.id,
            predicted_chords=canonical[:2] if canonical else [],
            bars_until_change=1,
            beats_until_change=beats_remaining_in_bar,
            confidence=0.45,
            prediction_reason=f"Padrão {current_pattern.id} em observação inicial (1ª ocorrência)"
        )

    def predict_harmonic_lookahead(
        self,
        pattern: MusicalPattern,
        bar_in_pattern: int = 1,
        lookahead_bars: int = 4
    ) -> List[str]:
        """Projeta os acordes dos próximos N compassos dentro do padrão ou repetição."""
        chords = pattern.canonical_chords
        if not chords:
            return []
        res = []
        for b in range(1, lookahead_bars + 1):
            idx = (bar_in_pattern - 1 + b) % len(chords)
            res.append(chords[idx])
        return res

    def get_lookahead_chords(
        self,
        current_position: MusicPosition,
        current_pattern: Optional[MusicalPattern],
        pattern_memory: PatternMemory,
        bars_ahead: int = 4
    ) -> List[Tuple[int, str]]:
        """Retorna projeção direta dos acordes esperados para os próximos N compassos."""
        pred = self.predict_next(
            current_position=current_position,
            current_pattern=current_pattern,
            pattern_memory=pattern_memory,
            bars_ahead=bars_ahead
        )
        return pred.predicted_sequence
