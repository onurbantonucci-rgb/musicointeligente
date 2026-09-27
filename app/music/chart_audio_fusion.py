"""Fusão Inteligente entre Cifra, Análise de Áudio e Relógio Musical (ChartAudioFusion v0.4).

Implementa a hierarquia prioritária de conhecimento musical:
1. CHART / CIFRA — O que DEVERIA acontecer (Referência principal).
2. AUDIO ANALYSIS — O que ESTÁ acontecendo (Confirmação, sincronização, variações).
3. EXECUTION VARIATIONS — Detecção de divergências sem sobrescrever a cifra.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from app.music.chart_alignment import ChartPosition
from app.music.chord_chart import parse_chord
from app.input.chart_semantic_classifier import ChartSemanticClassifier


@dataclass
class DiscrepancyEvent:
    """Registro de divergência entre a cifra esperada e a execução detectada no áudio."""
    timestamp: float
    bar: int
    beat: int
    expected_chord: str
    detected_chord: str
    detected_confidence: float
    duration: float = 0.0
    classification: str = "possible_performance_variation" # "match", "variation", "transposition", "low_confidence"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "bar": self.bar,
            "beat": self.beat,
            "expected_chord": self.expected_chord,
            "detected_chord": self.detected_chord,
            "detected_confidence": round(self.detected_confidence, 2),
            "duration": round(self.duration, 2),
            "classification": self.classification,
        }


@dataclass
class FusedMusicalState:
    """Estado musical consolidado após a fusão de Cifra + Áudio + Relógio."""
    expected_chord: str = "--"
    detected_chord: str = "--"
    effective_chord: str = "--"         # Acorde utilizado pela banda (cifra por padrão)
    next_expected_chord: str = "--"
    current_section: str = "UNKNOWN"
    next_section: str = "UNKNOWN"
    current_bar: int = 1
    current_beat: int = 1
    is_discrepancy: bool = False
    discrepancy_duration: float = 0.0
    confirmed_variation: bool = False
    confidence: float = 0.90
    lyric: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_chord": self.expected_chord,
            "detected_chord": self.detected_chord,
            "effective_chord": self.effective_chord,
            "next_expected_chord": self.next_expected_chord,
            "current_section": self.current_section,
            "next_section": self.next_section,
            "current_bar": self.current_bar,
            "current_beat": self.current_beat,
            "is_discrepancy": self.is_discrepancy,
            "discrepancy_duration": round(self.discrepancy_duration, 2),
            "confirmed_variation": self.confirmed_variation,
            "confidence": round(self.confidence, 2),
            "lyric": self.lyric,
        }


class ChartAudioFusion:
    """Fusor que combina o modelo de expectativa (cifra) com o sinal de percepção (áudio)."""

    def __init__(self, variation_confirmation_seconds: float = 3.0):
        self._variation_threshold = variation_confirmation_seconds
        self._current_discrepancy: Optional[DiscrepancyEvent] = None
        self._discrepancies_history: List[DiscrepancyEvent] = []
        self._last_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Limpa o histórico de discrepâncias e variações acumuladas."""
        self._current_discrepancy = None
        self._discrepancies_history.clear()
        self._last_timestamp = None

    @property
    def discrepancies(self) -> List[DiscrepancyEvent]:
        return list(self._discrepancies_history)

    def fuse(
        self,
        chart_pos: ChartPosition,
        detected_chord: str,
        chord_confidence: float,
        timestamp: float
    ) -> FusedMusicalState:
        """Executa a fusão hierárquica entre a posição da cifra e o acorde detectado no áudio."""
        expected = chart_pos.current_chord
        # Percepção externa só entra na fusão se for um símbolo harmônico válido.
        # Texto, números e metadados são ausência de observação, nunca variação.
        if (detected_chord not in ("", "--", "UNKNOWN", "N") and
                not ChartSemanticClassifier.is_chord_shaped(detected_chord)):
            detected_chord = "--"
            chord_confidence = 0.0
        dt = max(0.0, timestamp - self._last_timestamp) if self._last_timestamp is not None else 0.0
        self._last_timestamp = timestamp

        # Se não há acorde esperado na cifra, utiliza a percepção do áudio
        if not expected or expected == "--":
            return FusedMusicalState(
                expected_chord="--",
                detected_chord=detected_chord,
                effective_chord=detected_chord if detected_chord != "--" else "--",
                next_expected_chord=chart_pos.next_chord,
                current_section=chart_pos.section_name,
                next_section=chart_pos.next_section_name,
                current_bar=chart_pos.current_bar,
                current_beat=int(chart_pos.current_beat),
                confidence=chord_confidence,
                lyric=chart_pos.current_lyric
            )

        # Se o áudio não detectou nada convincente, a cifra comanda soberana
        if not detected_chord or detected_chord == "--" or chord_confidence < 0.35:
            if self._current_discrepancy is not None:
                self._current_discrepancy = None

            return FusedMusicalState(
                expected_chord=expected,
                detected_chord=detected_chord if detected_chord else "--",
                effective_chord=expected,
                next_expected_chord=chart_pos.next_chord,
                current_section=chart_pos.section_name,
                next_section=chart_pos.next_section_name,
                current_bar=chart_pos.current_bar,
                current_beat=int(chart_pos.current_beat),
                is_discrepancy=False,
                confidence=0.92,
                lyric=chart_pos.current_lyric
            )

        # Comparação teórica estruturada entre esperado e detectado
        expected_sym = parse_chord(expected)
        detected_sym = parse_chord(detected_chord)

        is_match = expected_sym.is_synonym_of(detected_sym) or (expected_sym.root == detected_sym.root and expected_sym.quality == detected_sym.quality)

        if is_match:
            # Concordância harmônica plena
            self._current_discrepancy = None
            return FusedMusicalState(
                expected_chord=expected,
                detected_chord=detected_chord,
                effective_chord=expected, # Preserva grafia original da cifra
                next_expected_chord=chart_pos.next_chord,
                current_section=chart_pos.section_name,
                next_section=chart_pos.next_section_name,
                current_bar=chart_pos.current_bar,
                current_beat=int(chart_pos.current_beat),
                is_discrepancy=False,
                confidence=min(0.98, 0.85 + (chord_confidence * 0.13)),
                lyric=chart_pos.current_lyric
            )

        # Divergência detectada: Cifra diz X, mas áudio detecta Y
        if self._current_discrepancy is None or self._current_discrepancy.detected_chord != detected_chord:
            # Início de uma nova discrepância
            self._current_discrepancy = DiscrepancyEvent(
                timestamp=timestamp,
                bar=chart_pos.current_bar,
                beat=int(chart_pos.current_beat),
                expected_chord=expected,
                detected_chord=detected_chord,
                detected_confidence=chord_confidence,
                duration=dt,
                classification="possible_performance_variation"
            )
            self._discrepancies_history.append(self._current_discrepancy)
        else:
            self._current_discrepancy.duration += dt

        discrepancy_dur = self._current_discrepancy.duration
        confirmed_var = discrepancy_dur >= self._variation_threshold

        return FusedMusicalState(
            expected_chord=expected,
            detected_chord=detected_chord,
            effective_chord=expected, # A cifra NÃO é sobrescrita arbitrariamente
            next_expected_chord=chart_pos.next_chord,
            current_section=chart_pos.section_name,
            next_section=chart_pos.next_section_name,
            current_bar=chart_pos.current_bar,
            current_beat=int(chart_pos.current_beat),
            is_discrepancy=True,
            discrepancy_duration=discrepancy_dur,
            confirmed_variation=confirmed_var,
            confidence=0.75,
            lyric=chart_pos.current_lyric
        )
