"""Modelos e estruturas de dados para representação da Estrutura Musical (v0.3).

Define as entidades de Seção, Padrão, Ocorrência, Transição, Posição e Estrutura Global
do Virtual Band AI, sem forçar classificações rígidas e admitindo UNKNOWN com confiança probabilística.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any


class SectionType(Enum):
    """Tipos de seções formais aceitos pelo sistema."""
    UNKNOWN = "UNKNOWN"                 # Sem evidência conclusiva suficiente (comportamento seguro)
    INTRO = "INTRO"                     # Seção inicial / introdução
    VERSE = "VERSE"                     # Estrofe / Verso
    PRE_CHORUS = "PRE_CHORUS"           # Pré-refrão / preparação
    CHORUS = "CHORUS"                   # Refrão / gancho temático principal
    BRIDGE = "BRIDGE"                   # Ponte / quebra temática contrastante
    SOLO = "SOLO"                       # Solo instrumental
    INSTRUMENTAL = "INSTRUMENTAL"       # Seção instrumental intermediária
    BREAK = "BREAK"                     # Quebra rítmica / harmônica
    OUTRO = "OUTRO"                     # Encerramento / finalização

    @classmethod
    def from_str(cls, val: str) -> "SectionType":
        try:
            return cls(val.upper().strip())
        except Exception:
            return cls.UNKNOWN


@dataclass
class PatternOccurrence:
    """Instância concreta de onde e quando um padrão musical ocorreu na linha do tempo."""
    pattern_id: str                     # Ex: "P01", "P02"
    start_time: float                   # Início em segundos
    end_time: float                     # Término em segundos
    start_bar: int                      # Compasso inicial (1-indexado)
    end_bar: int                        # Compasso final
    key_context: str = "--"             # Tonalidade no momento da ocorrência
    confidence: float = 0.85            # Confiança da detecção
    chord_sequence: List[str] = field(default_factory=list) # Acordes absolutos tocados
    duration_bars: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "start_bar": self.start_bar,
            "end_bar": self.end_bar,
            "duration_bars": self.duration_bars,
            "key_context": self.key_context,
            "confidence": round(self.confidence, 2),
            "chord_sequence": list(self.chord_sequence),
        }


@dataclass
class MusicalPattern:
    """Padrão musical harmônico recorrente identificado pelo sistema."""
    id: str                             # Ex: "P01", "P02"
    duration_bars: int = 4              # Duração típica em compassos (ex: 4, 8)
    roman_numerals: List[str] = field(default_factory=list)      # Ex: ["I", "V", "vi", "IV"]
    relative_intervals: List[int] = field(default_factory=list)  # Semitons em relação à tônica: [0, 7, 9, 5]
    canonical_chords: List[str] = field(default_factory=list)    # Exemplo canônico (ex: ["C", "G", "Am", "F"])
    chord_signatures: Optional[List[str]] = None                 # Alias de conveniência para canonical_chords
    occurrences: List[PatternOccurrence] = field(default_factory=list)
    occurrence_count: int = 0
    associated_section_type: str = "UNKNOWN"
    expected_next_section: str = "UNKNOWN"
    confidence: float = 0.80

    def __post_init__(self):
        if self.chord_signatures is not None and not self.canonical_chords:
            self.canonical_chords = list(self.chord_signatures)
        elif self.canonical_chords and self.chord_signatures is None:
            self.chord_signatures = list(self.canonical_chords)

    @property
    def stability_score(self) -> float:
        """Estabilidade estimada do padrão com base na sua repetição."""
        return min(0.99, 0.70 + (self.occurrence_count * 0.10))

    def add_occurrence(self, occ: PatternOccurrence) -> None:
        self.occurrences.append(occ)
        self.occurrence_count = len(self.occurrences)
        # Refina confiança com base em repetição confirmada
        self.confidence = min(0.99, self.confidence + 0.05)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "roman_numerals": " - ".join(self.roman_numerals),
            "canonical_chords": " | ".join(self.canonical_chords),
            "duration_bars": self.duration_bars,
            "occurrence_count": self.occurrence_count,
            "associated_section_type": self.associated_section_type,
            "confidence": round(self.confidence, 2),
            "stability_score": round(self.stability_score, 2),
            "occurrences": [occ.to_dict() for occ in self.occurrences],
        }



@dataclass
class MusicSection:
    """Segmento estrutural formal da composição (ex: Verso 1, Refrão 2)."""
    id: str                             # Ex: "sec_01", "verse_01"
    section_type: str                   # "INTRO", "VERSE", "CHORUS", "UNKNOWN", etc.
    start_time: float
    end_time: float
    start_bar: int
    end_bar: int
    duration_bars: int
    chord_sequence: List[str]
    pattern_id: str = "P00"
    confidence: float = 0.70
    occurrence_index: int = 1           # Ocorrência #N deste tipo de seção (ex: 2 para Verso 2)
    occurrences: int = 1                # Total de vezes que essa seção aparece
    label: str = "A"                    # Rótulo de forma abstrata (A, B, C...)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "section_type": self.section_type,
            "type": self.section_type,
            "label": self.label,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "start_bar": self.start_bar,
            "end_bar": self.end_bar,
            "duration_bars": self.duration_bars,
            "chord_sequence": list(self.chord_sequence),
            "pattern_id": self.pattern_id,
            "occurrence_index": self.occurrence_index,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class SectionTransition:
    """Transição estatística observada entre seções ou padrões na música."""
    from_section: str                   # Ex: "VERSE"
    to_section: str                     # Ex: "CHORUS"
    from_pattern: str = "--"            # Ex: "P01"
    to_pattern: str = "--"              # Ex: "P02"
    count: int = 1                      # Quantidade de vezes observada
    probability: float = 1.0            # Probabilidade condicional P(to | from)
    average_duration: float = 0.0       # Duração média em segundos
    confidence: float = 0.80

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_section": self.from_section,
            "to_section": self.to_section,
            "from_pattern": self.from_pattern,
            "to_pattern": self.to_pattern,
            "count": self.count,
            "probability": round(self.probability, 2),
            "confidence": round(self.confidence, 2),
        }


@dataclass
class MusicPosition:
    """Posicionamento estrutural em tempo real na obra musical."""
    current_time: float = 0.0           # Posição em segundos
    current_bar: int = 1                # Compasso atual
    current_beat: int = 1               # Tempo atual
    current_section: str = "UNKNOWN"    # Nome da seção ativa
    section_progress: float = 0.0       # Progresso relativo na seção [0.0 a 1.0]
    pattern_id: str = "--"              # ID do padrão ativo
    pattern_occurrence_index: int = 1   # Índice desta ocorrência
    bar_in_section: int = 1             # Compasso dentro da seção ativa
    remaining_bars_in_section: int = 0  # Compassos restantes para o término da seção
    confidence: float = 0.50            # Confiança na estimativa de posição


    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_time": round(self.current_time, 3),
            "current_bar": self.current_bar,
            "current_beat": self.current_beat,
            "current_section": self.current_section,
            "section_progress": round(self.section_progress, 3),
            "pattern_id": self.pattern_id,
            "pattern_occurrence_index": self.pattern_occurrence_index,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class MusicStructure:
    """Representação estrutural global e completa de uma composição musical."""
    sections: List[MusicSection] = field(default_factory=list)
    patterns: Dict[str, MusicalPattern] = field(default_factory=dict)
    phrases: List[Dict[str, Any]] = field(default_factory=list)
    transitions: List[SectionTransition] = field(default_factory=list)
    current_position: MusicPosition = field(default_factory=MusicPosition)
    structure_sequence: List[str] = field(default_factory=list) # Ex: ["INTRO", "VERSE", "VERSE", "CHORUS"]
    abstract_sequence: List[str] = field(default_factory=list)  # Ex: ["A", "A", "B", "A", "B"]
    analysis_confidence: float = 0.0

    def get_section_at_bar(self, bar: int) -> Optional[MusicSection]:
        for sec in self.sections:
            if sec.start_bar <= bar <= sec.end_bar:
                return sec
        return None

    def get_section_at_time(self, time_sec: float) -> Optional[MusicSection]:
        for sec in self.sections:
            if sec.start_time <= time_sec <= sec.end_time:
                return sec
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_confidence": round(self.analysis_confidence, 2),
            "structure_sequence": list(self.structure_sequence),
            "abstract_sequence": list(self.abstract_sequence),
            "total_sections": len(self.sections),
            "total_patterns": len(self.patterns),
            "sections": [s.to_dict() for s in self.sections],
            "patterns": {pid: pat.to_dict() for pid, pat in self.patterns.items()},
            "transitions": [t.to_dict() for t in self.transitions],
            "current_position": self.current_position.to_dict(),
        }
