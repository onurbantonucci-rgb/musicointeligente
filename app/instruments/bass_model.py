"""Modelos de dados e estruturas para a performance do Baixista Virtual (v0.2)."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
import numpy as np


class BassPatternType(Enum):
    """Padrões rítmicos disponíveis para o Baixista Virtual."""
    AUTO = "AUTO"                               # Adaptação automática baseada em andamento e harmonia
    FUNDAMENTALS = "FUNDAMENTALS"               # Apenas tônicas, na subdivisão escolhida pelo usuário
    ROOT = "ROOT"                               # Apenas fundamental nos tempos principais
    ROOT_FIFTH = "ROOT + FIFTH"                 # Fundamental e quinta alternadas (clássico pop/rock)
    ROOT_FIFTH_OCTAVE = "ROOT + FIFTH + OCTAVE" # Fundamental, quinta e oitava
    SUSTAINED = "SUSTAINED"                     # Fundamental sustentada ao longo do compasso/acorde


class BassNoteValue(Enum):
    """Distância entre ataques, expressa em tempos de semínima."""
    WHOLE = 4.0
    HALF = 2.0
    QUARTER = 1.0
    EIGHTH = 0.5
    SIXTEENTH = 0.25

    @property
    def beats(self) -> float:
        return float(self.value)


class BassHarmonySource(Enum):
    """Fonte harmônica da nota; o relógio continua responsável pelo ataque."""
    FOLLOW = "FOLLOW"
    CHART = "CHART"


@dataclass
class BassNoteEvent:
    """Evento discreto de nota executada pelo Baixista Virtual."""
    note: str                           # Nome da nota com oitava (ex: "C2", "G1")
    midi_note: int                      # Altura em número MIDI (ex: 36 para C2)
    start_time: float                   # Momento de início da nota em segundos
    duration: float                     # Duração da nota em segundos
    velocity: int = 100                 # Intensidade da palhetada (0 a 127)
    beat: int = 1                       # Tempo do compasso em que a nota ocorre (1, 2, 3, 4)
    bar: int = 1                        # Número do compasso
    confidence: float = 0.90            # Grau de certeza harmônica da escolha
    reason: str = "chord_root"          # Justificativa musical ("chord_root", "chord_fifth", "chord_third", "octave", etc.)

    def to_dict(self) -> Dict:
        return {
            "time_str": self.format_timestamp(self.start_time),
            "bar": self.bar,
            "beat": self.beat,
            "note": self.note,
            "midi_note": self.midi_note,
            "duration": round(self.duration, 3),
            "velocity": self.velocity,
            "reason": self.reason,
            "confidence": round(self.confidence, 2),
        }

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        if seconds < 0 or np.isnan(seconds):
            return "00:00.00"
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m:02d}:{s:05.2f}"


@dataclass
class BassDecision:
    """Decisão harmônica tomada pelo BassDecisionEngine para o acorde e contexto atual."""
    chord: str                          # Acorde de referência (ex: "C", "Am", "G/B")
    root_note: str                      # Nota fundamental no registro de baixo (ex: "C2")
    root_midi: int                      # MIDI da fundamental (ex: 36)
    fifth_note: str = "--"              # Quinta justa (ex: "G1" ou "G2")
    fifth_midi: int = 0
    third_note: str = "--"              # Terça maior ou menor (ex: "E2" ou "Eb2")
    third_midi: int = 0
    octave_note: str = "--"             # Oitava superior ou inferior (ex: "C3")
    octave_midi: int = 0
    scale_notes: List[str] = field(default_factory=list) # Notas diatônicas da escala do tom
    confidence: float = 0.0             # Confiança da decisão
    key: str = "--"                     # Tonalidade considerada
    pattern_type: BassPatternType = BassPatternType.AUTO
    reason: str = "Fundamental do acorde" # Motivo da decisão harmônica

    @property
    def root_note_name(self) -> str:
        """Alias para conveniência e compatibilidade com a UI."""
        return self.root_note

    @property
    def root_midi_note(self) -> int:
        """Alias para conveniência e compatibilidade com a UI."""
        return self.root_midi
