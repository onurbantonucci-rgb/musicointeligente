"""Normalização Harmônica e Reconhecimento Transpositivo (v0.3).

Permite abstrair sequências de acordes absolutos (ex: C - G - Am - F e G - D - Em - C)
em suas representações estruturais relativas (numerais romanos: I - V - vi - IV e intervalos relativos),
garantindo invariância à transposição tonal e tolerância a ruídos de detecção.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from app.music.theory import PITCH_CLASSES, ENHARMONIC_MAP, get_chord_notes


# Mapeamento canônico de numerais romanos para escalas Maiores
ROMAN_MAJOR = {
    0: ("I", "i"),
    1: ("bII", "bii"),
    2: ("II", "ii"),
    3: ("bIII", "biii"),
    4: ("III", "iii"),
    5: ("IV", "iv"),
    6: ("bV", "bv"),
    7: ("V", "v"),
    8: ("bVI", "bvi"),
    9: ("VI", "vi"),
    10: ("bVII", "bvii"),
    11: ("VII", "vii°"),
}

# Mapeamento canônico de numerais romanos para escalas Menores
ROMAN_MINOR = {
    0: ("I", "i"),
    1: ("bII", "bii"),
    2: ("II", "ii°"),
    3: ("III", "iii"),
    4: ("#III", "#iii"),
    5: ("IV", "iv"),
    6: ("#IV", "#iv"),
    7: ("V", "v"),
    8: ("VI", "vi"),
    9: ("#VI", "#vi"),
    10: ("VII", "vii"),
    11: ("VII", "vii°"),
}


@dataclass
class ChordSignature:
    """Assinatura estrutural detalhada de um único acorde."""
    root: str                           # Tônica do acorde (ex: "C", "F#")
    quality: str                        # Qualidade ("major", "minor", "7", "maj7", etc.)
    bass_note: str                      # Nota no baixo (ex: "B" em "G/B")
    inversion: str                      # "root", "first", "second"
    relative_degree: int                # Distância em semitons a partir do centro tonal (0 a 11)
    roman_numeral: str                  # Numeral romano (ex: "I", "V", "vi", "IV")
    normalized_form: str                # Forma canônica única (ex: "I", "vi", "V/7")

    def to_dict(self) -> Dict:
        return {
            "root": self.root,
            "quality": self.quality,
            "bass_note": self.bass_note,
            "inversion": self.inversion,
            "relative_degree": self.relative_degree,
            "roman_numeral": self.roman_numeral,
            "normalized_form": self.normalized_form,
        }


@dataclass
class NormalizedChordSequence:
    """Sequência harmônica normalizada com propriedades relativas e temporais."""
    absolute_chords: List[str]          # Ex: ["C", "G", "Am", "F"]
    roman_numerals: List[str]           # Ex: ["I", "V", "vi", "IV"]
    relative_intervals: List[int]       # Ex: [0, 7, 9, 5]
    duration_per_chord: List[float] = field(default_factory=list) # Durações em segundos
    beat_positions: List[float] = field(default_factory=list)     # Tempos no compasso
    confidence: float = 0.85
    key_context: str = "C Major"

    def __len__(self) -> int:
        return len(self.roman_numerals)

    def to_signature_string(self) -> str:
        """Representação canônica serializável (ex: 'I-V-vi-IV')."""
        return "-".join(self.roman_numerals)

    def to_dict(self) -> Dict:
        return {
            "absolute_chords": list(self.absolute_chords),
            "roman_numerals": list(self.roman_numerals),
            "relative_intervals": list(self.relative_intervals),
            "signature": self.to_signature_string(),
            "confidence": round(self.confidence, 2),
            "key_context": self.key_context,
        }


def _clean_pitch_name(note: str) -> str:
    """Normaliza enarmonias para a convenção interna de sustenidos."""
    n = note.strip()
    return ENHARMONIC_MAP.get(n, n)


def parse_chord_signature(chord_symbol: str, key_str: str = "C Major") -> ChordSignature:
    """Converte um símbolo de cifra (ex: 'Am', 'G/B', 'F#7') em ChordSignature estruturado."""
    if not chord_symbol or chord_symbol == "--":
        return ChordSignature("C", "major", "C", "root", 0, "I", "I")

    # 1. Separar baixo em acordes invertidos (ex: "G/B")
    parts = chord_symbol.split('/')
    main_chord = parts[0].strip()
    bass_note = _clean_pitch_name(parts[1].strip()) if len(parts) > 1 else ""

    # 2. Extrair tônica e qualidade
    if len(main_chord) >= 2 and main_chord[1] in ('#', 'b'):
        root_raw = main_chord[:2]
        rest = main_chord[2:]
    else:
        root_raw = main_chord[:1]
        rest = main_chord[1:]

    root = _clean_pitch_name(root_raw)
    if root not in PITCH_CLASSES:
        root = "C"

    # Inferir qualidade
    rest_lower = rest.lower()
    if "dim" in rest_lower or "°" in rest:
        quality = "dim"
    elif "aug" in rest_lower or "+" in rest:
        quality = "augmented"
    elif "sus4" in rest_lower:
        quality = "sus4"
    elif "sus2" in rest_lower:
        quality = "sus2"
    elif "maj7" in rest_lower:
        quality = "maj7"
    elif "m7" in rest_lower or "min7" in rest_lower:
        quality = "m7"
    elif "7" in rest_lower:
        quality = "7"
    elif "m" in rest_lower and "maj" not in rest_lower:
        quality = "minor"
    else:
        quality = "major"

    # Inversão
    inversion = "root"
    if bass_note and bass_note != root:
        tones = get_chord_notes(root, "minor" if quality in ("minor", "m7") else "major")
        if len(tones) >= 2 and bass_note == tones[1]:
            inversion = "first"
        elif len(tones) >= 3 and bass_note == tones[2]:
            inversion = "second"
        elif len(tones) >= 4 and bass_note == tones[3]:
            inversion = "third"
        else:
            inversion = "slash"

    # 3. Extrair tônica e modo da tonalidade
    key_parts = key_str.strip().split()
    key_root = _clean_pitch_name(key_parts[0]) if key_parts else "C"
    key_mode = key_parts[1].lower() if len(key_parts) > 1 else "major"
    is_major = "maj" in key_mode

    if key_root not in PITCH_CLASSES:
        key_root = "C"

    root_idx = PITCH_CLASSES.index(root)
    key_idx = PITCH_CLASSES.index(key_root)
    relative_degree = (root_idx - key_idx) % 12

    # 4. Determinar numeral romano
    roman_map = ROMAN_MAJOR if is_major else ROMAN_MINOR
    maj_sym, min_sym = roman_map.get(relative_degree, ("I", "i"))

    is_minor_chord = quality in ("minor", "m7", "diminished")
    base_roman = min_sym if is_minor_chord else maj_sym

    # Adicionar sufixo de extensão se relevante
    suffix = ""
    if quality == "7":
        suffix = "7"
    elif quality == "maj7":
        suffix = "maj7"
    elif quality == "m7":
        suffix = "7"

    roman_numeral = f"{base_roman}{suffix}"
    normalized_form = roman_numeral

    return ChordSignature(
        root=root,
        quality=quality,
        bass_note=bass_note if bass_note else root,
        inversion=inversion,
        relative_degree=relative_degree,
        roman_numeral=roman_numeral,
        normalized_form=normalized_form
    )


def normalize_chord_sequence(
    chords: List[str],
    key_str: str = "C Major",
    durations: Optional[List[float]] = None,
    beat_positions: Optional[List[float]] = None,
    confidence: float = 0.85,
    tonal_center: Optional[str] = None
) -> NormalizedChordSequence:
    """Normaliza uma lista de acordes absolutos para sua representação transpositiva de graus romanos."""
    if tonal_center is not None:
        key_str = tonal_center

    if not chords:
        return NormalizedChordSequence([], [], [], [], [], 0.0, key_str)

    signatures = [parse_chord_signature(c, key_str) for c in chords]
    romans = [s.roman_numeral for s in signatures]
    intervals = [s.relative_degree for s in signatures]

    durs = durations if durations is not None else [2.0] * len(chords)
    beats = beat_positions if beat_positions is not None else [1.0] * len(chords)

    return NormalizedChordSequence(
        absolute_chords=list(chords),
        roman_numerals=romans,
        relative_intervals=intervals,
        duration_per_chord=list(durs),
        beat_positions=list(beats),
        confidence=confidence,
        key_context=key_str
    )


def sequence_similarity(
    seq_a: Any,
    seq_b: Any,
    key1: Optional[str] = None,
    key2: Optional[str] = None
) -> float:
    """Calcula a similaridade estrutural entre duas sequências harmônicas (0.0 a 1.0).
    
    Aceita sequências como NormalizedChordSequence ou listas de nomes de acordes (strings).
    A avaliação combina:
    1. Correspondência de numerais romanos / graus harmônicos relativos.
    2. Correspondência da curva de intervalos de transposição.
    3. Tolerância a acordes intermediários isolados (ruído de detecção).
    4. Proporção de duração rítmica.
    """
    if not isinstance(seq_a, NormalizedChordSequence):
        seq_a = normalize_chord_sequence(list(seq_a), key_str=key1 or "C Major")
    if not isinstance(seq_b, NormalizedChordSequence):
        seq_b = normalize_chord_sequence(list(seq_b), key_str=key2 or "C Major")

    if len(seq_a) == 0 or len(seq_b) == 0:
        return 0.0

    # 1. Correspondência exata de numerais romanos
    romans_a = seq_a.roman_numerals
    romans_b = seq_b.roman_numerals

    if romans_a == romans_b:
        return 1.00

    # 2. Correspondência exata de intervalos relativos (transposição direta com mesma tônica relativa)
    if seq_a.relative_intervals == seq_b.relative_intervals:
        return 1.00

    # 2b. Transposição em qualquer semitom caso tonalidades relativas não tenham sido alinhadas
    if len(seq_a.relative_intervals) == len(seq_b.relative_intervals) and len(seq_a.relative_intervals) > 0:
        deltas = set((a - b) % 12 for a, b in zip(seq_a.relative_intervals, seq_b.relative_intervals))
        if len(deltas) == 1:
            sig_a = [parse_chord_signature(c) for c in seq_a.absolute_chords]
            sig_b = [parse_chord_signature(c) for c in seq_b.absolute_chords]
            if [s.quality for s in sig_a] == [s.quality for s in sig_b]:
                return 1.00

    # 3. Similaridade por alinhamento de sequência (Levenshtein / Needleman-Wunsch adaptado)
    m = len(romans_a)
    n = len(romans_b)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            deg_a = seq_a.relative_intervals[i - 1]
            deg_b = seq_b.relative_intervals[j - 1]
            rom_a = romans_a[i - 1]
            rom_b = romans_b[j - 1]

            if rom_a == rom_b:
                match_score = 1.0
            elif deg_a == deg_b:
                match_score = 0.90
            elif abs(deg_a - deg_b) in (5, 7): # Relação de quarta / quinta justa
                match_score = 0.50
            else:
                match_score = 0.0

            dp[i][j] = max(
                dp[i - 1][j - 1] + match_score,
                dp[i - 1][j] - 0.2, # Deleção suave (tolerância a acorde extra de ruído)
                dp[i][j - 1] - 0.2  # Inserção suave
            )

    raw_score = dp[m][n]
    max_possible = max(m, n)
    normalized_align_score = max(0.0, min(1.0, raw_score / max_possible))

    return round(float(normalized_align_score), 3)
