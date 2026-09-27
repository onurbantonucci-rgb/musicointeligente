"""Definições fundamentais de Teoria Musical para o Virtual Band AI.

Tabelas de frequências padrão A440, mapeamentos de notas, intervalos e modelos de acordes.
"""

from typing import Tuple, List
import numpy as np

# Nomes das 12 classes de pitch na escala temperada
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Mapeamento de sustenidos e bemóis equivalentes (enarmonia)
ENHARMONIC_MAP = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
}


def note_to_pc(note: str) -> int:
    """Converte um nome de nota (com enarmonia) para a classe de pitch 0..11, ou -1 se inválido.

    Aceita 'C', 'C#', 'Db', 'F#', 'Bb' etc. Ignora sufixos após a tônica.
    """
    if not note or note in ("--", "N"):
        return -1
    n = note.strip()
    # Tônica = letra + acidente opcional
    if len(n) >= 2 and n[1] in ("#", "b"):
        root = n[0].upper() + n[1]
    else:
        root = n[0].upper()
    root = ENHARMONIC_MAP.get(root, root)
    if root in PITCH_CLASSES:
        return PITCH_CLASSES.index(root)
    return -1


def transpose_note(note: str, semitones: int) -> str:
    """Transpõe o nome de uma nota por um número de semitons (invariante a oitava)."""
    pc = note_to_pc(note)
    if pc < 0:
        return note
    return PITCH_CLASSES[(pc + semitones) % 12]


def hz_to_midi(freq: float) -> float:
    """Converte frequência em Hz para nota MIDI fracionária (A4 = 69 = 440 Hz)."""
    if freq <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def midi_to_hz(midi: float) -> float:
    """Converte número de nota MIDI para frequência em Hz."""
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def hz_to_note_name(freq: float) -> Tuple[str, int, float]:
    """Converte frequência em Hz para nome da nota, oitava e desvio em cents.
    
    Retorna: (nome_da_nota, oitava, cents_offset)
    Exemplo: 440.0 -> ("A", 4, 0.0)
             98.0  -> ("G", 2, +0.01)
    """
    if freq < 16.0 or np.isnan(freq) or np.isinf(freq):
        return ("--", 0, 0.0)

    midi_float = hz_to_midi(freq)
    nearest_midi = int(round(midi_float))
    cents = (midi_float - nearest_midi) * 100.0

    pitch_class_idx = nearest_midi % 12
    octave = (nearest_midi // 12) - 1

    note_name = PITCH_CLASSES[pitch_class_idx]
    return (note_name, octave, cents)


def get_chord_notes(root: str, quality: str) -> List[str]:
    """Retorna as notas componentes de um acorde."""
    if root not in PITCH_CLASSES:
        return []
    root_idx = PITCH_CLASSES.index(root)

    # Intervalos em semitons a partir da tônica
    intervals = {
        "major": [0, 4, 7],
        "minor": [0, 3, 7],
        "diminished": [0, 3, 6],
        "augmented": [0, 4, 8],
        "sus2": [0, 2, 7],
        "sus4": [0, 5, 7],
        "7": [0, 4, 7, 10],
        "maj7": [0, 4, 7, 11],
        "m7": [0, 3, 7, 10],
    }.get(quality.lower(), [0, 4, 7])

    return [PITCH_CLASSES[(root_idx + interval) % 12] for interval in intervals]


# Intervalos diatônicos em semitons (a partir da tônica)
SCALE_DEGREES_MAJOR = [0, 2, 4, 5, 7, 9, 11]
SCALE_DEGREES_MINOR = [0, 2, 3, 5, 7, 8, 10]


def get_scale_notes(root: str, scale_type: str = "Major") -> List[str]:
    """Retorna as 7 notas da escala Maior ou Menor natural."""
    if root not in PITCH_CLASSES:
        return []
    root_idx = PITCH_CLASSES.index(root)
    intervals = SCALE_DEGREES_MAJOR if "maj" in scale_type.lower() else SCALE_DEGREES_MINOR
    return [PITCH_CLASSES[(root_idx + interval) % 12] for interval in intervals]


def compute_chord_key_compatibility(
    chord_root: str,
    chord_quality: str,
    key_root: str,
    key_scale_type: str = "Major"
) -> float:
    """Calcula a compatibilidade harmônica de um acorde com uma tonalidade candidata (0.0 a 1.0).
    
    Graus diatônicos primários (I, IV, V) recebem pontuação máxima (~0.95 a 1.0).
    Graus diatônicos secundários (vi, ii, iii) recebem pontuação alta (~0.80 a 0.85).
    Acordes de empréstimo modal ou dominantes secundárias recebem pontuação moderada (~0.35 a 0.40).
    Acordes fora da escala recebem pontuação residual (0.05).
    """
    if chord_root not in PITCH_CLASSES or key_root not in PITCH_CLASSES:
        return 0.50

    root_diff = (PITCH_CLASSES.index(chord_root) - PITCH_CLASSES.index(key_root)) % 12
    q = chord_quality.lower()
    is_major_key = "maj" in key_scale_type.lower()

    if is_major_key:
        if root_diff == 0:     # I (Tônica)
            return 1.00 if q in ("major", "maj7", "7", "sus2", "sus4") else 0.20
        elif root_diff == 5:   # IV (Subdominante)
            return 0.95 if q in ("major", "maj7") else 0.35
        elif root_diff == 7:   # V (Dominante)
            return 0.95 if q in ("major", "7", "sus4") else 0.35
        elif root_diff == 9:   # vi (Relativo menor)
            return 0.85 if q in ("minor", "m7") else 0.35
        elif root_diff == 2:   # ii (Sobretônica)
            return 0.85 if q in ("minor", "m7") else 0.40
        elif root_diff == 4:   # iii (Mediante)
            return 0.80 if q in ("minor", "m7") else 0.35
        elif root_diff == 11:  # vii° (Sensível)
            return 0.65 if q in ("diminished", "m7b5") else 0.25
        elif root_diff == 10:  # bVII (Subtônica / Empréstimo mixolídio)
            return 0.40 if q in ("major", "7") else 0.15
        elif root_diff == 1:   # bII (Napolitano)
            return 0.25 if q in ("major", "maj7") else 0.05
        else:
            return 0.05
    else:
        # Escala Menor Natural / Harmônica
        if root_diff == 0:     # i (Tônica menor)
            return 1.00 if q in ("minor", "m7") else 0.30
        elif root_diff == 5:   # iv (Subdominante menor)
            return 0.95 if q in ("minor", "m7") else 0.35
        elif root_diff == 7:   # v/V (Dominante menor ou Maior harmônico)
            return 0.95 if q in ("minor", "major", "7") else 0.35
        elif root_diff == 3:   # III (Relativo maior)
            return 0.85 if q in ("major", "maj7") else 0.30
        elif root_diff == 8:   # VI (Submediante maior)
            return 0.85 if q in ("major", "maj7") else 0.30
        elif root_diff == 10:  # VII (Subtônica maior)
            return 0.80 if q in ("major", "7") else 0.30
        elif root_diff == 11:  # vii° (Sensível diminuta)
            return 0.65 if q in ("diminished", "m7b5") else 0.20
        elif root_diff == 2:   # ii° (Sobretônica diminuta)
            return 0.65 if q in ("diminished", "m7b5") else 0.35
        else:
            return 0.05

