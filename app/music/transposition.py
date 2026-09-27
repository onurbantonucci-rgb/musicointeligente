"""Transposição de Cifras (Transposition v0.6).

Transpõe acordes e o texto de uma cifra por um número de semitons, ou de um tom para
outro, preservando a QUALIDADE, EXTENSÕES, ALTERAÇÕES e INVERSÕES de cada acorde e
re-grafando as notas conforme a convenção do tom-alvo (sustenidos ou bemóis).

Exemplos:
    "G"        + 2 semitons -> "A"
    "Em7"      + 2 semitons -> "F#m7"
    "F#m7(b5)" + 1 semitom  -> "Gm7(b5)"
    "C/E"      + 5 semitons -> "F/A"
    "Dsus4"    - 2 semitons -> "Csus4"
"""

import re
from typing import Tuple

from app.music.theory import PITCH_CLASSES, ENHARMONIC_MAP, note_to_pc

# Grafias por convenção
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Tônicas (maiores) que, por convenção, usam bemóis na armadura
_FLAT_MAJOR_TONICS = {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"}
# Tônicas (menores) que, por convenção, usam bemóis
_FLAT_MINOR_TONICS = {"D", "G", "C", "F", "Bb", "Eb", "Ab"}

# Tônica no início de um acorde: letra A-G + acidente opcional
_ROOT_RE = re.compile(r'^([A-G])([#b]?)')
# Reaproveita a regex de acorde do classificador de forma preguiçosa (evita ciclo de import)
_CHORD_REGEX = None
_classify_line = None
_SemanticLineType = None


def parse_key_root_mode(key: str) -> Tuple[str, str]:
    """Extrai (tônica, modo) de um tom como 'C Major', 'Am', 'F# Minor'. Modo: 'major'/'minor'."""
    if not key:
        return "C", "major"
    k = key.strip()
    low = k.lower()
    # tônica = letra + acidente opcional
    if len(k) >= 2 and k[1] in ("#", "b"):
        root = k[0].upper() + k[1]
        rest = k[2:]
    else:
        root = k[0].upper()
        rest = k[1:]
    root = ENHARMONIC_MAP.get(root, root)
    if root not in PITCH_CLASSES:
        root = "C"
    rest_low = rest.strip().lower()
    if "min" in low or rest_low.startswith("m") and "maj" not in low:
        mode = "minor"
    else:
        mode = "major"
    return root, mode


def key_uses_flats(key: str) -> bool:
    """Decide se o tom-alvo deve ser grafado com bemóis (True) ou sustenidos (False)."""
    root, mode = parse_key_root_mode(key)
    # A grafia explícita do nome do tom (acidente na tônica) tem prioridade
    tonic_token = key.split()[0] if key and key.split() else ""
    accidental = tonic_token[1:2]  # '#', 'b' ou ''
    if accidental == "b":
        return True
    if accidental == "#":
        return False
    if mode == "minor":
        return root in _FLAT_MINOR_TONICS
    return root in _FLAT_MAJOR_TONICS


def semitones_between_keys(from_key: str, to_key: str) -> int:
    """Semitons (0..11) da tônica de ``from_key`` até a de ``to_key``."""
    fr, _ = parse_key_root_mode(from_key)
    to, _ = parse_key_root_mode(to_key)
    return (PITCH_CLASSES.index(to) - PITCH_CLASSES.index(fr)) % 12


def transpose_note_name(note: str, semitones: int, use_flats: bool = False) -> str:
    """Transpõe um nome de nota isolado, re-grafando conforme sustenido/bemol."""
    pc = note_to_pc(note)
    if pc < 0:
        return note
    names = FLAT_NAMES if use_flats else SHARP_NAMES
    return names[(pc + semitones) % 12]


def transpose_chord_symbol(symbol: str, semitones: int, use_flats: bool = False) -> str:
    """Transpõe a grafia de um acorde preservando qualidade/extensão/alteração e inversão."""
    if not symbol or symbol.strip() in ("--", "N"):
        return symbol
    sym = symbol.strip()

    def shift(token: str) -> str:
        m = _ROOT_RE.match(token)
        if not m:
            return token
        root = m.group(1) + m.group(2)
        pc = note_to_pc(root)
        if pc < 0:
            return token
        names = FLAT_NAMES if use_flats else SHARP_NAMES
        return names[(pc + semitones) % 12] + token[m.end():]

    parts = sym.split('/')
    new_chord = shift(parts[0])
    if len(parts) > 1 and parts[1].strip():
        return f"{new_chord}/{shift(parts[1].strip())}"
    return new_chord


def _ensure_classifier():
    """Carrega, sob demanda, a regex de acorde e o classificador de linhas."""
    global _CHORD_REGEX, _classify_line, _SemanticLineType
    if _CHORD_REGEX is None:
        from app.input.chart_semantic_classifier import (
            CHORD_REGEX,
            ChartSemanticClassifier,
            SemanticLineType,
        )
        _CHORD_REGEX = CHORD_REGEX
        _classify_line = ChartSemanticClassifier
        _SemanticLineType = SemanticLineType


def transpose_text_block(text: str, semitones: int, use_flats: bool = False) -> str:
    """Transpõe apenas os acordes das LINHAS DE ACORDE de um texto de cifra.

    Linhas de letra, metadados, seções e tablatura permanecem intactas.
    """
    if not text or semitones % 12 == 0:
        return text
    _ensure_classifier()
    out_lines = []
    for line in text.split('\n'):
        cl = _classify_line.classify_line(line)
        if cl.line_type == _SemanticLineType.CHORD:
            def _repl(m):
                tok = m.group(1)
                if _classify_line.is_chord_shaped(tok):
                    return transpose_chord_symbol(tok, semitones, use_flats)
                return tok
            out_lines.append(_CHORD_REGEX.sub(_repl, line))
        elif cl.line_type == _SemanticLineType.METADATA and cl.extracted_metadata.get("key"):
            # Atualiza a tônica na linha "Tom:/Tonalidade:/Key:"
            def _rep_key(m):
                prefix, note = m.group(1), m.group(2)
                return f"{prefix}{transpose_note_name(note, semitones, use_flats)}"
            out_lines.append(re.sub(
                r'(^\s*(?:tom|tonalidade|key)\s*[:=]\s*)([A-G][#b]?)',
                _rep_key, line, flags=re.IGNORECASE
            ))
        else:
            out_lines.append(line)
    return '\n'.join(out_lines)
