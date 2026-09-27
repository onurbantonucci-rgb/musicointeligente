"""Motor de Decisão Harmônica para o Baixista Virtual (BassDecisionEngine).

Responsável por:
1. Extrair a tônica e notas harmônicas do acorde atual (fundamental, quinta, terça, oitava).
2. Respeitar inversões de acordes (ex: G/B toca B).
3. Aplicar Voice Leading inteligente para minimizar saltos abruptos no braço do baixo.
4. Manter todas as notas dentro da extensão física do instrumento (E1 a G3, MIDI 28–55).
5. Considerar a tonalidade para identificação da escala diatônica.
"""

from typing import Optional, Tuple
from app.instruments.bass_model import BassDecision, BassPatternType
from app.music.musical_context import MusicalContext
from app.music.theory import PITCH_CLASSES, get_scale_notes
from app.music.constants import (
    BASS_MIN_MIDI_NOTE,
    BASS_MAX_MIDI_NOTE,
    BASS_DEFAULT_OCTAVE,
    MIN_BASS_CHORD_CONFIDENCE,
)


def note_name_to_midi(pitch_class: str, octave: int) -> int:
    """Converte nome de nota e oitava em número de nota MIDI."""
    if pitch_class not in PITCH_CLASSES:
        return 36
    return (octave + 1) * 12 + PITCH_CLASSES.index(pitch_class)


def midi_to_note_name(midi_num: int) -> str:
    """Converte número de nota MIDI em nome de nota com oitava (ex: 36 -> 'C2')."""
    p_idx = midi_num % 12
    octave = (midi_num // 12) - 1
    return f"{PITCH_CLASSES[p_idx]}{octave}"


class BassDecisionEngine:
    """Motor que analisa o contexto musical e decide as notas fundamentais e de apoio do baixo."""

    def __init__(
        self,
        min_midi: int = BASS_MIN_MIDI_NOTE,
        max_midi: int = BASS_MAX_MIDI_NOTE,
        default_octave: int = BASS_DEFAULT_OCTAVE
    ):
        self._min_midi = min_midi
        self._max_midi = max_midi
        self._default_octave = default_octave
        self._last_root_midi: Optional[int] = None

    def reset(self) -> None:
        """Reinicia o estado interno de voice leading."""
        self._last_root_midi = None

    def _choose_best_octave_for_root(self, pitch_class: str) -> int:
        """Escolhe a melhor oitava para a fundamental minimizando saltos (Voice Leading)."""
        candidates = []
        for oct_cand in [1, 2, 3]:
            m = note_name_to_midi(pitch_class, oct_cand)
            if self._min_midi <= m <= self._max_midi:
                candidates.append((m, oct_cand))

        if not candidates:
            # Fallback seguro
            return 36

        # Se já houver nota anterior, escolhe a que minimiza a distância intervalar
        if self._last_root_midi is not None:
            candidates.sort(key=lambda item: abs(item[0] - self._last_root_midi))
            return candidates[0][0]

        # Padrão de registro confortável para baixo elétrico:
        # Notas graves (E, F, F#, G, G#, A, A#, B) em oitava 1 (MIDI 28 a 35)
        # Notas médias (C, C#, D, D#) em oitava 2 (MIDI 36 a 39)
        p_idx = PITCH_CLASSES.index(pitch_class)
        if p_idx >= 4:  # E a B
            preferred = note_name_to_midi(pitch_class, 1)
        else:           # C a D#
            preferred = note_name_to_midi(pitch_class, 2)

        if self._min_midi <= preferred <= self._max_midi:
            return preferred

        return candidates[0][0]

    def _calculate_harmonic_intervals(self, root_midi: int, quality: str) -> Tuple[int, int, int]:
        """Calcula quinta justa, terça e oitava dentro dos limites físicos do baixo."""
        # 1. Quinta Justa (+7 semitons, ou -5 se estourar limite superior)
        if root_midi + 7 <= self._max_midi:
            fifth_midi = root_midi + 7
        elif root_midi - 5 >= self._min_midi:
            fifth_midi = root_midi - 5
        else:
            fifth_midi = root_midi

        # 2. Terça (Maior = +4, Menor = +3)
        q = quality.lower()
        third_interval = 3 if ("minor" in q or "m" in q and "maj" not in q) else 4
        if root_midi + third_interval <= self._max_midi:
            third_midi = root_midi + third_interval
        elif root_midi - (12 - third_interval) >= self._min_midi:
            third_midi = root_midi - (12 - third_interval)
        else:
            third_midi = root_midi

        # 3. Oitava (+12 ou -12)
        if root_midi + 12 <= self._max_midi:
            octave_midi = root_midi + 12
        elif root_midi - 12 >= self._min_midi:
            octave_midi = root_midi - 12
        else:
            octave_midi = root_midi

        return fifth_midi, third_midi, octave_midi

    def decide(
        self,
        context: MusicalContext,
        pattern_override: Optional[BassPatternType] = None
    ) -> BassDecision:
        """Determina as notas e parâmetros de baixo a partir do MusicalContext."""
        chord_symbol = context.chord if context.chord != "--" else "C"

        # Extração resiliente da fundamental a partir de context.chord_root ou context.chord
        root_name = context.chord_root
        if not root_name or root_name not in PITCH_CLASSES:
            clean = chord_symbol.split('/')[0]
            if len(clean) >= 2 and clean[1] in ['#', 'b']:
                cand = clean[:2]
            elif len(clean) >= 1:
                cand = clean[:1]
            else:
                cand = "C"
            root_name = cand if cand in PITCH_CLASSES else "C"

        # Qualidade harmônica
        quality = context.chord_quality
        if not quality or quality == "--":
            clean = chord_symbol.split('/')[0]
            if len(clean) > 1 and "m" in clean[1:] and "maj" not in clean:
                quality = "minor"
            else:
                quality = "major"

        # Respeitar inversão: se o acorde possui baixo específico (ex: G/B), o baixo toca o B
        if (pattern_override != BassPatternType.FUNDAMENTALS and
                context.inversion != "root" and context.bass_note in PITCH_CLASSES):
            effective_root = context.bass_note
            reason = f"Inversão do acorde (Baixo em {context.bass_note})"
        else:
            effective_root = root_name
            reason = "Fundamental do acorde"

        # Determina a fundamental com voice leading
        root_midi = self._choose_best_octave_for_root(effective_root)
        self._last_root_midi = root_midi
        root_note_str = midi_to_note_name(root_midi)

        # Determina intervalos harmônicos
        fifth_midi, third_midi, octave_midi = self._calculate_harmonic_intervals(root_midi, quality)
        fifth_note_str = midi_to_note_name(fifth_midi)
        third_note_str = midi_to_note_name(third_midi)
        octave_note_str = midi_to_note_name(octave_midi)

        # Escala da tonalidade atual
        scale_notes = []
        if context.key != "--" and len(context.key.split()) == 2:
            k_root, k_scale = context.key.split()
            scale_notes = get_scale_notes(k_root, k_scale)

        pat_type = pattern_override if pattern_override is not None else BassPatternType.AUTO

        # Fallback de segurança em caso de baixa confiança harmônica (< 0.40)
        if pat_type == BassPatternType.AUTO and context.chord_confidence < MIN_BASS_CHORD_CONFIDENCE:
            pat_type = BassPatternType.ROOT
            reason = f"Fallback por baixa confiança ({context.chord_confidence:.2f} < {MIN_BASS_CHORD_CONFIDENCE:.2f})"

        return BassDecision(
            chord=chord_symbol,
            root_note=root_note_str,
            root_midi=root_midi,
            fifth_note=fifth_note_str,
            fifth_midi=fifth_midi,
            third_note=third_note_str,
            third_midi=third_midi,
            octave_note=octave_note_str,
            octave_midi=octave_midi,
            scale_notes=scale_notes,
            confidence=context.chord_confidence,
            key=context.key,
            pattern_type=pat_type,
            reason=reason
        )
