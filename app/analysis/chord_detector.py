"""Detector de Acordes com suporte a Tríades Maiores, Menores, Inversões e Histórico."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np

from app.music.theory import (
    PITCH_CLASSES,
    hz_to_note_name,
    note_to_pc,
    compute_chord_key_compatibility,
)


def _is_minor_symbol(symbol: str) -> bool:
    """Heurística simples: o símbolo denota um acorde menor (m, min) e não maj/dim."""
    if not symbol:
        return False
    s = symbol.split("/")[0]  # ignora o baixo da inversão
    low = s.lower()
    if "maj" in low or "dim" in low or "sus" in low or "aug" in low:
        return False
    # 'm' após a tônica (ex.: Am, F#m, Cm7), mas não 'M' de maj
    body = s[2:] if (len(s) >= 2 and s[1] in ("#", "b")) else s[1:]
    return body.startswith("m")


def _parse_key_hint(key: str):
    """Extrai (tônica, tipo_de_escala) de uma string de tom como 'C Major' ou 'Am'."""
    if not key or key in ("--", "N"):
        return None
    k = key.strip()
    scale = "Major"
    low = k.lower()
    if "min" in low or low.endswith("m") and "maj" not in low:
        scale = "Minor"
    if "maj" in low:
        scale = "Major"
    # tônica = primeira letra + acidente opcional
    if len(k) >= 2 and k[1] in ("#", "b"):
        root = k[0].upper() + k[1]
    else:
        root = k[0].upper()
    from app.music.theory import ENHARMONIC_MAP
    root = ENHARMONIC_MAP.get(root, root)
    if root not in PITCH_CLASSES:
        return None
    return root, scale


@dataclass
class Chord:
    """Representação rica de um acorde musical detectado."""
    root: str = "--"                            # Tônica do acorde (ex: "G")
    quality: str = "--"                         # Qualidade (ex: "major", "minor", "7", "sus4")
    symbol: str = "--"                          # Cifra completa (ex: "G", "Em", "G/B")
    bass_note: str = "--"                       # Nota mais grave (essencial para o baixista)
    inversion: str = "root"                     # "root" (fundamental), "first" (1ª inv), "second" (2ª inv)
    detected_notes: List[str] = field(default_factory=list)  # Notas identificadas (ex: ["G", "B", "D"])
    confidence: float = 0.0                     # Grau de certeza harmônica (0.0 a 1.0)
    timestamp: float = 0.0                      # Momento do início do acorde em segundos
    duration: float = 0.0                       # Duração acumulada em segundos


# Re-exportado (não usado neste módulo): harmonic_analyzer.py, app/analysis/__init__.py
# e tests/test_chords.py importam ChordHistory/ChordEvent a partir DAQUI. Não remover.
from app.analysis.chord_history import ChordHistory, ChordEvent


class ChordDetector:
    """Detector de acordes baseado em templates harmônicos de cromagrama e análise de graves."""

    def __init__(self, detect_extensions: bool = False):
        self._templates = self._build_templates()
        self._smoothed_chroma = np.zeros(12, dtype=np.float32)
        self._last_timestamp: Optional[float] = None
        self._alpha = 0.45  # Suavização temporal (EMA): menor = mais responsivo (menos atraso)
        # Quando habilitado, refina a tríade detectada em acordes de tétrade/suspensos
        # (7, 7M, m7, sus4, sus2) analisando a energia das notas de tensão no cromagrama.
        # Mantido desligado por padrão para preservar a detecção base de tríades pura.
        self._detect_extensions = detect_extensions

    def reset(self) -> None:
        """Limpa memória de frames sem alterar o modo de análise configurado."""
        self._smoothed_chroma.fill(0.0)
        self._last_timestamp = None

    def _build_templates(self) -> Dict[str, Tuple[np.ndarray, str, str, List[str]]]:
        """Constrói perfis harmônicos de referência para as 12 tonalidades."""
        templates = {}
        for i, root in enumerate(PITCH_CLASSES):
            # 1. Tríades Maiores (Tônica, 3ª Maior, 5ª Justa)
            maj = np.zeros(12, dtype=np.float32)
            maj[i] = 1.0
            maj[(i + 4) % 12] = 0.85
            maj[(i + 7) % 12] = 0.90
            templates[root] = (
                maj / np.linalg.norm(maj),
                root,
                "major",
                [root, PITCH_CLASSES[(i + 4) % 12], PITCH_CLASSES[(i + 7) % 12]]
            )

            # 2. Tríades Menores (Tônica, 3ª Menor, 5ª Justa)
            min_tpl = np.zeros(12, dtype=np.float32)
            min_tpl[i] = 1.0
            min_tpl[(i + 3) % 12] = 0.85
            min_tpl[(i + 7) % 12] = 0.90
            templates[f"{root}m"] = (
                min_tpl / np.linalg.norm(min_tpl),
                root,
                "minor",
                [root, PITCH_CLASSES[(i + 3) % 12], PITCH_CLASSES[(i + 7) % 12]]
            )

        return templates

    def detect(self, chroma: np.ndarray, audio_chunk: np.ndarray, sample_rate: int,
               timestamp: float = 0.0, expected_chord: Optional[str] = None,
               key: Optional[str] = None, prior_weight: float = 0.08) -> Chord:
        """Compara o cromagrama com os templates harmônicos e analisa a nota do baixo.

        Se ``expected_chord`` (da cifra) ou ``key`` forem fornecidos, aplica um *prior*
        musical: um pequeno bônus de pontuação a candidatos coerentes com a expectativa.
        O bônus só desempata casos harmonicamente ambíguos (ex.: Cmaj7 vs Em) — não
        fabrica confiança nem sobrepõe uma evidência de áudio forte. A CONFIANÇA reportada
        continua baseada na similaridade de cosseno bruta, não no escore enviesado.
        """
        if len(chroma) < 12 or np.max(chroma) < 0.05:
            self._smoothed_chroma.fill(0.0)
            self._last_timestamp = timestamp
            return Chord(timestamp=timestamp)

        if (self._last_timestamp is not None and
                (timestamp < self._last_timestamp or timestamp - self._last_timestamp > 0.8)):
            self._smoothed_chroma.fill(0.0)
        self._last_timestamp = timestamp

        # Suavização temporal do cromagrama
        self._smoothed_chroma = (self._alpha * self._smoothed_chroma) + ((1.0 - self._alpha) * chroma)
        norm_c = float(np.linalg.norm(self._smoothed_chroma))

        if norm_c < 1e-4:
            return Chord(timestamp=timestamp)

        c_unit = self._smoothed_chroma / norm_c
        relative = self._smoothed_chroma / max(1e-9, float(np.max(self._smoothed_chroma)))
        # Uma única classe forte é pitch, não uma tríade. Esperar outras notas
        # (simultâneas ou reunidas de um arpejo) evita que o template complete
        # por conta própria os intervalos que o áudio não sustentou.
        if int(np.count_nonzero(relative >= 0.22)) < 2:
            return Chord(timestamp=timestamp)

        # Prior musical (opcional): coerência com a cifra esperada e/ou o tom.
        exp_pc = note_to_pc(expected_chord) if expected_chord else -1
        exp_is_minor = bool(expected_chord) and _is_minor_symbol(expected_chord)
        key_info = _parse_key_hint(key) if key else None

        # O prior só participa entre candidatos realmente ambíguos no áudio.
        # Assim uma expectativa errada da cifra não consegue vencer um template
        # cuja similaridade bruta é claramente superior.
        raw_scores = {
            name: float(np.dot(c_unit, tpl))
            for name, (tpl, _root, _qual, _notes) in self._templates.items()
        }
        raw_best = max(raw_scores.values(), default=0.0)
        ambiguity_margin = max(0.03, prior_weight * 0.75)

        # Encontrar template com maior escore (cosseno + prior), guardando o cosseno bruto
        best_adj = -1.0
        best_score = 0.0          # cosseno bruto do vencedor (para a confiança)
        best_name = "--"
        best_root = "--"
        best_quality = "--"
        best_notes: List[str] = []

        for name, (tpl, root, qual, notes) in self._templates.items():
            cos = raw_scores[name]
            bonus = 0.0
            prior_is_eligible = raw_best - cos <= ambiguity_margin
            if prior_is_eligible and exp_pc >= 0 and PITCH_CLASSES.index(root) == exp_pc:
                # Bônus por coincidir com a tônica esperada (+ extra se a qualidade casa)
                bonus += prior_weight
                if (qual == "minor") == exp_is_minor:
                    bonus += prior_weight * 0.5
            elif prior_is_eligible and key_info is not None:
                # Sem acorde esperado direto: enviesa levemente para o diatônico do tom
                compat = compute_chord_key_compatibility(root, qual, key_info[0], key_info[1])
                bonus += prior_weight * 0.5 * compat
            adj = cos + bonus
            if adj > best_adj:
                best_adj = adj
                best_score = cos
                best_name = name
                best_root = root
                best_quality = qual
                best_notes = notes

        if best_score < 0.40:
            return Chord(timestamp=timestamp)

        # Refinamento opcional da tríade em tétrade/suspenso (7, 7M, m7, sus4, sus2)
        if self._detect_extensions:
            best_name = self._refine_extension(best_root, best_quality, best_name)

        # Detecção da nota de baixo (inversão)
        bass_note, inversion, full_symbol = self._detect_inversion(
            audio_chunk, sample_rate, best_root, best_quality, best_name, best_notes
        )

        return Chord(
            root=best_root,
            quality=best_quality,
            symbol=full_symbol,
            bass_note=bass_note,
            inversion=inversion,
            detected_notes=best_notes,
            confidence=float(np.clip(best_score, 0.0, 1.0)),
            timestamp=timestamp,
            duration=0.05
        )

    def _refine_extension(self, root: str, quality: str, base_symbol: str) -> str:
        """Refina a tríade em acorde suspenso, dominante ou tétrade (7/7M/m7).

        Mede a energia relativa das notas de tensão frente à tônica, no cromagrama suavizado.
        A 7ª MAIOR (B sobre C) e a 7ª da menor (G sobre Am) coincidem com o 3º harmônico
        da 3ª do acorde: no cromagrama por-FFT elas "vazam" e o limiar alto (``SEVENTH``)
        as barra; no cromagrama por SOMA HARMÔNICA esse vazamento é suprimido, então a
        presença real dessas notas ultrapassa o limiar e a tétrade é reconhecida.
        """
        if root not in PITCH_CLASSES:
            return base_symbol

        c = self._smoothed_chroma
        peak = float(np.max(c))
        if peak <= 1e-6:
            return base_symbol
        rel = c / peak  # energia relativa [0..1] por classe de pitch

        i = PITCH_CLASSES.index(root)
        third_energy = rel[(i + 4) % 12] if quality == "major" else rel[(i + 3) % 12]
        min7 = rel[(i + 10) % 12]
        maj7 = rel[(i + 11) % 12]
        second = rel[(i + 2) % 12]
        fourth = rel[(i + 5) % 12]

        SUS_TENSION = 0.50   # presença clara da 2ª/4ª para acorde suspenso
        DOM_TENSION = 0.30   # 7ª menor sobre maior é limpa mesmo no cromagrama FFT
        SEVENTH = 0.50       # 7ª maior / 7ª da menor: só cruza este limiar com chroma limpo
        WEAK_THIRD = 0.35    # 3ª "ausente" (indício de acorde suspenso)

        # 1. Acordes suspensos: a 3ª está ausente e a 2ª ou 4ª está presente
        if third_energy < WEAK_THIRD:
            if fourth >= SUS_TENSION and fourth >= second:
                return f"{root}sus4"
            if second >= SUS_TENSION:
                return f"{root}sus2"

        # 2. Sétimas (tétrades)
        if quality == "major":
            if maj7 >= SEVENTH and maj7 >= min7:
                return f"{root}7M"      # maior com 7ª maior (ex.: C7M)
            if min7 >= DOM_TENSION:
                return f"{root}7"       # dominante (ex.: G7)
        elif quality == "minor":
            if min7 >= SEVENTH:
                return f"{base_symbol}7"  # menor com 7ª (ex.: Am7)

        return base_symbol

    def _detect_inversion(self, audio_chunk: np.ndarray, sample_rate: int,
                          root: str, quality: str, base_symbol: str, chord_notes: List[str]) -> Tuple[str, str, str]:
        """Detecta a nota mais grave no espectro de baixa frequência para identificar inversões (ex: G/B)."""
        if len(audio_chunk) < 512 or sample_rate <= 0:
            return root, "root", base_symbol

        # Filtro passa-baixa espectral (< 250 Hz) para focar na região do contrabaixo/bordão
        n_fft = max(4096, 2 ** int(np.ceil(np.log2(len(audio_chunk)))))
        xw = audio_chunk * np.hanning(len(audio_chunk))
        mag = np.abs(np.fft.rfft(xw, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

        bass_mask = (freqs >= 40.0) & (freqs <= 240.0)
        if not np.any(bass_mask):
            return root, "root", base_symbol

        bass_mag = mag[bass_mask]
        bass_freqs = freqs[bass_mask]

        peak_idx = int(np.argmax(bass_mag))
        strongest_bass_freq = float(bass_freqs[peak_idx])

        # Se o pico for relevante, converte para nota
        if bass_mag[peak_idx] > 0.05 * np.max(mag):
            note, _, _ = hz_to_note_name(strongest_bass_freq)
            if note in chord_notes:
                if note == root:
                    return root, "root", base_symbol
                elif len(chord_notes) >= 2 and note == chord_notes[1]:
                    # Primeira inversão (baixo na terça, ex: G/B)
                    return note, "first", f"{base_symbol}/{note}"
                elif len(chord_notes) >= 3 and note == chord_notes[2]:
                    # Segunda inversão (baixo na quinta, ex: G/D)
                    return note, "second", f"{base_symbol}/{note}"
                else:
                    return note, "inversion", f"{base_symbol}/{note}"

        return root, "root", base_symbol
