"""Modelos de Cifra, Acordes Estruturados e Letras (ChordChart v0.4).

Preserva a informação completa de acordes complexos, extensões, alterações,
inversões, graus harmônicos, sinônimos, repetições e segmentos de letra.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING
import re

if TYPE_CHECKING:  # evita ciclo de import (key_timeline usa theory/transposition)
    from app.music.key_timeline import KeyChangeEvent, KeyTimeline

from app.music.theory import PITCH_CLASSES, ENHARMONIC_MAP, get_chord_notes
from app.music.harmonic_normalization import ROMAN_MAJOR, ROMAN_MINOR


# Tabela canônica de sinônimos de qualidade/extensão
CHORD_SYNONYMS = {
    "7M": "maj7",
    "Δ7": "maj7",
    "Δ": "maj7",
    "M7": "maj7",
    "maj7": "maj7",
    "°": "dim",
    "dim": "dim",
    "dim7": "dim7",
    "+": "aug",
    "aug": "aug",
    "-": "m",
    "min": "m",
    "m": "m",
}


@dataclass
class ChordSymbol:
    """Representação rica de um acorde contendo estrutura teórica completa e grafia original."""
    original_symbol: str               # Ex: "C7M", "Cmaj7", "G/B", "F#m7(b5)"
    root: str                          # Ex: "C", "F#"
    quality: str = "major"             # "major", "minor", "dim", "aug", "sus2", "sus4"
    extension: str = ""                # "7", "maj7", "9", "maj9", "m7", "m9", "11", "13", "add9", "m7(b5)"
    alterations: List[str] = field(default_factory=list) # ["b9", "#9", "b5", "#5", "b13"]
    bass_note: str = ""                # Ex: "B" para "G/B", "E" para "C/E"
    inversion: str = "root"            # "root", "first", "second", "third", "slash"
    harmonic_degree: str = "I"         # Grau na tonalidade: "I", "vi", "ii", "V"
    roman_numeral: str = "I"           # "I7M", "vi7", "ii9", "V7"
    confidence: float = 1.0

    @property
    def canonical_name(self) -> str:
        """Nome canônico unificado do acorde para agrupamento de sinônimos."""
        ext = self.extension
        # Converte sinônimos para forma canônica
        if ext in ("7M", "Δ7", "Δ", "M7"):
            ext = "maj7"
        elif ext in ("°",):
            ext = "dim"
        elif ext in ("+",):
            ext = "aug"

        base = f"{self.root}"
        if self.quality == "minor" and not ext.startswith("m"):
            base += "m"
        elif self.quality == "dim" and not "dim" in ext:
            base += "dim"
        elif self.quality == "aug" and not "aug" in ext:
            base += "aug"

        if ext:
            base += ext
        if self.alterations:
            base += f"({','.join(self.alterations)})"
        if self.bass_note and self.bass_note != self.root:
            base += f"/{self.bass_note}"
        return base

    def transposed(self, semitones: int, use_flats: bool = False,
                   key_context: str = "C Major") -> "ChordSymbol":
        """Retorna uma cópia deste acorde transposta por ``semitones``, re-grafada.

        Preserva qualidade, extensão, alterações e inversão; re-deriva grau/numeral
        romano no novo ``key_context``.
        """
        from app.music.transposition import transpose_chord_symbol
        new_text = transpose_chord_symbol(self.original_symbol, semitones, use_flats)
        new_sym = parse_chord(new_text, key_context)
        new_sym.confidence = self.confidence
        return new_sym

    def is_synonym_of(self, other: "ChordSymbol") -> bool:
        """Verifica se dois acordes representam o mesmo conceito harmônico."""
        if self.root != other.root:
            return False
        if (self.bass_note or self.root) != (other.bass_note or other.root):
            return False
        return self.canonical_name == other.canonical_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_symbol": self.original_symbol,
            "canonical_name": self.canonical_name,
            "root": self.root,
            "quality": self.quality,
            "extension": self.extension,
            "alterations": list(self.alterations),
            "bass_note": self.bass_note if self.bass_note else self.root,
            "inversion": self.inversion,
            "harmonic_degree": self.harmonic_degree,
            "roman_numeral": self.roman_numeral,
            "confidence": round(self.confidence, 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChordSymbol":
        return cls(
            original_symbol=data.get("original_symbol", "--"),
            root=data.get("root", "C"),
            quality=data.get("quality", "major"),
            extension=data.get("extension", ""),
            alterations=list(data.get("alterations", [])),
            bass_note=data.get("bass_note", ""),
            inversion=data.get("inversion", "root"),
            harmonic_degree=data.get("harmonic_degree", "I"),
            roman_numeral=data.get("roman_numeral", "I"),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class LyricSegment:
    """Fragmento de letra musical associado a uma posição ou acorde da cifra."""
    text: str
    chord_symbol: str = ""
    bar_offset: int = 1
    beat_offset: float = 1.0
    section_id: str = ""
    line_number: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "chord_symbol": self.chord_symbol,
            "bar_offset": self.bar_offset,
            "beat_offset": self.beat_offset,
            "section_id": self.section_id,
            "line_number": self.line_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LyricSegment":
        return cls(
            text=data.get("text", ""),
            chord_symbol=data.get("chord_symbol", ""),
            bar_offset=int(data.get("bar_offset", 1)),
            beat_offset=float(data.get("beat_offset", 1.0)),
            section_id=data.get("section_id", ""),
            line_number=int(data.get("line_number", 1)),
        )


@dataclass
class ChartChord:
    """Instância de um acorde posicionado no tempo musical da cifra."""
    symbol: ChordSymbol
    bar: int = 1
    beat: float = 1.0
    duration_bars: float = 1.0
    lyric: str = ""
    line_number: int = 1
    column: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol.to_dict(),
            "bar": self.bar,
            "beat": self.beat,
            "duration_bars": self.duration_bars,
            "lyric": self.lyric,
            "line_number": self.line_number,
            "column": self.column,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChartChord":
        sym_dict = data.get("symbol", {})
        return cls(
            symbol=ChordSymbol.from_dict(sym_dict) if isinstance(sym_dict, dict) else parse_chord(str(sym_dict)),
            bar=int(data.get("bar", 1)),
            beat=float(data.get("beat", 1.0)),
            duration_bars=float(data.get("duration_bars", 1.0)),
            lyric=data.get("lyric", ""),
            line_number=int(data.get("line_number", 1)),
            column=int(data.get("column", 0)),
        )


@dataclass
class ChartLineInfo:
    """Mapeamento semântico e temporal de cada linha do documento da cifra."""
    line_number: int                    # Linha 1-indexada no texto original
    line_type: str                      # "METADATA", "SECTION", "CHORD", "LYRIC", "TABLATURE", "ANNOTATION", "EMPTY"
    text: str = ""
    section_id: str = ""
    section_name: str = ""
    start_bar: int = 1
    end_bar: int = 1
    chords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_number": self.line_number,
            "line_type": self.line_type,
            "text": self.text,
            "section_id": self.section_id,
            "section_name": self.section_name,
            "start_bar": self.start_bar,
            "end_bar": self.end_bar,
            "chords": list(self.chords),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChartLineInfo":
        return cls(
            line_number=int(data.get("line_number", 1)),
            line_type=str(data.get("line_type", "LYRIC")),
            text=str(data.get("text", "")),
            section_id=str(data.get("section_id", "")),
            section_name=str(data.get("section_name", "")),
            start_bar=int(data.get("start_bar", 1)),
            end_bar=int(data.get("end_bar", 1)),
            chords=list(data.get("chords", [])),
        )


@dataclass
class ChartSection:
    """Seção formal da cifra (ex: [Intro], [Verso 1], [Refrão])."""
    id: str
    name: str                           # Nome semântico original (ex: "Intro", "Verso 1", "Refrão")
    section_type: str = "UNKNOWN"       # "INTRO", "VERSE", "CHORUS", "BRIDGE", etc.
    label: str = "A"                    # Rótulo formal "A", "B", etc.
    chords: List[ChartChord] = field(default_factory=list)
    lyrics: List[LyricSegment] = field(default_factory=list)
    repeat_count: int = 1               # 1 por padrão, 2 para x2, etc.
    raw_lines: List[str] = field(default_factory=list)
    start_line: int = 1
    end_line: int = 1
    header_line: int = 1

    @property
    def chord_names(self) -> List[str]:
        """Lista dos símbolos originais de acordes contidos na seção."""
        return [c.symbol.original_symbol for c in self.chords]

    @property
    def expanded_chords(self) -> List[ChartChord]:
        """Retorna lista de acordes expandida de acordo com repetições (ex: x2)."""
        expanded = []
        for rep in range(self.repeat_count):
            for c in self.chords:
                expanded.append(c)
        return expanded

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "section_type": self.section_type,
            "label": self.label,
            "chords": [c.to_dict() for c in self.chords],
            "lyrics": [l.to_dict() for l in self.lyrics],
            "repeat_count": self.repeat_count,
            "raw_lines": list(self.raw_lines),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "header_line": self.header_line,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChartSection":
        chords_raw = data.get("chords", [])
        lyrics_raw = data.get("lyrics", [])
        return cls(
            id=data.get("id", "sec_01"),
            name=data.get("name", "Seção"),
            section_type=data.get("section_type", "UNKNOWN"),
            label=data.get("label", "A"),
            chords=[ChartChord.from_dict(c) for c in chords_raw],
            lyrics=[LyricSegment.from_dict(l) for l in lyrics_raw],
            repeat_count=int(data.get("repeat_count", 1)),
            raw_lines=list(data.get("raw_lines", [])),
            start_line=int(data.get("start_line", 1)),
            end_line=int(data.get("end_line", 1)),
            header_line=int(data.get("header_line", 1)),
        )


@dataclass
class ChordChart:
    """Estrutura completa da cifra musical de uma composição."""
    title: str = "Sem Título"
    artist: str = "Artista Desconhecido"
    key: str = "C Major"
    bpm: float = 120.0
    meter: str = "4/4"
    sections: List[ChartSection] = field(default_factory=list)
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    line_map: List[ChartLineInfo] = field(default_factory=list)
    # Modulações declaradas na cifra ("Tom: A" depois que a música começou).
    key_changes: List["KeyChangeEvent"] = field(default_factory=list)

    @property
    def key_timeline(self) -> "KeyTimeline":
        """Tom vigente em cada compasso escrito: tom inicial + modulações em ordem."""
        from app.music.key_timeline import KeyTimeline
        return KeyTimeline(self.key, list(self.key_changes))

    def key_at(self, bar: int, beat: float = 1.0) -> str:
        return self.key_timeline.key_at(bar, beat) if self.key_changes else self.key

    @property
    def capo_semitones(self) -> int:
        """Nº de semitons do capotraste (casa) a partir dos metadados (CifraClub).

        No CifraClub, os acordes impressos são os SHAPES a tocar com o capotraste; o som
        real ("Tom:") fica ``capo_semitones`` acima. Útil para comparar o áudio (que soa
        transposto) com os acordes escritos e localizar o músico na cifra.
        Ex.: "Capotraste na 2ª casa" -> 2 ; "Sem capotraste" -> 0.
        """
        raw = str(self.metadata.get("capo", "")).lower()
        if not raw or "sem" in raw or "nenhum" in raw:
            return 0
        m = re.search(r'(\d{1,2})', raw)
        return int(m.group(1)) if m else 0

    @property
    def all_chords(self) -> List[ChartChord]:
        """Sequência cronológica completa de todos os acordes de todas as seções (com repetições)."""
        seq = []
        for s in self.sections:
            seq.extend(s.expanded_chords)
        return seq

    @property
    def total_bars(self) -> int:
        """Contagem estimada de compassos totais da cifra."""
        chords = self.all_chords
        if not chords:
            return 0
        return max(c.bar for c in chords)

    def transpose(self, semitones: int, target_key: Optional[str] = None) -> "ChordChart":
        """Transpõe TODA a cifra por ``semitones`` (in place) e retorna self.

        Atualiza os acordes estruturados, os segmentos de letra, o mapa de linhas, o texto
        bruto exibido e a tonalidade declarada. A grafia (sustenido/bemol) segue o tom-alvo.
        """
        from app.music.transposition import (
            transpose_chord_symbol,
            transpose_text_block,
            key_uses_flats,
            transpose_note_name,
            parse_key_root_mode,
        )

        semitones %= 12
        if target_key is None:
            # Deriva o novo tom a partir do atual + semitons
            cur_root, cur_mode = parse_key_root_mode(self.key)
            new_root = transpose_note_name(cur_root, semitones, use_flats=False)
            target_key = f"{new_root} {'Minor' if cur_mode == 'minor' else 'Major'}"

        use_flats = key_uses_flats(target_key)

        if semitones != 0:
            for section in self.sections:
                for chord in section.chords:
                    chord.symbol = chord.symbol.transposed(semitones, use_flats, key_context=target_key)
                    if chord.lyric:
                        pass  # letra permanece
                for lyr in section.lyrics:
                    if lyr.chord_symbol:
                        lyr.chord_symbol = transpose_chord_symbol(lyr.chord_symbol, semitones, use_flats)
            for lm in self.line_map:
                if lm.chords:
                    lm.chords = [transpose_chord_symbol(c, semitones, use_flats) for c in lm.chords]
            if self.raw_text:
                self.raw_text = transpose_text_block(self.raw_text, semitones, use_flats)
            # As modulações andam junto: G → A → Bb transposto +2 vira A → B → C.
            from app.music.key_timeline import transpose_event
            self.key_changes = [transpose_event(e, semitones, use_flats) for e in self.key_changes]
            if self.key_changes:
                self.key_changes[0].from_key = target_key

        self.key = target_key
        self.metadata["transposed_semitones"] = int(self.metadata.get("transposed_semitones", 0) + semitones) % 12
        return self

    def transpose_to_key(self, target_key: str) -> "ChordChart":
        """Transpõe a cifra do tom atual para ``target_key`` (ex.: 'D Major', 'Bm')."""
        from app.music.transposition import semitones_between_keys
        semitones = semitones_between_keys(self.key, target_key)
        return self.transpose(semitones, target_key=target_key)

    # ------------------------------------------------------------------
    # Operações de bloco (copiar / reordenar / remover seções)
    # ------------------------------------------------------------------
    def _reindex_bars(self) -> None:
        """Recompacta a numeração de compassos na ordem atual das seções."""
        bar = 1
        for sec in self.sections:
            if not sec.chords:
                continue
            first = min(c.bar for c in sec.chords)
            offset = bar - first
            for c in sec.chords:
                c.bar += offset
            bar = max(c.bar for c in sec.chords) + 1

    def duplicate_section(self, index: int, at: Optional[int] = None) -> "ChartSection":
        """Copia a seção em ``index`` (bloco inteiro) e a insere em ``at`` (ou logo após)."""
        import copy
        if not (0 <= index < len(self.sections)):
            raise IndexError("Índice de seção inválido")
        clone = copy.deepcopy(self.sections[index])
        used_ids = {section.id for section in self.sections}
        base_id = f"{clone.id}_copy"
        clone.id = base_id
        suffix = 2
        while clone.id in used_ids:
            clone.id = f"{base_id}_{suffix}"
            suffix += 1
        dest = index + 1 if at is None else max(0, min(len(self.sections), at))
        self.sections.insert(dest, clone)
        self._reindex_bars()
        return clone

    def move_section(self, from_index: int, to_index: int) -> None:
        """Reordena um bloco: move a seção de ``from_index`` para ``to_index``."""
        if not (0 <= from_index < len(self.sections)):
            raise IndexError("Índice de origem inválido")
        to_index = max(0, min(len(self.sections) - 1, to_index))
        sec = self.sections.pop(from_index)
        self.sections.insert(to_index, sec)
        self._reindex_bars()

    def remove_section(self, index: int) -> None:
        """Remove o bloco na posição ``index``."""
        if not (0 <= index < len(self.sections)):
            raise IndexError("Índice de seção inválido")
        self.sections.pop(index)
        self._reindex_bars()

    def section_color(self, index: int) -> str:
        """Cor sugerida (hex) para o bloco na posição ``index``, por tipo de seção."""
        from app.music.sectionizer import section_color
        if not (0 <= index < len(self.sections)):
            return section_color("UNKNOWN")
        return section_color(self.sections[index].section_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "artist": self.artist,
            "key": self.key,
            "bpm": self.bpm,
            "meter": self.meter,
            "sections": [s.to_dict() for s in self.sections],
            "raw_text": self.raw_text,
            "metadata": dict(self.metadata),
            "line_map": [lm.to_dict() for lm in self.line_map],
            "key_changes": [event.to_dict() for event in self.key_changes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChordChart":
        from app.music.key_timeline import KeyChangeEvent
        sec_raw = data.get("sections", [])
        line_map_raw = data.get("line_map", [])
        return cls(
            key_changes=[KeyChangeEvent.from_dict(e) for e in data.get("key_changes", [])],
            title=data.get("title", "Sem Título"),
            artist=data.get("artist", "Artista Desconhecido"),
            key=data.get("key", "C Major"),
            bpm=float(data.get("bpm", 120.0)),
            meter=data.get("meter", "4/4"),
            sections=[ChartSection.from_dict(s) for s in sec_raw],
            raw_text=data.get("raw_text", ""),
            metadata=dict(data.get("metadata", {})),
            line_map=[ChartLineInfo.from_dict(lm) for lm in line_map_raw],
        )



# =====================================================================
# Parser e Normalizador de Acordes Individuais
# =====================================================================

def parse_chord(symbol: str, key_context: str = "C Major") -> ChordSymbol:
    """Analisa e decompõe um símbolo de acorde complexo preservando a originalidade.
    
    Suporta:
    - Tríades e Tétrades básicas: C, Cm, C7, Cmaj7, C7M, CΔ7
    - Extensões e tensões: C9, Cmaj9, Cm9, C11, C13, Cadd9, Csus2, Csus4
    - Alterações: C7(9), C7(b9), C7(#9), F#m7(b5), C7(b13)
    - Diminutos e Aumentados: Cdim, C°, Caug, C+
    - Inversões e Slash chords: C/E, C/G, Dm/F, G/B
    """
    raw = symbol.strip()
    if not raw or raw == "--":
        return ChordSymbol(original_symbol="--", root="C", quality="major")

    # 1. Separar baixo invertido (slash chord)
    parts = raw.split('/')
    chord_part = parts[0].strip()
    bass_part = parts[1].strip() if len(parts) > 1 else ""

    # 2. Extrair tônica (root)
    if len(chord_part) >= 2 and chord_part[1] in ('#', 'b'):
        root_raw = chord_part[:2]
        rest = chord_part[2:]
    else:
        root_raw = chord_part[:1]
        rest = chord_part[1:]

    # Normalizar enarmonia da tônica
    root = ENHARMONIC_MAP.get(root_raw, root_raw)
    if root not in PITCH_CLASSES:
        root = "C"

    # Baixo da inversão
    bass_note = ENHARMONIC_MAP.get(bass_part, bass_part) if bass_part else root

    # 3. Extrair alterações entre parênteses: ex: C7(b9) -> rest="7", alts=["b9"]
    alterations = []
    paren_match = re.search(r'\((.*?)\)', rest)
    if paren_match:
        alts_str = paren_match.group(1)
        alterations = [a.strip() for a in alts_str.split(',') if a.strip()]
        rest = rest[:paren_match.start()] + rest[paren_match.end():]

    # 4. Classificar qualidade e extensões
    rest_clean = rest.strip()
    quality = "major"
    extension = ""

    # Padrões específicos de extensões complexas
    if "m7(b5)" in raw or "ø" in raw:
        quality = "dim"
        extension = "m7(b5)"
        alterations.append("b5")
    elif "dim7" in rest_clean or "°7" in rest_clean or "º7" in rest_clean:
        quality = "dim"
        extension = "dim7"
    elif "dim" in rest_clean or "°" in rest_clean or "º" in rest_clean:
        quality = "dim"
        extension = "dim"
    elif "aug" in rest_clean or "+" in rest_clean:
        quality = "aug"
        extension = "aug"
    elif "sus4" in rest_clean:
        quality = "sus4"
        extension = "sus4"
    elif "sus2" in rest_clean:
        quality = "sus2"
        extension = "sus2"
    # Notação CifraClub: 5- = quinta diminuta, m5- = acorde diminuto
    elif "m5-" in rest_clean or "5-" in rest_clean:
        quality = "dim"
        extension = "dim"
    # Notação CifraClub: acorde "C4" = Csus4, "C2" = Csus2 (4/5/2 isolados)
    elif rest_clean == "4":
        quality = "sus4"
        extension = "sus4"
    elif rest_clean == "2":
        quality = "sus2"
        extension = "sus2"
    elif any(tag in rest_clean for tag in ("maj9", "7M(9)", "Δ9", "M9")):
        quality = "major"
        extension = "maj9"
    elif any(tag in rest_clean for tag in ("maj7", "7M", "Δ7", "Δ", "M7")):
        quality = "major"
        extension = "maj7"
    elif any(tag in rest_clean for tag in ("add9", "add2")):
        quality = "major"
        extension = "add9"
    elif "m9" in rest_clean or "min9" in rest_clean:
        quality = "minor"
        extension = "m9"
    elif "m7" in rest_clean or "min7" in rest_clean:
        quality = "minor"
        extension = "m7"
    elif "m11" in rest_clean:
        quality = "minor"
        extension = "m11"
    elif "13" in rest_clean:
        quality = "major"
        extension = "13"
    elif "11" in rest_clean:
        quality = "major"
        extension = "11"
    elif "9" in rest_clean:
        quality = "major"
        extension = "9"
    elif "7" in rest_clean:
        quality = "major"
        extension = "7"
    elif rest_clean.startswith("m") and not rest_clean.startswith("maj"):
        quality = "minor"
        extension = rest_clean[1:]
    else:
        quality = "major"
        extension = rest_clean

    # 5. Determinar Inversão teórica
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

    # 6. Graus Harmônicos e Numerais Romanos
    harmonic_degree, roman_numeral = calculate_harmonic_degree(
        root=root,
        quality=quality,
        extension=extension,
        key_context=key_context
    )

    return ChordSymbol(
        original_symbol=raw,
        root=root,
        quality=quality,
        extension=extension,
        alterations=alterations,
        bass_note=bass_note,
        inversion=inversion,
        harmonic_degree=harmonic_degree,
        roman_numeral=roman_numeral,
        confidence=1.0
    )


def calculate_harmonic_degree(
    root: str,
    quality: str,
    extension: str = "",
    key_context: str = "C Major"
) -> Tuple[str, str]:
    """Calcula o grau harmônico relativo à tonalidade de referência.
    
    Ex: Cmaj7 em C Major -> ("I", "Imaj7")
        Am7 em C Major -> ("vi", "vi7")
        Dm7 em C Major -> ("ii", "ii7")
        G7 em C Major -> ("V", "V7")
    """
    key_parts = key_context.strip().split()
    key_root = ENHARMONIC_MAP.get(key_parts[0], "C") if key_parts else "C"
    key_mode = key_parts[1].lower() if len(key_parts) > 1 else "major"
    is_major = "maj" in key_mode

    if key_root not in PITCH_CLASSES or root not in PITCH_CLASSES:
        return "I", "I"

    root_idx = PITCH_CLASSES.index(root)
    key_idx = PITCH_CLASSES.index(key_root)
    relative_degree = (root_idx - key_idx) % 12

    roman_map = ROMAN_MAJOR if is_major else ROMAN_MINOR
    maj_sym, min_sym = roman_map.get(relative_degree, ("I", "i"))

    is_minor = quality in ("minor", "m", "dim", "m7", "m9")
    harmonic_degree = min_sym if is_minor else maj_sym

    # Adicionar sufixo da tétrade/extensão
    ext_suffix = ""
    if extension in ("maj7", "7M", "Δ7", "M7"):
        ext_suffix = "maj7"
    elif extension in ("7", "m7"):
        ext_suffix = "7"
    elif extension in ("9", "m9", "maj9"):
        ext_suffix = "9"
    elif extension in ("dim", "°"):
        ext_suffix = "°"
    elif extension in ("aug", "+"):
        ext_suffix = "+"
    elif extension:
        ext_suffix = extension

    roman_numeral = f"{harmonic_degree}{ext_suffix}"
    return harmonic_degree, roman_numeral
