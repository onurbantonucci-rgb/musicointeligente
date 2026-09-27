"""Analisador Sintático e Léxico de Cifras e Letras (ChartParser v0.4).

Converte texto de cifra (padrão Cifra Club / cancioneiro popular) em ChordChart
estruturado com mapeamento temporal por linha, separando estritamente metadados,
seções, tablaturas, acordes e letras através do ChartSemanticClassifier.
"""

import re
from typing import List, Tuple, Optional, Dict, Any

from app.music.chord_chart import (
    ChordChart,
    ChartSection,
    ChartChord,
    LyricSegment,
    ChartLineInfo,
    parse_chord,
)
from app.input.chart_semantic_classifier import (
    ChartSemanticClassifier,
    SemanticLineType,
)


class ChartParser:
    """Parser robusto para conversão de texto de cifra e letra em ChordChart estruturado."""

    @staticmethod
    def is_chord_token(token: str) -> bool:
        """Determina se uma palavra individual é genuinamente um símbolo de acorde."""
        return ChartSemanticClassifier.is_chord_token(token)

    @classmethod
    def is_chord_line(cls, line: str) -> bool:
        """Verifica se a linha é predominantemente composta por acordes e separadores."""
        classified = ChartSemanticClassifier.classify_line(line)
        return classified.line_type == SemanticLineType.CHORD

    @classmethod
    def extract_repeats(cls, line: str) -> Tuple[str, int]:
        """Detecta anotações de repetição na linha (ex: 'C G Am F x2' -> repeat=2)."""
        repeat = 1
        clean_line = line

        # Padrão x2, 2x, (2x), (x2)
        match = re.search(r'\b(?:x([2-9])|([2-9])x)\b', line, re.IGNORECASE)
        if match:
            cnt = match.group(1) or match.group(2)
            repeat = int(cnt)
            clean_line = line[:match.start()] + line[match.end():]

        # Padrão repete 2 vezes / repeat 2x
        match_rep = re.search(r'(?:repete|repeat)\s*([2-9])', line, re.IGNORECASE)
        if match_rep:
            repeat = int(match_rep.group(1))
            clean_line = line[:match_rep.start()] + line[match_rep.end():]

        return clean_line.strip(), repeat

    @classmethod
    def parse_section_header(cls, line: str) -> Optional[Tuple[str, str]]:
        """Identifica seções isoladas ou com acordes na mesma linha."""
        section = ChartSemanticClassifier.parse_section_line(line)
        if section:
            return section
        match = re.match(r'^\s*(\[[^\]]+\])\s+(.+)$', line)
        if not match:
            return None
        section = ChartSemanticClassifier.parse_section_line(match.group(1))
        if (section and not ChartSemanticClassifier.parse_annotation_line(match.group(1)) and
                cls.is_chord_line(match.group(2))):
            return section
        return None

    @staticmethod
    def _split_inline_headers(lines: List[str], with_line_numbers: bool = False):
        """Separa cabeçalhos de seção que trazem acordes na MESMA linha (padrão CifraClub).

        Ex.: "[Intro]  G#  C  Fm  C#9"  ->  "[Intro]"  +  "G#  C  Fm  C#9".
        Só divide quando o rótulo é uma seção real (não uma anotação) e há conteúdo após ']'.
        """
        out: List[str] = []
        source_lines: List[int] = []
        pat = re.compile(r'^(\s*\[[^\]]+\])\s+(\S.*)$')
        for physical_line, line in enumerate(lines, 1):
            m = pat.match(line)
            if m and ChartParser.parse_section_header(line):
                header, rest = m.group(1), m.group(2)
                out.extend((header, rest))
                source_lines.extend((physical_line, physical_line))
                continue
            out.append(line)
            source_lines.append(physical_line)
        return (out, source_lines) if with_line_numbers else out

    @classmethod
    def parse(
        cls,
        text: str,
        default_key: str = "C Major",
        default_bpm: float = 120.0,
        default_meter: str = "4/4",
        title: str = "Sem Título",
        artist: str = "Artista Desconhecido"
    ) -> ChordChart:
        """Processa o texto completo da cifra e constrói o objeto ChordChart com resolução por linha."""
        lines, source_lines = cls._split_inline_headers(
            text.splitlines(), with_line_numbers=True)
        sections: List[ChartSection] = []
        current_section: Optional[ChartSection] = None
        current_bar = 1
        section_idx = 1
        section_label_idx = 0
        line_map: List[ChartLineInfo] = []

        # Metadados detectados
        detected_key = default_key
        detected_bpm = default_bpm
        detected_meter = default_meter
        song_metadata: Dict[str, Any] = {}

        labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

        current_has_header = False

        def ensure_section(name: str = "Intro", sec_type: str = "INTRO", line_num: int = 1,
                           from_header: bool = False) -> ChartSection:
            nonlocal current_section, section_idx, section_label_idx, current_has_header
            # Um cabeçalho explícito SEMPRE inicia um novo bloco. Uma seção com apenas
            # tablatura (raw_lines) NÃO é "vazia": sem isto, um [Solo] só de tab faria o
            # [Verso] seguinte ser engolido no mesmo bloco.
            has_content = current_section is not None and (
                current_section.chords or current_section.lyrics or current_section.raw_lines
            )
            create_new = (
                current_section is None
                or has_content
                or (from_header and current_has_header)
            )
            if create_new:
                lbl = labels[section_label_idx % len(labels)]
                sec_id = f"sec_{section_idx:02d}"
                current_section = ChartSection(
                    id=sec_id,
                    name=name,
                    section_type=sec_type,
                    label=lbl,
                    header_line=line_num,
                    start_line=line_num,
                    end_line=line_num
                )
                sections.append(current_section)
                section_idx += 1
                section_label_idx += 1
            elif from_header:
                # Reaproveita uma seção implícita ainda vazia, adotando o rótulo do cabeçalho
                current_section.name = name
                current_section.section_type = sec_type
            current_has_header = from_header
            return current_section

        i = 0
        while i < len(lines):
            line_str = lines[i]
            line_num = source_lines[i]  # linha física no texto exibido
            classified = ChartSemanticClassifier.classify_line(line_str, line_number=line_num)

            # 1. Linha Vazia
            if classified.line_type == SemanticLineType.EMPTY:
                line_map.append(ChartLineInfo(
                    line_number=line_num,
                    line_type="EMPTY",
                    text=line_str,
                    start_bar=0,
                    end_bar=0
                ))
                i += 1
                continue

            # 2. Metadados (Tom, BPM, Compasso, etc.) — NÃO polui a harmonia!
            if classified.line_type == SemanticLineType.METADATA:
                meta = classified.extracted_metadata
                if "key" in meta:
                    detected_key = meta["key"]
                if "bpm" in meta:
                    detected_bpm = meta["bpm"]
                if "meter" in meta:
                    detected_meter = meta["meter"]
                if "title" in meta:
                    title = meta["title"]
                if "artist" in meta:
                    artist = meta["artist"]
                song_metadata.update(meta)

                line_map.append(ChartLineInfo(
                    line_number=line_num,
                    line_type="METADATA",
                    text=line_str,
                    start_bar=0,
                    end_bar=0
                ))
                i += 1
                continue

            # 3. Tablatura — Preservada sem enviar notas como acordes
            if classified.line_type == SemanticLineType.TABLATURE:
                sec = current_section
                if sec is not None:
                    sec.raw_lines.append(line_str)
                    sec.end_line = max(sec.end_line, line_num)
                line_map.append(ChartLineInfo(
                    line_number=line_num,
                    line_type="TABLATURE",
                    text=line_str,
                    section_id=sec.id if sec else "",
                    section_name=sec.name if sec else "",
                    start_bar=0,
                    end_bar=0
                ))
                i += 1
                continue

            # 4. Cabeçalho de Seção (ex: [Refrão], [Verso 1])
            if classified.line_type == SemanticLineType.SECTION:
                sec_name = classified.section_name or "Seção"
                sec_type = classified.section_type or "UNKNOWN"
                sec = ensure_section(sec_name, sec_type, line_num, from_header=True)
                sec.header_line = line_num
                sec.start_line = line_num
                sec.end_line = line_num

                line_map.append(ChartLineInfo(
                    line_number=line_num,
                    line_type="SECTION",
                    text=line_str,
                    section_id=sec.id,
                    section_name=sec.name,
                    start_bar=0,
                    end_bar=0
                ))
                i += 1
                continue

            # 5. Anotações de Performance (ex: [Repete 2x], [Entra bateria])
            if classified.line_type == SemanticLineType.ANNOTATION:
                sec = current_section
                if sec is not None:
                    sec.raw_lines.append(line_str)
                    sec.end_line = max(sec.end_line, line_num)
                line_map.append(ChartLineInfo(
                    line_number=line_num,
                    line_type="ANNOTATION",
                    text=line_str,
                    section_id=sec.id if sec else "",
                    section_name=sec.name if sec else "",
                    start_bar=0,
                    end_bar=0
                ))
                i += 1
                continue

            # 5.5. Linhas numéricas são conteúdo visual/contagem, nunca eventos.
            if classified.line_type == SemanticLineType.NUMBER:
                sec = current_section
                if sec is not None:
                    sec.raw_lines.append(line_str)
                    sec.end_line = max(sec.end_line, line_num)
                line_map.append(ChartLineInfo(
                    line_number=line_num,
                    line_type="NUMBER",
                    text=line_str,
                    section_id=sec.id if sec else "",
                    section_name=sec.name if sec else "",
                    start_bar=0,
                    end_bar=0
                ))
                i += 1
                continue

            # 6. Linha de Acordes (com possível pareamento de linha de letra subsequente)
            if classified.line_type == SemanticLineType.CHORD:
                sec = current_section if current_section is not None else ensure_section("Intro", "INTRO", line_num)
                if classified.repeat_count > 1:
                    sec.repeat_count = max(sec.repeat_count, classified.repeat_count)

                # Defesa adicional: a classificação CHORD só produz eventos
                # que ainda passam pela allowlist harmônica exata.
                chord_tokens = [item for item in classified.chord_tokens
                                if ChartSemanticClassifier.is_chord_shaped(item[0])]
                if not chord_tokens:
                    i += 1
                    continue
                chord_start_bar = current_bar

                # Verifica se a próxima linha é uma linha de letra casada
                has_next_lyric = False
                lyric_line_str = ""
                next_line_num = source_lines[i + 1] if i + 1 < len(lines) else line_num + 1

                if (i + 1) < len(lines):
                    next_classified = ChartSemanticClassifier.classify_line(lines[i + 1], line_number=next_line_num)
                    if next_classified.line_type == SemanticLineType.LYRIC:
                        has_next_lyric = True
                        lyric_line_str = lines[i + 1]

                chord_symbols_list: List[str] = []
                for idx, (sym, col_pos) in enumerate(chord_tokens):
                    chord_obj = parse_chord(sym, key_context=detected_key)
                    chord_symbols_list.append(sym)

                    # Trecho de letra alinhado na coluna do acorde
                    lyric_snippet = ""
                    if has_next_lyric:
                        start_col = col_pos
                        end_col = chord_tokens[idx + 1][1] if (idx + 1) < len(chord_tokens) else len(lyric_line_str)
                        if start_col < len(lyric_line_str):
                            lyric_snippet = lyric_line_str[start_col:end_col].strip()

                    chart_chord = ChartChord(
                        symbol=chord_obj,
                        bar=current_bar,
                        beat=1.0,
                        duration_bars=1.0,
                        lyric=lyric_snippet,
                        line_number=line_num,
                        column=col_pos
                    )
                    sec.chords.append(chart_chord)

                    if lyric_snippet:
                        sec.lyrics.append(LyricSegment(
                            text=lyric_snippet,
                            chord_symbol=sym,
                            bar_offset=current_bar,
                            section_id=sec.id,
                            line_number=next_line_num if has_next_lyric else line_num
                        ))

                    current_bar += 1

                chord_end_bar = max(chord_start_bar, current_bar - 1)
                sec.raw_lines.append(line_str)
                sec.end_line = max(sec.end_line, line_num)

                line_map.append(ChartLineInfo(
                    line_number=line_num,
                    line_type="CHORD",
                    text=line_str,
                    section_id=sec.id,
                    section_name=sec.name,
                    start_bar=chord_start_bar,
                    end_bar=chord_end_bar,
                    chords=chord_symbols_list
                ))

                if has_next_lyric:
                    sec.raw_lines.append(lyric_line_str)
                    sec.end_line = max(sec.end_line, next_line_num)
                    line_map.append(ChartLineInfo(
                        line_number=next_line_num,
                        line_type="LYRIC",
                        text=lyric_line_str,
                        section_id=sec.id,
                        section_name=sec.name,
                        start_bar=chord_start_bar,
                        end_bar=chord_end_bar,
                        chords=chord_symbols_list
                    ))
                    i += 1  # Consome a linha de letra pareada

                i += 1
                continue

            # 7. Linha de Letra Avulsa (sem acordes na linha superior)
            if classified.line_type == SemanticLineType.LYRIC:
                # Texto livre antes da primeira estrutura musical é cabeçalho
                # visual (título/artista), não letra com duração na timeline.
                if current_section is None:
                    line_map.append(ChartLineInfo(
                        line_number=line_num,
                        line_type="HEADER",
                        text=line_str,
                        start_bar=0,
                        end_bar=0
                    ))
                    i += 1
                    continue
                sec = current_section if current_section is not None else ensure_section("Verso", "VERSE", line_num)
                sec.lyrics.append(LyricSegment(
                    text=classified.cleaned_text,
                    bar_offset=current_bar,
                    section_id=sec.id,
                    line_number=line_num
                ))
                sec.raw_lines.append(line_str)
                sec.end_line = max(sec.end_line, line_num)

                # Linha de letra avulsa ocupa 1 compasso de tempo musical
                lyric_start_bar = current_bar
                current_bar += 1
                lyric_end_bar = current_bar - 1

                line_map.append(ChartLineInfo(
                    line_number=line_num,
                    line_type="LYRIC",
                    text=line_str,
                    section_id=sec.id,
                    section_name=sec.name,
                    start_bar=lyric_start_bar,
                    end_bar=lyric_end_bar
                ))
                i += 1
                continue

            i += 1

        return ChordChart(
            title=title,
            artist=artist,
            key=detected_key,
            bpm=detected_bpm,
            meter=detected_meter,
            sections=sections,
            raw_text=text,
            metadata=song_metadata,
            line_map=line_map
        )
