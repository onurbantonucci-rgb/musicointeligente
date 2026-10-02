"""Auto-detecção de Seções Musicais em Cifras (Sectionizer v0.6).

Quando uma cifra NÃO traz as partes marcadas entre colchetes (ex.: [Intro], [Refrão]),
este módulo segmenta o texto em blocos (separados por linhas em branco) e rotula cada
bloco em Intro / Verso / Pré-refrão / Refrão / Ponte / Solo / Instrumental / Outro,
usando heurísticas de repetição harmônica e presença de letra.

Princípio central: o bloco de acordes que se REPETE ao longo da música é, quase sempre,
o REFRÃO (o gancho). Blocos únicos com letra são VERSOS; um bloco único e contrastante
que aparece depois de já existirem verso e refrão é tratado como PONTE.

Também expõe um mapa de cores por tipo de seção, para a interface diferenciar visualmente.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from app.music.chord_chart import parse_chord


# Cores (hex) por tipo de seção — usadas pela interface para colorir os blocos
SECTION_COLORS: Dict[str, str] = {
    "INTRO": "#6c757d",         # cinza
    "VERSE": "#2d6cdf",         # azul
    "PRE_CHORUS": "#17a2b8",    # ciano
    "CHORUS": "#e63946",        # vermelho (destaque — o gancho)
    "BRIDGE": "#8e44ad",        # roxo
    "SOLO": "#f39c12",          # laranja
    "INSTRUMENTAL": "#16a085",  # verde-azulado
    "BREAK": "#7f8c8d",         # cinza-azulado
    "OUTRO": "#495057",         # cinza-escuro
    "UNKNOWN": "#5a6677",       # neutro
}

# Nomes amigáveis (PT-BR) por tipo de seção
SECTION_LABELS_PT: Dict[str, str] = {
    "INTRO": "Intro",
    "VERSE": "Verso",
    "PRE_CHORUS": "Pré-refrão",
    "CHORUS": "Refrão",
    "BRIDGE": "Ponte",
    "SOLO": "Solo",
    "INSTRUMENTAL": "Instrumental",
    "BREAK": "Break",
    "OUTRO": "Outro",
    "UNKNOWN": "Parte",
}


def section_color(section_type: str) -> str:
    """Retorna a cor associada a um tipo de seção (fallback: cor neutra)."""
    return SECTION_COLORS.get((section_type or "UNKNOWN").upper(), SECTION_COLORS["UNKNOWN"])


@dataclass
class _Block:
    lines: List[str]
    chord_sig: Tuple[str, ...] = field(default_factory=tuple)
    lyric_sig: Tuple[str, ...] = field(default_factory=tuple)
    has_lyrics: bool = False
    has_chords: bool = False


def _classifier():
    from app.input.chart_semantic_classifier import ChartSemanticClassifier, SemanticLineType
    return ChartSemanticClassifier, SemanticLineType


def text_has_section_headers(text: str) -> bool:
    """Indica se a cifra já traz cabeçalhos de seção (ex.: [Refrão]) — então não mexemos."""
    Classifier, SemLineType = _classifier()
    from app.input.chart_parser import ChartParser
    for line in text.split('\n'):
        if (Classifier.classify_line(line).line_type == SemLineType.SECTION or
                ChartParser.parse_section_header(line)):
            return True
    return False


def _chord_signature(chord_lines: List[str]) -> Tuple[str, ...]:
    """Assinatura harmônica de um bloco: sequência canônica de acordes (grafia-invariante)."""
    Classifier, _ = _classifier()
    sig: List[str] = []
    for line in chord_lines:
        for tok, _pos in Classifier.classify_line(line).chord_tokens:
            sig.append(parse_chord(tok).canonical_name)
    return tuple(sig)


def _segment_blocks(text: str) -> List[_Block]:
    """Divide o texto em blocos separados por linhas em branco, ignorando metadados."""
    Classifier, SemLineType = _classifier()
    blocks: List[_Block] = []
    current: List[str] = []

    def flush():
        if not current:
            return
        chord_lines = []
        lyric_lines = []
        for ln in current:
            ct = Classifier.classify_line(ln)
            if ct.line_type == SemLineType.CHORD:
                chord_lines.append(ln)
            elif ct.line_type == SemLineType.LYRIC:
                lyric_lines.append(" ".join(ln.lower().split()))
        blocks.append(_Block(
            lines=list(current),
            chord_sig=_chord_signature(chord_lines),
            lyric_sig=tuple(lyric_lines),
            has_lyrics=bool(lyric_lines),
            has_chords=bool(chord_lines),
        ))
        current.clear()

    for line in text.split('\n'):
        ct = Classifier.classify_line(line)
        if ct.line_type == SemLineType.EMPTY:
            flush()
            continue
        if ct.line_type == SemLineType.METADATA:
            # Metadados ficam fora dos blocos (preservados à parte)
            continue
        current.append(line)
    flush()
    return blocks


def _label_blocks(blocks: List[_Block]) -> List[str]:
    """Atribui um tipo de seção a cada bloco pela heurística de repetição/letra.

    O REFRÃO é o bloco cujo conjunto se repete: preferimos a repetição da LETRA (o
    refrão repete letra + acordes; versos repetem acordes, mas têm letra própria). Em
    cifras sem letra, caímos para a repetição da progressão de acordes.
    """
    song_has_lyrics = any(b.has_lyrics for b in blocks)

    def rep_sig(b: _Block) -> Tuple[str, ...]:
        return b.lyric_sig if song_has_lyrics else b.chord_sig

    sig_count: Dict[Tuple[str, ...], int] = {}
    for b in blocks:
        s = rep_sig(b)
        if s:
            sig_count[s] = sig_count.get(s, 0) + 1
    repeated = {s for s, n in sig_count.items() if n >= 2 and s}

    def is_chorus(b: _Block) -> bool:
        return rep_sig(b) in repeated and (b.has_lyrics or not song_has_lyrics)

    # Progressão típica dos VERSOS = o chord_sig mais comum entre blocos com letra que NÃO
    # são refrão (versos costumam reusar a mesma progressão com letras diferentes).
    verse_chord_counts: Dict[Tuple[str, ...], int] = {}
    for b in blocks:
        if b.has_lyrics and not is_chorus(b) and b.chord_sig:
            verse_chord_counts[b.chord_sig] = verse_chord_counts.get(b.chord_sig, 0) + 1
    verse_sig = max(verse_chord_counts, key=verse_chord_counts.get) if verse_chord_counts else None

    types: List[str] = []
    seen_verse = False
    seen_chorus = False
    for idx, b in enumerate(blocks):
        if not b.has_chords and not b.has_lyrics:
            types.append("INSTRUMENTAL")
            continue

        if is_chorus(b):
            t = "CHORUS"
            seen_chorus = True
        elif not b.has_lyrics and b.has_chords:
            # Bloco só instrumental: Intro (primeiro), Outro (último) ou Instrumental
            if idx == 0 and not seen_verse and not seen_chorus:
                t = "INTRO"
            elif idx == len(blocks) - 1:
                t = "OUTRO"
            else:
                t = "INSTRUMENTAL"
        else:
            # Bloco com letra e assinatura única.
            # Ponte = progressão DIFERENTE da dos versos, aparecendo após já haver refrão.
            if verse_sig is not None and b.chord_sig != verse_sig and seen_chorus and seen_verse:
                t = "BRIDGE"
            else:
                t = "VERSE"
                seen_verse = True
        types.append(t)

    return types


def _minimal_period(seq: Tuple[str, ...]) -> Tuple[str, ...]:
    """Menor unidade que, repetida, forma a sequência (D C D C -> D C)."""
    for size in range(1, len(seq) + 1):
        if len(seq) % size == 0 and seq == seq[:size] * (len(seq) // size):
            return seq[:size]
    return seq


def _line_chords(line: str) -> Tuple[str, ...]:
    """Acordes canônicos de uma linha de acordes (ou do resto de um cabeçalho inline)."""
    Classifier, SemLineType = _classifier()
    ct = Classifier.classify_line(line)
    if ct.line_type != SemLineType.CHORD:
        return ()
    return tuple(parse_chord(tok).canonical_name for tok, _pos in ct.chord_tokens)


def split_chord_only_passages(text: str) -> str:
    """Separa em bloco próprio as passagens SÓ DE ACORDES que fecham uma seção com letra.

    No Cifra Club a volta da introdução (ou um trecho instrumental) costuma vir no fim
    do bloco cantado, sem cabeçalho próprio — ex.: "( D#m  C#4  B9  F# )" logo depois da
    última linha do Refrão. Sem um cabeçalho, o trecho fica grudado no Refrão e não dá
    para editá-lo/movê-lo como um bloco. Aqui ele ganha um cabeçalho: [Reintro] quando
    repete a progressão da Intro, senão [Instrumental].

    Só separa o que vem DEPOIS da última linha de letra da seção e é só de acordes,
    e somente quando o trecho está destacado da letra (linha em branco antes ou acordes
    entre parênteses) — acordes logo abaixo da letra continuam sendo o fim da frase.
    """
    if not text or not text.strip():
        return text
    Classifier, SemLineType = _classifier()
    from app.input.chart_parser import ChartParser

    lines = text.split("\n")
    headers = [i for i, ln in enumerate(lines) if ChartParser.parse_section_header(ln.strip())]
    if not headers:
        return text

    intro_unit: Tuple[str, ...] = ()
    for pos, h in enumerate(headers):
        _name, sec_type = ChartParser.parse_section_header(lines[h].strip())
        if sec_type != "INTRO":
            continue
        end = headers[pos + 1] if pos + 1 < len(headers) else len(lines)
        inline = lines[h].strip()
        chords = _line_chords(inline[inline.index("]") + 1:]) if inline.startswith("[") else ()
        for ln in lines[h + 1:end]:
            chords += _line_chords(ln)
        if chords:                      # [Tab - Intro] só com tablatura não define a progressão
            intro_unit = _minimal_period(chords)
            break

    inserts: Dict[int, str] = {}
    for pos, h in enumerate(headers):
        end = headers[pos + 1] if pos + 1 < len(headers) else len(lines)
        kinds = [Classifier.classify_line(lines[i]).line_type for i in range(h + 1, end)]
        lyric_rows = [h + 1 + k for k, kind in enumerate(kinds) if kind == SemLineType.LYRIC]
        if not lyric_rows:
            continue
        tail = [i for i in range(lyric_rows[-1] + 1, end) if lines[i].strip()]
        if not tail or any(Classifier.classify_line(lines[i]).line_type != SemLineType.CHORD
                           for i in tail):
            continue
        chords: Tuple[str, ...] = ()
        for i in tail:
            chords += _line_chords(lines[i])
        detached = tail[0] > lyric_rows[-1] + 1
        bracketed = all(lines[i].strip().startswith("(") for i in tail)
        if len(chords) < 2 or not (detached or bracketed):
            continue
        is_reintro = bool(intro_unit) and _minimal_period(chords) == intro_unit
        inserts[tail[0]] = "[Reintro]" if is_reintro else "[Instrumental]"

    if not inserts:
        return text
    out: List[str] = []
    for i, ln in enumerate(lines):
        if i in inserts:
            if out and out[-1].strip():
                out.append("")
            out.append(inserts[i])
        out.append(ln)
    return "\n".join(out)


def auto_sectionize_text(text: str) -> str:
    """Insere cabeçalhos de seção em uma cifra sem marcação e retorna o novo texto.

    Se a cifra já tiver seções marcadas, retorna o texto inalterado.
    """
    if not text or not text.strip():
        return text
    if text_has_section_headers(text):
        return text

    blocks = _segment_blocks(text)
    if not blocks:
        return text
    types = _label_blocks(blocks)

    # Preserva as linhas de metadados no topo (Tom/BPM/etc.)
    Classifier, SemLineType = _classifier()
    meta_lines = [ln for ln in text.split('\n')
                  if Classifier.classify_line(ln).line_type == SemLineType.METADATA]

    # Numera seções repetidas do mesmo tipo (Verso 1, Verso 2, Refrão...)
    type_counts: Dict[str, int] = {}
    out: List[str] = list(meta_lines)
    if meta_lines:
        out.append("")

    # Só numeramos tipos que costumam ter múltiplas variações (Verso). Refrão e demais
    # repetem o MESMO rótulo (o mesmo refrão aparece várias vezes).
    numbered_types = {"VERSE"}
    total_of_type: Dict[str, int] = {}
    for t in types:
        total_of_type[t] = total_of_type.get(t, 0) + 1

    for b, t in zip(blocks, types):
        base = SECTION_LABELS_PT.get(t, "Parte")
        if t in numbered_types and total_of_type.get(t, 0) > 1:
            type_counts[t] = type_counts.get(t, 0) + 1
            header = f"[{base} {type_counts[t]}]"
        else:
            header = f"[{base}]"
        out.append(header)
        out.extend(b.lines)
        out.append("")

    return "\n".join(out).rstrip() + "\n"
