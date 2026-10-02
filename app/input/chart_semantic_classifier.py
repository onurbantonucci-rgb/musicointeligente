"""Classificador Semântico de Linhas de Cifras e Letras (ChartSemanticClassifier v0.4).

Analisa e categoriza cada linha de um documento de cifra bruto em categorias semânticas:
- METADATA (Tom, BPM, Compasso, Afinação, Capotraste, Artista, Título)
- SECTION (Cabeçalhos de estrutura musical: Intro, Verso, Refrão, etc.)
- CHORD (Linha contendo predominantemente acordes e símbolos harmônicos)
- LYRIC (Linha contendo letra cantada)
- TABLATURE (Blocos de tablatura de instrumentos de cordas)
- ANNOTATION (Anotações e instruções de execução musical entre colchetes)
- REPEAT (Indicações de repetição, ex: x2, 2x, ||: :||)
- EMPTY (Linhas em branco)
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import List, Optional, Tuple, Dict, Any


class SemanticLineType(str, Enum):
    METADATA = "METADATA"
    SECTION = "SECTION"
    CHORD = "CHORD"
    LYRIC = "LYRIC"
    TABLATURE = "TABLATURE"
    ANNOTATION = "ANNOTATION"
    REPEAT = "REPEAT"
    NUMBER = "NUMBER"
    HEADER = "HEADER"
    EMPTY = "EMPTY"


# Palavras comuns da língua portuguesa e inglesa que NÃO podem ser confundidas com acordes
COMMON_WORDS = {
    # Português
    "A", "E", "O", "AS", "OS", "UM", "UMA", "UNS", "UMAS",
    "DE", "DA", "DO", "DAS", "DOS", "NA", "NO", "NAS", "NOS",
    "EM", "POR", "PARA", "COM", "SEM", "SOB", "SOBRE",
    "SE", "SO", "SÓ", "ME", "TE", "LHE", "NOS", "VOS", "LHES",
    "EU", "TU", "ELE", "ELA", "ELES", "ELAS", "VOCE", "VOCÊ", "VOCES", "VOCÊS",
    "NÓS", "VÓS", "MEU", "MINHA", "TEU", "TUA", "SEU", "SUA", "NOSSO", "NOSSA",
    "QUE", "QUEM", "QUANDO", "ONDE", "COMO", "PORQUE", "PORQUÊ", "QUAL",
    "HOJE", "ONTEM", "AMANHA", "AMANHÃ", "SEMPRE", "NUNCA", "JAMAIS",
    "VOU", "VAI", "VAMOS", "VAO", "VÃO", "FUI", "FOI", "FOMOS", "FORAM",
    "ERA", "ERAM", "SOU", "SOMOS", "SAO", "SÃO", "SER", "ESTAR", "ESTOU", "ESTA", "ESTÁ",
    "AMOR", "VIDA", "CORAÇÃO", "CORACAO", "DEUS", "SENHOR", "PAI", "FILHO",
    "BEM", "MAL", "MAIS", "MAS", "MENOS", "MUITO", "POUCO", "TUDO", "NADA",
    "DO", "RE", "MI", "FA", "SOL", "LA", "SI", # Palavras de solfejo por extenso quando isoladas
    # Inglês
    "IN", "ON", "AT", "TO", "FOR", "WITH", "WITHOUT",
    "YOU", "WE", "THEY", "HE", "SHE", "IT", "ME", "HIM", "HER", "US", "THEM",
    "MY", "YOUR", "HIS", "OUR", "THEIR", "ITS",
    "THE", "AN", "IS", "ARE", "WAS", "WERE", "BE", "BEEN", "BEING",
    "AND", "OR", "BUT", "SO", "BECAUSE", "AS", "IF",
    "ALL", "ANY", "SOME", "NO", "NOT", "NEVER", "ALWAYS"
}

# Regex estrita para identificação de acordes musicais
# Exemplos suportados: C, Am, G/B, F#m7(b5), Bbmaj7, C7M, C9, Dm/F, Cdim, C°, C+, Csus4, Cadd9, C7(9), C7(b9), C7(#9), C11, C13
CHORD_REGEX = re.compile(
    r'\b([A-G][#b]?'
    r'(?:m|min|maj|M|dim|aug|sus2|sus4|add9)?'
    r'(?:°|º)?'
    r'(?:[0-9]{1,2}|7M|Δ7|Δ)?'
    r'(?:\([^\)]+\))?'
    r'(?:/[A-G][#b]?)?)'
    # Limite final robusto: um acorde termina antes de espaço/delimitador/fim.
    # (usar \b quebrava acordes terminados em '#', ex.: "G#" virava "G".)
    r'(?=[\s)\]/|,;]|$)'
)


# Tablatura (padrão CifraClub / notação padrão de tab de cordas):
#   - Cada corda é uma linha iniciada pelo nome da corda (E A D G B e, com # ou b opcional)
#     seguido de '|', e uma sequência de traços, números (casas) e símbolos de técnica:
#       h=hammer-on  p=pull-off  b=bend  r=release  /=slide up  \\=slide down  s=slide
#       t=tapping  x=abafada  ~=vibrato  ()=nota fantasma  .=staccato
#   - Ex.: "E|--3-3--5h7--|", "e|-0h2p0-----|", "G|--7/9\\7--|", "B|---(5)---|"
# Também aceitamos linhas de tab SEM rótulo de corda ("|---0---2---|") e linhas densas
# de traços/números típicas de tablatura.
_TAB_TECHNIQUE_CHARS = r'-0-9/\\hpbrstx~().=|\s'
TABLATURE_LINE_REGEX = re.compile(
    r'^\s*[eEaAdDgGbB][#b]?\|[' + _TAB_TECHNIQUE_CHARS + r']+\|?\s*$'
)

# Seções canônicas mapeadas (Português & Inglês)
SECTION_MAP = {
    # Volta da introdução no meio da música (vem ANTES de "intro": a busca é por substring
    # e a reintro é uma passagem instrumental com a banda cheia, não um início calmo)
    "reintro": "INSTRUMENTAL",
    "re-intro": "INSTRUMENTAL",
    # Intro
    "intro": "INTRO",
    "introducao": "INTRO",
    "introdução": "INTRO",
    "entrada": "INTRO",
    # Verso / Estrofe
    "verso": "VERSE",
    "verse": "VERSE",
    "estrofe": "VERSE",
    "primeira parte": "VERSE",
    "segunda parte": "VERSE",
    "terceira parte": "VERSE",
    "quarta parte": "VERSE",
    "quinta parte": "VERSE",
    "parte 1": "VERSE",
    "parte 2": "VERSE",
    "parte 3": "VERSE",
    "parte 4": "VERSE",
    "parte i": "VERSE",
    "parte ii": "VERSE",
    "parte iii": "VERSE",
    "parte a": "VERSE",
    "parte b": "VERSE",
    "dedilhado": "VERSE",
    # Pré-refrão
    "pre-refrao": "PRE_CHORUS",
    "pré-refrão": "PRE_CHORUS",
    "pre-chorus": "PRE_CHORUS",
    "pre refrao": "PRE_CHORUS",
    "pré refrão": "PRE_CHORUS",
    "pré-refrao": "PRE_CHORUS",
    # Refrão / Coro (inclui variações do CifraClub: Pós-Refrão, Refrão Final)
    "refrao": "CHORUS",
    "refrão": "CHORUS",
    "chorus": "CHORUS",
    "coro": "CHORUS",
    "pos-refrao": "CHORUS",
    "pós-refrão": "CHORUS",
    "pós - refrão": "CHORUS",
    "pos - refrao": "CHORUS",
    "refrao final": "CHORUS",
    "refrão final": "CHORUS",
    "meio refrão": "CHORUS",
    # Ponte
    "ponte": "BRIDGE",
    "bridge": "BRIDGE",
    # Solo & Instrumental
    "solo": "SOLO",
    "guitar solo": "SOLO",
    "solo guitarra": "SOLO",
    "instrumental": "INSTRUMENTAL",
    "interludio": "INSTRUMENTAL",
    "interlúdio": "INSTRUMENTAL",
    "passagem": "INSTRUMENTAL",
    "break": "BREAK",
    # Tablatura / trechos técnicos do CifraClub (ex.: [Tab - Intro], [Tab - Passagem])
    "tab": "INSTRUMENTAL",
    "tablatura": "INSTRUMENTAL",
    "riff": "INSTRUMENTAL",
    "base": "INSTRUMENTAL",
    "meio": "INSTRUMENTAL",
    # Finalização
    "outro": "OUTRO",
    "final": "OUTRO",
    "fim": "OUTRO",
    "termino": "OUTRO",
    "término": "OUTRO",
    "coda": "OUTRO",
    "encerramento": "OUTRO"
}


@dataclass
class ClassifiedLine:
    """Resultado da análise semântica de uma linha individual da cifra."""
    line_number: int                    # Linha 1-indexada no documento original
    line_type: SemanticLineType
    raw_text: str
    cleaned_text: str
    section_name: Optional[str] = None
    section_type: Optional[str] = None
    extracted_metadata: Dict[str, Any] = field(default_factory=dict)
    chord_tokens: List[Tuple[str, int]] = field(default_factory=list) # (symbol, column_index)
    repeat_count: int = 1


class ChartSemanticClassifier:
    """Classificador semântico especializado em documentos de cancioneiro e cifras populares."""

    @classmethod
    def _clean_token(cls, token: str) -> str:
        """Remove pontuação/delimitadores em volta de um token para avaliação harmônica."""
        tok = token.strip().strip("[]{}|:,%?*~\"'").strip()
        if tok.startswith("(") and tok.endswith(")"):
            tok = tok[1:-1].strip()
        return tok

    @classmethod
    def is_chord_shaped(cls, token: str) -> bool:
        """Verifica APENAS o formato harmônico do token (sem desambiguação de stopwords).

        Usado para decidir se uma linha inteira é uma linha de acordes. Ao contrário de
        ``is_chord_token``, NÃO rejeita 'A', 'E', 'B' etc., pois esses também são acordes.
        """
        tok = cls._clean_token(token)
        if not tok:
            return False
        return CHORD_REGEX.fullmatch(tok) is not None

    @classmethod
    def is_chord_token(cls, token: str) -> bool:
        """Determina se um token é um acorde, aplicando desambiguação de stopwords.

        Deve ser usado apenas em linhas AMBÍGUAS (mistura de acorde e letra). Em uma linha
        comprovadamente de acordes, use ``is_chord_shaped`` para não descartar acordes que
        coincidem com palavras comuns (ex.: os acordes A, E, B).
        """
        tok = cls._clean_token(token)
        if not tok:
            return False

        # Palavras minúsculas isoladas 'a', 'e', 'o' são artigos ou conjunções
        if tok in ("a", "e", "o"):
            return False

        # Palavras comuns da língua portuguesa ou inglesa
        if tok.upper() in COMMON_WORDS:
            return False

        # Caso especial: 'Amor' ou 'Do' com pontuação
        if tok.upper() == "AMOR" or tok.upper() == "DO":
            return False

        # Verifica formato harmônico
        return cls.is_chord_shaped(tok)

    @classmethod
    def is_tablature_line(cls, line: str) -> bool:
        """Identifica linhas de tablatura de cordas para IGNORÁ-LAS na harmonia.

        Reconhece o padrão do CifraClub: rótulo de corda + '|' + traços/casas/técnicas,
        inclusive linhas sem rótulo e linhas densas de traços típicas de tab.
        """
        trimmed = line.strip()
        if not trimmed:
            return False
        # 1. Padrão canônico: [corda]|....  (com ou sem rótulo de corda)
        if TABLATURE_LINE_REGEX.match(trimmed):
            return True
        if re.search(r'^[eEaAdDgGbB]?[#b]?\|[' + _TAB_TECHNIQUE_CHARS + r']{4,}\|?', trimmed):
            return True
        # 2. Heurística: linha densa de tablatura — muitos traços com casas/técnicas,
        #    sem parecer letra (contém '-' em sequência e dígitos, poucos espaços/letras).
        if trimmed.count('-') >= 4:
            tabish = sum(1 for c in trimmed if c in '-0123456789|/\\hpbrstx~().')
            if tabish / max(1, len(trimmed)) >= 0.6 and re.search(r'-{2,}', trimmed):
                return True
        return False

    @classmethod
    def parse_metadata_line(cls, line: str) -> Optional[Dict[str, Any]]:
        """Extrai metadados estruturados (Tom, BPM, Compasso, Afinação, Capo) de uma linha."""
        stripped = line.strip()
        if not stripped:
            return None

        meta: Dict[str, Any] = {}

        # 1. Tom / Tonalidade / Key
        # Exemplos: "Tom: G", "Tom: Am", "Tom: A (com forma de G)", "Tonalidade: C#m", "Key: F#m"
        tom_match = re.search(r'^(?:tom|tonalidade|key)\s*[:=]\s*([A-G][#b]?(?:m|min|maj|M)?)(?:\s|\(|@|$)', stripped, re.IGNORECASE)
        if tom_match:
            token = tom_match.group(1).strip()
            root = token[:2] if len(token) >= 2 and token[1] in "#b" else token[:1]
            suffix = token[len(root):]
            root = root[0].upper() + root[1:]
            mode_word = re.match(r'\s*(maior|menor|major|minor)\b',
                                 stripped[tom_match.end(1):], re.IGNORECASE)
            minor = suffix in ("m", "min") or (suffix == "" and mode_word is not None and
                                                mode_word.group(1).lower() in ("menor", "minor"))
            meta["key"] = f"{root} {'Minor' if minor else 'Major'}"
            from app.music.key_timeline import parse_key_anchor
            anchor = parse_key_anchor(stripped)
            if anchor is not None:
                meta["key_anchor"] = anchor

        # 2. BPM / Tempo / Andamento
        # Exemplos: "165 bpm", "165 BPM", "BPM: 165", "Tempo: 165", "Andamento: 165 bpm", "[Ritmo Padrão] 165 bpm"
        bpm_match = re.search(r'(?:(?:bpm|tempo|andamento)\s*[:=]?\s*([0-9]{2,3})|([0-9]{2,3})\s*bpm)', stripped, re.IGNORECASE)
        if bpm_match:
            bpm_str = bpm_match.group(1) or bpm_match.group(2)
            if bpm_str:
                meta["bpm"] = float(bpm_str)

        # 3. Compasso / Time Signature / Meter
        # Exemplos: "Compasso: 4/4", "4/4", "3/4", "6/8", "12/8", "2/4"
        meter_match = re.search(r'(?:compasso|meter|time\s*signature)\s*[:=]\s*([0-9]{1,2}/[0-9]{1,2})', stripped, re.IGNORECASE)
        if not meter_match and re.fullmatch(r'^(?:compasso\s*)?[0-9]{1,2}/[0-9]{1,2}$', stripped, re.IGNORECASE):
            meter_match = re.search(r'([0-9]{1,2}/[0-9]{1,2})', stripped)
        if meter_match:
            meta["meter"] = meter_match.group(1)

        # 4. Afinação
        # Exemplos: "Afinação: E A D G B E", "Afinação: Eb Ab Db Gb Bb Eb"
        afin_match = re.search(r'^(?:afina[cç][aã]o|tuning)\s*[:=]\s*(.+)$', stripped, re.IGNORECASE)
        if afin_match:
            meta["tuning"] = afin_match.group(1).strip()

        # 5. Capotraste / Capo
        # Exemplos: "Capotraste: 2ª casa", "Capo: 2", "Capotraste na 2ª casa" (CifraClub),
        #           "Capotraste: Sem capotraste"
        capo_match = re.search(r'^(?:capotraste|capo)\b\s*(?:[:=]\s*|n[ao]\s+)?(.+)$', stripped, re.IGNORECASE)
        if capo_match:
            meta["capo"] = capo_match.group(1).strip()

        # 6. Título e Artista
        title_match = re.search(r'^(?:t[ií]tulo|title|can[cç][aã]o)\s*[:=]\s*(.+)$', stripped, re.IGNORECASE)
        if title_match:
            meta["title"] = title_match.group(1).strip()

        artist_match = re.search(r'^(?:artista|artist|banda|band|autor|composi[cç][aã]o\s*de)\s*[:=]\s*(.+)$', stripped, re.IGNORECASE)
        if artist_match:
            meta["artist"] = artist_match.group(1).strip()

        return meta if meta else None

    @classmethod
    def parse_section_line(cls, line: str) -> Optional[Tuple[str, str]]:
        """Identifica cabeçalhos formais de seção musical (ex: [Intro], [Refrão])."""
        stripped = line.strip()
        if not stripped:
            return None

        bracket_match = re.match(r'^\[(.*?)\]$', stripped)
        dash_match = re.match(r'^(?:--+|==+)\s*([A-Za-zÀ-ÿ0-9\s_-]+)\s*(?:--+|==+)$', stripped)

        raw_header = None
        if bracket_match:
            raw_header = bracket_match.group(1).strip()
        elif dash_match:
            raw_header = dash_match.group(1).strip()
        else:
            lower = stripped.lower()
            if lower in SECTION_MAP:
                raw_header = stripped

        if not raw_header:
            return None

        lower_header = raw_header.lower()

        # Anotações que não são seções estruturais
        if any(ann in lower_header for ann in ("repete", "repeat", "entra", "bateria", "somente", "voz", "ritmo", "varia")):
            return None

        # Mapeia para SectionType canônico
        sec_type = "UNKNOWN"
        for keyword, mapped_type in SECTION_MAP.items():
            if keyword in lower_header:
                sec_type = mapped_type
                break

        return raw_header, sec_type

    @classmethod
    def parse_annotation_line(cls, line: str) -> Optional[str]:
        """Identifica instruções de performance entre colchetes (ex: [Repete 2x], [Entra bateria])."""
        stripped = line.strip()
        if not (stripped.startswith('[') and stripped.endswith(']')):
            return None

        content = stripped[1:-1].strip()
        lower = content.lower()

        annotation_keywords = (
            "repete", "repeat", "entra", "sai", "bateria", "voz",
            "somente", "ritmo", "variação", "variaçao", "fade", "stop"
        )
        if any(kw in lower for kw in annotation_keywords):
            return content

        return None

    @classmethod
    def classify_line(cls, line: str, line_number: int = 1) -> ClassifiedLine:
        """Classifica uma linha do documento em sua respectiva categoria semântica."""
        stripped = line.strip()

        # 1. Linha vazia
        if not stripped:
            return ClassifiedLine(
                line_number=line_number,
                line_type=SemanticLineType.EMPTY,
                raw_text=line,
                cleaned_text=""
            )

        # 2. Tablatura
        if cls.is_tablature_line(stripped):
            return ClassifiedLine(
                line_number=line_number,
                line_type=SemanticLineType.TABLATURE,
                raw_text=line,
                cleaned_text=stripped
            )

        # 3. Metadados estruturados (Tom, BPM, Compasso, etc.)
        meta = cls.parse_metadata_line(stripped)
        if meta:
            return ClassifiedLine(
                line_number=line_number,
                line_type=SemanticLineType.METADATA,
                raw_text=line,
                cleaned_text=stripped,
                extracted_metadata=meta
            )

        # 4. Cabeçalho de Seção
        sec = cls.parse_section_line(stripped)
        if sec:
            sec_name, sec_type = sec
            return ClassifiedLine(
                line_number=line_number,
                line_type=SemanticLineType.SECTION,
                raw_text=line,
                cleaned_text=stripped,
                section_name=sec_name,
                section_type=sec_type
            )

        # 5. Anotação / Instrução de performance
        ann = cls.parse_annotation_line(stripped)
        if ann:
            return ClassifiedLine(
                line_number=line_number,
                line_type=SemanticLineType.ANNOTATION,
                raw_text=line,
                cleaned_text=stripped,
                extracted_metadata={"instruction": ann}
            )

        # 6. Contagens, telefones, anos e outros números nunca são harmonia.
        # Esta classificação positiva impede que qualquer caminho posterior tente
        # reinterpretar números como uma sequência de acordes.
        if re.fullmatch(r'[0-9\s]+', stripped):
            return ClassifiedLine(
                line_number=line_number,
                line_type=SemanticLineType.NUMBER,
                raw_text=line,
                cleaned_text=stripped
            )

        # 7. Avaliação de Acordes vs Letra
        # Tokeniza respeitando posições
        tokens = [t for t in re.split(r'[\s|:,%]+', stripped) if t]
        repeat_count = 1

        # Verifica repetição na linha (ex: x2, 2x)
        rep_match = re.search(r'\b(?:x([2-9])|([2-9])x)\b', stripped, re.IGNORECASE)
        if rep_match:
            cnt = rep_match.group(1) or rep_match.group(2)
            repeat_count = int(cnt)

        # Todos os candidatos com formato de acorde (sem descartar A/E/B ainda), com posição
        shaped_matches: List[Tuple[str, int]] = [
            (m.group(1).strip(), m.start())
            for m in CHORD_REGEX.finditer(line)
            if cls.is_chord_shaped(m.group(1))
        ]
        # Ignora o marcador de repetição (x2/2x) e parênteses/traços soltos — ex.: a
        # passagem "( G  D  Em  C )" do Cifra Club — na contagem de tokens de letra.
        content_tokens = [t for t in tokens
                          if not re.fullmatch(r'(?i)x[2-9]|[2-9]x|[()\[\]{}\-–—.]+', t)]

        # Uma linha é de ACORDES quando todos os seus tokens têm formato harmônico.
        # Nesse caso preservamos TODOS os acordes, mesmo os que coincidem com palavras
        # comuns (A, E, B), corrigindo a perda silenciosa de acordes.
        all_shaped = bool(content_tokens) and all(cls.is_chord_shaped(t) for t in content_tokens)

        if all_shaped:
            chord_tokens = shaped_matches
            is_chord = True
        else:
            # Linha ambígua (mistura acorde + letra): aplica desambiguação por stopwords
            # para não confundir palavras da letra com acordes.
            chord_tokens = [(tok, pos) for tok, pos in shaped_matches if cls.is_chord_token(tok)]
            ratio = len(chord_tokens) / len(content_tokens) if content_tokens else 0.0
            is_chord = bool(chord_tokens) and ratio >= 0.50

            # Reforço por ESPAÇAMENTO (padrão de cifra sobre a letra): acordes numa linha
            # de cifra costumam estar isolados ou separados por 2+ espaços (alinhados às
            # sílabas). Se a linha tem esse espaçamento largo e NÃO contém palavras de letra
            # (tokens com minúsculas), tratamos como linha de acordes mesmo com razão menor.
            if not is_chord and chord_tokens:
                wide_spaced = bool(re.search(r'\S {2,}\S', stripped))
                non_chord = [t for t in content_tokens
                             if not cls.is_chord_shaped(t)
                             and not re.fullmatch(r'(?i)x[2-9]|[2-9]x', t)]
                has_lyric_word = any(re.search(r'[a-zà-öø-ÿ]', t) for t in non_chord)
                if wide_spaced and not has_lyric_word and ratio >= 0.34:
                    is_chord = True

        if is_chord:
            return ClassifiedLine(
                line_number=line_number,
                line_type=SemanticLineType.CHORD,
                raw_text=line,
                cleaned_text=stripped,
                chord_tokens=chord_tokens,
                repeat_count=repeat_count
            )

        # 7. Caso padrão: Linha de Letra pura
        return ClassifiedLine(
            line_number=line_number,
            line_type=SemanticLineType.LYRIC,
            raw_text=line,
            cleaned_text=stripped,
            repeat_count=repeat_count
        )
