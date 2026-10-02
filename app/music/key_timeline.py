"""Mudanças de tom na cifra (modulações) e a linha do tempo de tonalidades.

Sintaxe na cifra (texto livre do editor):

    Tom: G            <- antes de qualquer acorde: tom INICIAL
    ...
    Tom: A            <- depois que a música começou: MUDANÇA DE TOM
    A          E          o novo tom vale a partir do próximo acorde tocável
    ...
    Tom: Bb @ compasso 33           <- âncora explícita (opcional)
    Tom: C @ compasso 41 tempo 3

Uma mudança de tom é metadado harmônico: não abre nem fecha seção (pode
acontecer no meio de um refrão). Pode haver quantas forem necessárias.

O "Tom:" do Cifra Club é o tom SOANTE (forma + capotraste); as mudanças seguem a
mesma convenção, então capo e transposição global tratam todas do mesmo jeito.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.music.theory import note_to_pc
from app.music.transposition import parse_key_root_mode, transpose_note_name


def normalize_key(key: str) -> str:
    """'A', 'Am', 'A menor', 'Bb Major' -> 'A Major', 'A Minor', 'Bb Major'."""
    root, mode = parse_key_root_mode(key or "C")
    written = (key or "").strip().split()[0] if (key or "").strip() else root
    if len(written) >= 2 and written[1] in "#b":
        root = written[0].upper() + written[1]
    return f"{root} {'Minor' if mode == 'minor' else 'Major'}"


def key_semitone_delta(from_key: str, to_key: str) -> int:
    """Menor deslocamento (−5..+6) entre as tônicas de dois tons."""
    a = note_to_pc(parse_key_root_mode(from_key)[0])
    b = note_to_pc(parse_key_root_mode(to_key)[0])
    if a < 0 or b < 0:
        return 0
    delta = (b - a) % 12
    return delta - 12 if delta > 6 else delta


def transpose_key(key: str, semitones: int, use_flats: bool = False) -> str:
    root, mode = parse_key_root_mode(key)
    return f"{transpose_note_name(root, semitones, use_flats)} {'Minor' if mode == 'minor' else 'Major'}"


@dataclass
class KeyChangeEvent:
    """Uma modulação declarada pelo usuário na cifra."""
    id: str
    source_line_number: int
    anchor_bar: int                     # compasso ESCRITO da cifra onde o novo tom entra
    anchor_beat: float = 1.0
    anchor_event_index: int = -1        # ordem do acorde escrito (sem expandir repetições)
    from_key: str = "--"
    to_key: str = "--"
    semitone_delta: int = 0
    source: str = "USER_CHART"
    confidence: float = 1.0
    explicit_anchor: bool = False       # "@ compasso N" em vez do próximo acorde

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "source_line_number": self.source_line_number,
            "anchor_bar": self.anchor_bar, "anchor_beat": self.anchor_beat,
            "anchor_event_index": self.anchor_event_index,
            "from_key": self.from_key, "to_key": self.to_key,
            "semitone_delta": self.semitone_delta, "source": self.source,
            "confidence": self.confidence, "explicit_anchor": self.explicit_anchor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyChangeEvent":
        return cls(
            id=str(data.get("id", "key_01")),
            source_line_number=int(data.get("source_line_number", 0)),
            anchor_bar=int(data.get("anchor_bar", 1)),
            anchor_beat=float(data.get("anchor_beat", 1.0)),
            anchor_event_index=int(data.get("anchor_event_index", -1)),
            from_key=str(data.get("from_key", "--")), to_key=str(data.get("to_key", "--")),
            semitone_delta=int(data.get("semitone_delta", 0)),
            source=str(data.get("source", "USER_CHART")),
            confidence=float(data.get("confidence", 1.0)),
            explicit_anchor=bool(data.get("explicit_anchor", False)),
        )


@dataclass
class KeyRegion:
    index: int
    key: str
    start_bar: int
    start_beat: float = 1.0
    end_bar: Optional[int] = None       # último compasso da região (None = até o fim)


@dataclass
class KeyTimeline:
    """Tom vigente em cada ponto da cifra (compasso escrito, tempo)."""
    initial_key: str
    events: List[KeyChangeEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.events = sorted(self.events, key=lambda e: (e.anchor_bar, e.anchor_beat,
                                                         e.source_line_number))

    @property
    def regions(self) -> List[KeyRegion]:
        regions = [KeyRegion(0, self.initial_key, 1, 1.0)]
        for event in self.events:
            regions[-1].end_bar = max(regions[-1].start_bar, event.anchor_bar - (
                1 if event.anchor_beat <= 1.0 else 0))
            regions.append(KeyRegion(len(regions), event.to_key, event.anchor_bar, event.anchor_beat))
        return regions

    @staticmethod
    def _after(bar: int, beat: float, event: KeyChangeEvent) -> bool:
        return (bar, beat) >= (event.anchor_bar, event.anchor_beat)

    def region_index_at(self, bar: int, beat: float = 1.0) -> int:
        index = 0
        for i, event in enumerate(self.events, start=1):
            if self._after(bar, beat, event):
                index = i
        return index

    def key_at(self, bar: int, beat: float = 1.0) -> str:
        index = self.region_index_at(bar, beat)
        return self.events[index - 1].to_key if index else self.initial_key

    def next_change_after(self, bar: int, beat: float = 1.0) -> Optional[KeyChangeEvent]:
        for event in self.events:
            if not self._after(bar, beat, event):
                return event
        return None

    def transposed(self, semitones: int, use_flats: bool = False) -> "KeyTimeline":
        return KeyTimeline(transpose_key(self.initial_key, semitones, use_flats),
                           [transpose_event(e, semitones, use_flats) for e in self.events])


def transpose_event(event: KeyChangeEvent, semitones: int,
                    use_flats: bool = False) -> KeyChangeEvent:
    """A mesma modulação, deslocada junto com a música (os intervalos se mantêm)."""
    moved = KeyChangeEvent(**event.to_dict())
    moved.from_key = transpose_key(event.from_key, semitones, use_flats) if event.from_key != "--" else "--"
    moved.to_key = transpose_key(event.to_key, semitones, use_flats)
    return moved


def parse_key_anchor(text: str) -> Optional[Tuple[int, float]]:
    """'... @ compasso 17 tempo 3' -> (17, 3.0); sem âncora -> None."""
    import re
    match = re.search(r'@\s*(?:compasso|comp\.?|bar|c)\s*(\d{1,4})'
                      r'(?:\s*(?:tempo|beat|t)\s*(\d{1,2}(?:[.,]\d+)?))?', text, re.IGNORECASE)
    if not match:
        return None
    beat = float(match.group(2).replace(",", ".")) if match.group(2) else 1.0
    return max(1, int(match.group(1))), max(1.0, beat)
