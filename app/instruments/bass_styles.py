"""Biblioteca de estilos do baixista (vocabulário rítmico e melódico por estilo).

Um estilo é DADO, não código: levadas (células de um compasso em semicolcheias),
viradas e as probabilidades de cada recurso por nível de criatividade. O
``BassArranger`` escolhe e combina esse vocabulário de acordo com a música.

Papéis das notas (``role``):
  R      nota do baixo do acorde (fundamental ou o baixo da inversão da cifra)
  R8     R uma oitava acima
  5 / 3  quinta / terça do acorde
  PUSH   antecipação: no contratempo antes de uma troca, já toca o PRÓXIMO acorde
         (ligado ao tempo 1); sem troca à frente, vira R
  S-n    n graus da escala ABAIXO do alvo (próximo acorde) — subida até ele
  S+n    n graus da escala ACIMA do alvo — descida até ele
  C-1/C+1  aproximação cromática por baixo/por cima do alvo
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Posições em semicolcheias num compasso 4/4: 0=tempo 1, 4=tempo 2, 8=tempo 3,
# 12=tempo 4; 2/6/10/14 = contratempos ("e"); ímpares = semicolcheias.
Hit = Tuple[int, str, int, int]   # (posição, papel, velocidade, duração em semicolcheias)


@dataclass(frozen=True)
class Groove:
    name: str
    label: str
    tier: int                      # 0 calmo · 1 moderado · 2 cheio · 3 clímax
    hits: Tuple[Hit, ...]
    weight: float = 1.0
    max_bpm: float = 999.0
    min_bpm: float = 0.0


@dataclass(frozen=True)
class Fill:
    name: str
    label: str
    start: int                     # posição a partir da qual substitui a levada
    hits: Tuple[Hit, ...]
    max_bpm: float = 999.0
    min_tier: int = 0
    weight: float = 1.0


@dataclass(frozen=True)
class BassStyle:
    name: str
    label: str
    grooves: Tuple[Groove, ...]
    fills: Tuple[Fill, ...]
    # Probabilidades por nível de criatividade (índice 1..5; o 0 não é usado).
    approach_prob: Tuple[float, ...] = (0, 0.0, 0.15, 0.35, 0.55, 0.70)
    push_prob: Tuple[float, ...] = (0, 0.0, 0.10, 0.25, 0.40, 0.55)
    fill_section_prob: Tuple[float, ...] = (0, 0.0, 0.30, 0.60, 0.80, 0.90)
    fill_phrase_prob: Tuple[float, ...] = (0, 0.0, 0.0, 0.20, 0.40, 0.60)
    ornament_prob: Tuple[float, ...] = (0, 0.0, 0.0, 0.20, 0.35, 0.50)
    max_tier: Tuple[int, ...] = (0, 1, 2, 3, 3, 3)
    # Cromatismo (notas fora da escala) nas aproximações/viradas, por criatividade.
    chromatic_prob: Tuple[float, ...] = (0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # Manter a levada quando a banda segue igual dentro da mesma seção.
    keep_groove_prob: Tuple[float, ...] = (0, 1.0, 1.0, 0.95, 0.75, 0.5)
    # Seções em que a intensidade sobe/desce além do que a energia do áudio diz.
    section_bias: Dict[str, float] = field(default_factory=dict)
    phrase_bars: int = 4


def _eighths(role_at=None, accents=(0, 8), vel=(104, 84), dur=2) -> Tuple[Hit, ...]:
    role_at = role_at or {}
    return tuple((pos, role_at.get(pos, "R"), vel[0] if pos in accents else vel[1], dur)
                 for pos in range(0, 16, 2))


GOSPEL_WORSHIP = BassStyle(
    name="GOSPEL_WORSHIP",
    label="Gospel / Worship",
    grooves=(
        # --- tier 0: calmo (intro, versos suaves, finais) ---------------------
        Groove("semibreve", "Nota longa (semibreve)", 0, ((0, "R", 100, 15),), weight=1.0),
        Groove("minimas", "Mínimas", 0, ((0, "R", 100, 7), (8, "R", 88, 7)), weight=1.2),
        Groove("longa_antecipa", "Nota longa + antecipação", 0,
               ((0, "R", 100, 13), (14, "PUSH", 92, 2)), weight=1.0),
        # --- tier 1: moderado (verso 2, pré-refrão) ---------------------------
        Groove("seminimas", "Pulso de semínimas", 1,
               ((0, "R", 102, 3), (4, "R", 84, 3), (8, "R", 96, 3), (12, "R", 84, 3)), weight=1.0),
        Groove("balada", "Balada worship (1 · 2e · 3 · 4e)", 1,
               ((0, "R", 102, 6), (6, "R", 84, 2), (8, "R", 96, 6), (14, "PUSH", 90, 2)), weight=1.4),
        Groove("tonica_quinta", "Tônica e quinta", 1,
               ((0, "R", 102, 7), (8, "5", 90, 5), (14, "R", 84, 2)), weight=0.8),
        Groove("pontuada", "Semínima pontuada", 1,
               ((0, "R", 102, 6), (6, "R", 86, 6), (12, "R", 90, 2), (14, "PUSH", 90, 2)), weight=1.0),
        # --- tier 2: cheio (refrão) -------------------------------------------
        Groove("colcheias", "Colcheias pulsantes", 2, _eighths(), weight=1.4, max_bpm=168),
        Groove("colcheias_oitava", "Colcheias com oitava", 2,
               _eighths({6: "R8", 14: "R8"}), weight=1.0, max_bpm=160),
        Groove("colcheias_antecipa", "Colcheias + antecipação", 2,
               _eighths({14: "PUSH"}), weight=1.2, max_bpm=168),
        Groove("seminima_colcheias", "Semínima e colcheias", 2,
               ((0, "R", 104, 3), (4, "R", 84, 2), (6, "R", 86, 2), (8, "R", 98, 3),
                (12, "R", 84, 2), (14, "PUSH", 90, 2)), weight=1.0),
        Groove("seminimas_cheias", "Semínimas cheias", 2,
               ((0, "R", 106, 3), (4, "R", 92, 3), (8, "R", 100, 3), (12, "R", 92, 3)),
               weight=0.6, min_bpm=140),
        # --- tier 3: clímax (último refrão, ponte forte) ----------------------
        Groove("colcheias_quinta_oitava", "Colcheias com quinta e oitava", 3,
               _eighths({4: "R8", 10: "5", 14: "PUSH"}, vel=(108, 88)), weight=1.2, max_bpm=160),
        Groove("sincope", "Síncope worship", 3,
               ((0, "R", 110, 3), (3, "R", 86, 1), (6, "R", 96, 2), (8, "R", 102, 2),
                (10, "R", 86, 2), (12, "R8", 92, 2), (14, "PUSH", 96, 2)), weight=1.0, max_bpm=132),
        Groove("galope", "Galope (colcheia + duas semicolcheias)", 3,
               ((0, "R", 108, 2), (2, "R", 82, 1), (3, "R", 82, 1), (4, "R", 96, 2), (6, "R", 82, 1),
                (7, "R", 82, 1), (8, "R", 102, 2), (10, "R", 82, 1), (11, "R", 82, 1),
                (12, "R", 96, 2), (14, "PUSH", 92, 2)), weight=0.8, max_bpm=124),
        Groove("colcheias_fortes", "Colcheias fortes", 3,
               _eighths(accents=(0, 4, 8, 12), vel=(110, 90)), weight=1.0, max_bpm=168),
    ),
    fills=(
        Fill("subida_colcheias", "Virada: subida em colcheias", 8,
             ((8, "R", 98, 2), (10, "S-3", 88, 2), (12, "S-2", 90, 2), (14, "S-1", 94, 2))),
        Fill("descida_colcheias", "Virada: descida em colcheias", 8,
             ((8, "R8", 98, 2), (10, "S+3", 88, 2), (12, "S+2", 90, 2), (14, "S+1", 94, 2)),
             weight=0.8),
        Fill("oitava_cromatica", "Virada: oitava e cromatismo", 8,
             ((8, "R", 98, 2), (10, "R8", 90, 2), (12, "5", 90, 2), (14, "C-1", 94, 2)), weight=0.9),
        Fill("subida_semicolcheias", "Virada: subida em semicolcheias", 12,
             ((12, "S-3", 90, 1), (13, "S-2", 88, 1), (14, "S-1", 92, 1), (15, "C-1", 94, 1)),
             max_bpm=118, min_tier=1, weight=1.1),
        Fill("descida_semicolcheias", "Virada: descida em semicolcheias", 12,
             ((12, "R8", 92, 1), (13, "S+2", 88, 1), (14, "S+1", 90, 1), (15, "C+1", 92, 1)),
             max_bpm=118, min_tier=1, weight=0.9),
        Fill("pausa_antecipa", "Virada: respiro + antecipação", 8,
             ((8, "R", 96, 4), (14, "PUSH", 98, 2)), weight=0.7),
    ),
    section_bias={"CHORUS": 0.15, "REFRAO": 0.15, "BRIDGE": 0.10, "PONTE": 0.10,
                  "INTRO": -0.15, "OUTRO": -0.15, "FINAL": -0.15},
)

STYLES: Dict[str, BassStyle] = {GOSPEL_WORSHIP.name: GOSPEL_WORSHIP}
DEFAULT_STYLE = GOSPEL_WORSHIP.name

CREATIVITY_LABELS = {1: "Simples", 2: "Contido", 3: "Equilibrado", 4: "Criativo", 5: "Solto"}


def style_labels() -> List[str]:
    return [style.label for style in STYLES.values()]


def style_by_label(label: str) -> BassStyle:
    for style in STYLES.values():
        if style.label == label:
            return style
    return STYLES[DEFAULT_STYLE]
