"""Intenção musical do baixista: O QUE DEVO TOCAR AGORA.

A teoria (BassDecisionEngine, papéis R/5/3/APP...) diz o que é harmonicamente
SEGURO. Esta camada decide o que faz SENTIDO MUSICAL neste compasso, olhando:

- a SEÇÃO da cifra (verso, pré-refrão, refrão...), quanto dela já passou e quanto
  falta — cada tipo de seção é um PRIOR (não uma regra) por estilo;
- o GROOVE ouvido no áudio (``GrooveProfile``): a música real corrige o prior;
- a MEMÓRIA da música: o 2º refrão lembra o 1º, o verso 2 lembra o verso 1;
- a CONFIANÇA do acompanhamento: menos certeza ⇒ menos risco (nunca o contrário);
- a próxima MODULAÇÃO da cifra (o baixista sabe antes que ela chegue).

O resultado é um ``BassMusicalIntent``: energia e densidade-alvo, articulação,
vocabulário permitido, política de viradas e aproximações. A CRIATIVIDADE (modo
simples x solto) só amplia o VOCABULÁRIO possível; a densidade vem da música —
"avançado" NUNCA significa automaticamente "mais notas".

Tudo é local, determinístico e barato (uma decisão por compasso).
"""

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.analysis.groove_feel import GrooveProfile, SPACIOUS

# ------------------------------------------------------------------ códigos de razão
REASON_SECTION_CONTINUITY = "REASON_SECTION_CONTINUITY"
REASON_SECTION_MEMORY = "REASON_SECTION_MEMORY"
REASON_LOW_ENERGY = "REASON_LOW_ENERGY"
REASON_HIGH_ENERGY = "REASON_HIGH_ENERGY"
REASON_BUILD = "REASON_SECTION_BUILD"
REASON_LONG_CHORD = "REASON_LONG_CHORD_SUSTAIN"
REASON_OBSERVE = "REASON_OBSERVATION_MODE"
REASON_LOW_CONFIDENCE = "REASON_LOW_CONFIDENCE_SAFE_FALLBACK"
REASON_GROOVE_PENDING = "REASON_GROOVE_CHANGE_PENDING"
REASON_GROOVE_TRANSITION = "REASON_GROOVE_TRANSITION"
REASON_AUDIO_OVERRIDES_PRIOR = "REASON_AUDIO_OVERRIDES_PRIOR"
REASON_FILL_SECTION_END = "REASON_FILL_SECTION_END"
REASON_FILL_PHRASE_END = "REASON_FILL_PHRASE_END"
REASON_APPROACH_KEY_CHANGE = "REASON_APPROACH_KEY_CHANGE"
REASON_DIRECT_KEY_CHANGE = "REASON_DIRECT_KEY_CHANGE"
REASON_PREPARE_KEY_CHANGE = "REASON_PREPARE_KEY_CHANGE"
REASON_FINAL = "REASON_FINAL_BAR"

# Densidade em ATAQUES DO BAIXO por compasso (referência: compasso de 4 tempos).
DENSITY_LEVELS = ("VERY_LOW", "LOW", "MEDIUM", "HIGH")
_LEVEL_ATTACKS = {"VERY_LOW": 1.0, "LOW": 2.0, "MEDIUM": 4.0, "HIGH": 7.0}


def density_level(attacks: float) -> str:
    return ("VERY_LOW" if attacks < 1.6 else "LOW" if attacks < 3.1 else
            "MEDIUM" if attacks < 5.6 else "HIGH")


@dataclass(frozen=True)
class SectionPrior:
    """Como esse tipo de seção COSTUMA soar no estilo (o áudio pode contrariar)."""
    energy_start: float
    energy_end: float
    attacks: Tuple[float, float]      # faixa típica de ataques do baixo por compasso
    articulation: str                 # LONG / MEDIUM / SHORT
    syncopation: float                # alvo 0..1
    fill_policy: str                  # NONE / RARE / PHRASE_END / SECTION_END
    variation_budget: float           # 0..1: quanto pode variar o motivo dentro da seção
    phrase_bars: int = 4
    register: str = "LOW"


def _priors(scale: float) -> Dict[str, SectionPrior]:
    a = lambda lo, hi: (lo * scale, hi * scale)  # noqa: E731 - tabela compacta
    return {
        "INTRO": SectionPrior(0.25, 0.32, a(1, 2), "LONG", 0.05, "NONE", 0.10),
        "VERSE": SectionPrior(0.28, 0.34, a(1, 3), "LONG", 0.10, "RARE", 0.20),
        "PRE_CHORUS": SectionPrior(0.36, 0.62, a(2, 4), "MEDIUM", 0.15, "SECTION_END", 0.30),
        "CHORUS": SectionPrior(0.60, 0.68, a(2, 6), "MEDIUM", 0.20, "SECTION_END", 0.35),
        "BRIDGE": SectionPrior(0.45, 0.72, a(1, 6), "MEDIUM", 0.20, "SECTION_END", 0.40),
        "INSTRUMENTAL": SectionPrior(0.50, 0.56, a(2, 6), "MEDIUM", 0.20, "PHRASE_END", 0.40),
        "SOLO": SectionPrior(0.50, 0.56, a(2, 6), "MEDIUM", 0.20, "PHRASE_END", 0.40),
        "BREAK": SectionPrior(0.15, 0.20, a(0.5, 1), "LONG", 0.00, "NONE", 0.00),
        "OUTRO": SectionPrior(0.30, 0.15, a(1, 2), "LONG", 0.05, "NONE", 0.10),
        "UNKNOWN": SectionPrior(0.40, 0.40, a(1, 4), "MEDIUM", 0.10, "RARE", 0.20),
    }


# Priors CONFIGURÁVEIS por família de estilo. Worship: versos espaçosos, pré-refrão
# crescendo, refrão presente. "GENERIC" (pop/rock) é um pouco mais movimentado.
SECTION_PRIORS: Dict[str, Dict[str, SectionPrior]] = {
    "WORSHIP": _priors(1.0),
    "GENERIC": _priors(1.4),
}
STYLE_PRIOR_FAMILY = {"GOSPEL_WORSHIP": "WORSHIP"}

_SECTION_ALIASES = (
    ("PRE", "PRE_CHORUS"), ("CHORUS", "CHORUS"), ("REFR", "CHORUS"), ("VERSE", "VERSE"),
    ("VERSO", "VERSE"), ("BRIDGE", "BRIDGE"), ("PONTE", "BRIDGE"), ("INTRO", "INTRO"),
    ("OUTRO", "OUTRO"), ("FINAL", "OUTRO"), ("BREAK", "BREAK"), ("SOLO", "SOLO"),
    ("INSTRUMENTAL", "INSTRUMENTAL"), ("INTERLUD", "INSTRUMENTAL"),
)


def normalize_section_type(section_type: str) -> str:
    upper = (section_type or "").upper().replace("-", "_").replace(" ", "_")
    for needle, canonical in _SECTION_ALIASES:
        if needle in upper:
            return canonical
    return "UNKNOWN"


# ------------------------------------------------------------------ energia
class SectionEnergyEnvelope:
    """Energia-alvo ao longo da seção: arco do prior, corrigido pelo áudio e pela memória."""

    AUDIO_WEIGHT = 0.65          # quanto o áudio (confiante) corrige o prior
    MEMORY_WEIGHT = 0.6          # quanto a ocorrência anterior da seção pesa no prior

    @staticmethod
    def prior_energy(prior: SectionPrior, progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        return prior.energy_start + (prior.energy_end - prior.energy_start) * p

    def target(self, prior: SectionPrior, progress: float, audio_energy: Optional[float],
               audio_confidence: float, memory_energy: Optional[float] = None) -> float:
        base = self.prior_energy(prior, progress)
        if memory_energy is not None:
            arc = base - prior.energy_start          # o arco da seção continua valendo
            base = (1 - self.MEMORY_WEIGHT) * base + self.MEMORY_WEIGHT * (memory_energy + arc)
        if audio_energy is None:
            return float(np.clip(base, 0.0, 1.0))
        w = self.AUDIO_WEIGHT * float(np.clip(audio_confidence, 0.0, 1.0))
        return float(np.clip((1 - w) * base + w * audio_energy, 0.0, 1.0))


# ------------------------------------------------------------------ identidade da seção
@dataclass
class SectionPerformanceProfile:
    section_key: str
    section_type: str
    occurrence: int
    base_energy: float
    target_attacks: float             # ataques do baixo por compasso
    groove_family: str
    subdivision: str
    syncopation_level: float
    articulation: str
    register: str
    fill_policy: str
    variation_budget: float
    phrase_length: int
    from_memory: bool = False
    bars: int = 0
    groove_changes: int = 0
    pending_family: str = "--"
    pending_bars: int = 0
    bars_since_change: int = 0
    energies: List[float] = field(default_factory=list)

    @property
    def density(self) -> str:
        return density_level(self.target_attacks)


@dataclass
class _SectionMemory:
    energy: float
    attacks: float
    groove_family: str
    articulation: str
    occurrences: int = 1


class SectionGrooveController:
    """Mantém a IDENTIDADE musical de cada seção (sem flip-flop compasso a compasso).

    Ao entrar numa seção, fixa um perfil (densidade, família de groove, articulação)
    a partir do prior do estilo + o que o áudio já mostra + a memória da mesma seção.
    Dentro dela, uma mudança só acontece com evidência PERSISTENTE (2+ compassos
    seguidos, confiança alta) e de forma gradual; depois de mudar, segura o novo
    estado alguns compassos.
    """

    MIN_PERSIST_BARS = 2
    MIN_HOLD_BARS = 3
    ATTACK_CHANGE = 1.4          # diferença de ataques/compasso que conta como mudança real
    MIN_CONFIDENCE = 0.55
    TRANSITION_RATE = 0.5        # fração do caminho percorrida por compasso numa transição
    BASS_SHARE = 0.45            # o baixo ataca ~metade dos pontos de ataque da banda

    def __init__(self, family: str = "WORSHIP", beats_per_bar: int = 4):
        self.family = family if family in SECTION_PRIORS else "GENERIC"
        self.beats_per_bar = max(1, int(beats_per_bar))
        self.reset()

    def reset(self) -> None:
        self._memory: Dict[str, _SectionMemory] = {}
        self._memory_by_type: Dict[str, _SectionMemory] = {}
        self.profile: Optional[SectionPerformanceProfile] = None

    def prior(self, section_type: str) -> SectionPrior:
        return SECTION_PRIORS[self.family][normalize_section_type(section_type)]

    def observed_attacks(self, groove: GrooveProfile) -> float:
        """Ataques do baixo por compasso que combinam com a levada ouvida.

        Num trecho calmo o baixo toca ~1/3 dos pontos de ataque da banda; num trecho
        forte, acompanha a maior parte deles (colcheias com a banda em colcheias).
        """
        slots = 16.0
        share = self.BASS_SHARE - 0.1 + 0.45 * float(np.clip(groove.energy, 0.0, 1.0))
        return float(np.clip(groove.onset_density * slots * share, 0.5, 8.0))

    def enter(self, section_key: str, section_type: str, occurrence: int,
              groove: Optional[GrooveProfile]) -> SectionPerformanceProfile:
        prior = self.prior(section_type)
        canonical = normalize_section_type(section_type)
        memory = self._memory.get(section_key) or self._memory_by_type.get(canonical)
        lo, hi = prior.attacks
        attacks = (lo + hi) / 2.0
        family = SPACIOUS if prior.articulation == "LONG" else "STEADY"
        subdivision = "QUARTER"
        if groove is not None and groove.confidence >= self.MIN_CONFIDENCE:
            observed = self.observed_attacks(groove)
            # Na entrada, o groove ouvido ainda é o da seção anterior: pesa pouco. Se a
            # seção nova soar mesmo diferente do prior, ``update_bar`` corrige (com
            # persistência) — a música real manda, sem trocar de identidade no susto.
            attacks = float(np.clip(0.7 * attacks + 0.3 * observed, lo * 0.5, hi + 2.0))
            family = groove.label if groove.label != "--" else family
            subdivision = groove.subdivision
        if memory is not None:
            # 2º refrão lembra o 1º: mesma identidade-base, pode evoluir com o áudio.
            attacks = 0.65 * memory.attacks + 0.35 * attacks
            family = memory.groove_family
        self.profile = SectionPerformanceProfile(
            section_key=section_key, section_type=canonical, occurrence=occurrence,
            base_energy=memory.energy if memory else prior.energy_start,
            target_attacks=attacks, groove_family=family, subdivision=subdivision,
            syncopation_level=prior.syncopation,
            articulation=memory.articulation if memory else prior.articulation,
            register=prior.register, fill_policy=prior.fill_policy,
            variation_budget=prior.variation_budget, phrase_length=prior.phrase_bars,
            from_memory=memory is not None)
        return self.profile

    def update_bar(self, groove: Optional[GrooveProfile]) -> Tuple[SectionPerformanceProfile, List[str]]:
        """Um compasso da seção passou; só muda com evidência persistente."""
        profile, reasons = self.profile, []
        if profile is None:
            raise RuntimeError("update_bar antes de enter")
        profile.bars += 1
        profile.bars_since_change += 1
        if groove is None or groove.confidence < self.MIN_CONFIDENCE:
            profile.pending_bars, profile.pending_family = 0, "--"
            return profile, [REASON_SECTION_CONTINUITY]
        profile.energies.append(groove.energy)
        observed = self.observed_attacks(groove)
        family_changed = groove.label not in ("--", profile.groove_family)
        attacks_changed = abs(observed - profile.target_attacks) >= self.ATTACK_CHANGE
        if not (family_changed or attacks_changed):
            profile.pending_bars, profile.pending_family = 0, "--"
            return profile, [REASON_SECTION_CONTINUITY]
        candidate = groove.label if family_changed else profile.groove_family
        profile.pending_bars = profile.pending_bars + 1 if candidate == profile.pending_family else 1
        profile.pending_family = candidate
        if (profile.pending_bars < self.MIN_PERSIST_BARS or
                profile.bars_since_change < self.MIN_HOLD_BARS):
            return profile, [REASON_SECTION_CONTINUITY, REASON_GROOVE_PENDING]
        # Mudança real e persistente: transição GRADUAL na densidade.
        profile.target_attacks += self.TRANSITION_RATE * (observed - profile.target_attacks)
        if family_changed:
            profile.groove_family = candidate
            profile.groove_changes += 1
            profile.bars_since_change = 0
            profile.pending_bars, profile.pending_family = 0, "--"
        reasons.append(REASON_GROOVE_TRANSITION)
        return profile, reasons

    def learn(self, position_confidence: float, groove_confidence: float) -> bool:
        """Fim da seção: guarda a identidade dela — só se o acompanhamento era confiável."""
        profile = self.profile
        if (profile is None or profile.bars < 2 or position_confidence < 0.65 or
                groove_confidence < self.MIN_CONFIDENCE or not profile.energies):
            return False
        entry = _SectionMemory(energy=float(np.mean(profile.energies)),
                               attacks=profile.target_attacks,
                               groove_family=profile.groove_family,
                               articulation=profile.articulation)
        previous = self._memory.get(profile.section_key)
        if previous is not None:
            entry.occurrences = previous.occurrences + 1
        self._memory[profile.section_key] = entry
        self._memory_by_type[profile.section_type] = entry
        return True

    def memory_energy(self, section_key: str, section_type: str) -> Optional[float]:
        memory = self._memory.get(section_key) or self._memory_by_type.get(
            normalize_section_type(section_type))
        return memory.energy if memory else None


# ------------------------------------------------------------------ memória de frase
class PhraseMemory:
    """Motivo rítmico da frase: repetição com variação, em vez de aleatoriedade.

    Guarda o ritmo (16 semicolcheias) escolhido no começo da frase; os compassos
    seguintes preferem ritmos parecidos. Cada seção lembra o seu motivo (o refrão 2
    começa pelo motivo do refrão 1).
    """

    SIMILAR = 0.6

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._section_motifs: Dict[str, np.ndarray] = {}
        self._motif: Optional[np.ndarray] = None
        self._section: str = ""
        self.bar_in_phrase = 0
        self.phrase_length = 4
        self.phrase_changes = 0
        self.motif_id = ""

    def begin_section(self, section_key: str, phrase_length: int) -> None:
        if self._section and self._motif is not None:
            self._section_motifs[self._section] = self._motif
        self._section = section_key
        self._motif = self._section_motifs.get(section_key)
        self.phrase_length = max(1, int(phrase_length))
        self.bar_in_phrase = 0

    @property
    def motif(self) -> Optional[np.ndarray]:
        return self._motif

    def bonus(self, rhythms: np.ndarray, variation_budget: float) -> np.ndarray:
        """Preferência por ritmos parecidos com o motivo (maior no meio da frase)."""
        if self._motif is None or rhythms is None or len(rhythms) == 0:
            return np.zeros(len(rhythms) if rhythms is not None else 0)
        motif = self._motif / (np.linalg.norm(self._motif) + 1e-9)
        norms = np.linalg.norm(rhythms, axis=1) + 1e-9
        similarity = (rhythms @ motif) / norms
        weight = 2.0 * (1.0 - variation_budget) if self.bar_in_phrase else 1.0 * (1.0 - variation_budget)
        return weight * similarity

    def record(self, rhythm: np.ndarray) -> None:
        rhythm = np.asarray(rhythm, dtype=np.float64)
        if self._motif is None:
            self._motif = rhythm
            self.motif_id = f"{self._section}#1"
        elif self.bar_in_phrase == 0:
            na, nb = np.linalg.norm(rhythm), np.linalg.norm(self._motif)
            similarity = float(rhythm @ self._motif / (na * nb)) if na > 0 and nb > 0 else 0.0
            if similarity < self.SIMILAR:
                self.phrase_changes += 1
                self.motif_id = f"{self._section}#{self.phrase_changes + 1}"
            self._motif = 0.5 * self._motif + 0.5 * rhythm
        self.bar_in_phrase = (self.bar_in_phrase + 1) % self.phrase_length

    @property
    def bars_left_in_phrase(self) -> int:
        return self.phrase_length - self.bar_in_phrase


# ------------------------------------------------------------------ modulação
KEEP = "KEEP"
PREPARE = "PREPARE"
APPROACH = "APPROACH"
DIRECT_CHANGE = "DIRECT_CHANGE"


class BassModulationPlanner:
    """Como o baixo atravessa uma mudança de tom anunciada na cifra.

    Modulação NÃO obriga virada: em worship lento a troca seca e sustentada é
    muitas vezes a melhor escolha. Aproximação só com energia, espaço, confiança
    e vocabulário para isso — e sempre pela escala do tom de DESTINO.
    """

    def plan(self, beats_until: float, beats_per_bar: int, energy: float, density: str,
             confidence: float, vocabulary_level: int, section_type: str) -> Tuple[str, str]:
        if beats_until is None or beats_until < 0 or beats_until > beats_per_bar:
            return KEEP, ""
        calm = density in ("VERY_LOW", "LOW") or energy < 0.42 or normalize_section_type(
            section_type) in ("INTRO", "OUTRO", "BREAK")
        if confidence < 0.7 or calm:
            return DIRECT_CHANGE, REASON_DIRECT_KEY_CHANGE
        if vocabulary_level >= 3 and energy >= 0.5:
            return APPROACH, REASON_APPROACH_KEY_CHANGE
        return PREPARE, REASON_PREPARE_KEY_CHANGE


# ------------------------------------------------------------------ intenção
@dataclass
class BassMusicalIntent:
    bar_number: int = 0
    section_type: str = "UNKNOWN"
    section_key: str = ""
    section_occurrence: int = 1
    section_progress: float = 0.0
    current_key: str = "--"
    next_key: str = "--"
    beats_until_key_change: float = -1.0
    current_chord: str = "--"
    next_chord: str = "--"
    energy_target: float = 0.4
    energy_trend: float = 0.0
    audio_energy: float = -1.0
    density_target: str = "LOW"
    attacks_cap: float = 2.0            # ataques do baixo por compasso
    groove_label: str = "--"
    groove_confidence: float = 0.0
    subdivision: str = "QUARTER"
    syncopation_target: float = 0.1
    articulation: str = "LONG"
    note_length_factor: float = 0.95
    register: str = "LOW"
    fill_probability: float = 0.0
    transition_fill_allowed: bool = False
    approach_note_policy: str = "NONE"  # NONE / DIATONIC
    push_allowed: bool = False
    sustain_preferred: bool = False
    phrase_length_bars: int = 4
    bars_left_in_phrase: int = 4
    variation_budget: float = 0.2
    complexity_ceiling: int = 5
    vocabulary_level: int = 1
    velocity_base: int = 90
    accent: int = 8
    modulation_action: str = KEEP
    fallback_pattern: str = ""          # "", SUSTAINED, ROOT, ROOT_FIFTH
    observation_mode: bool = False
    confidence: float = 1.0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class BarContext:
    """O que se sabe ANTES de tocar o compasso (cifra + passado do áudio)."""
    bar_number: int
    beats_per_bar: int = 4
    section_key: str = ""                 # nome da seção (chave da memória)
    section_type: str = "UNKNOWN"
    section_occurrence: int = 1
    section_progress: float = 0.0
    bars_until_section_change: int = 99
    chords: Sequence[Optional[str]] = ()
    next_chord: Optional[str] = None
    chord_duration_beats: float = 4.0     # quanto o acorde do tempo 1 dura (harmonia)
    current_key: str = "--"
    next_key: str = "--"
    beats_until_key_change: float = -1.0
    groove: Optional[GrooveProfile] = None
    position_confidence: float = 1.0
    chord_confidence: float = 1.0
    final_bar: bool = False
    compound: bool = False                # grade em colcheias de compasso composto


_FILL_BY_CREATIVITY = (0.0, 0.0, 0.35, 0.55, 0.7, 0.8)
_LENGTH = {"LONG": 0.97, "MEDIUM": 0.82, "SHORT": 0.6}


class BassMusicalIntentEngine:
    """Decide a intenção de cada compasso (seção + groove + memória + confiança)."""

    OBSERVE_BARS = 2
    ENERGY_ALPHA = 0.35
    MAX_ATTACK_STEP = 1.5

    def __init__(self, style_name: str = "GOSPEL_WORSHIP", creativity: int = 3,
                 beats_per_bar: int = 4):
        self.style_name = style_name
        self.creativity = max(1, min(5, int(creativity)))
        self.beats_per_bar = max(1, int(beats_per_bar))
        family = STYLE_PRIOR_FAMILY.get(style_name, "GENERIC")
        self.controller = SectionGrooveController(family, beats_per_bar)
        self.envelope = SectionEnergyEnvelope()
        self.phrases = PhraseMemory()
        self.modulation = BassModulationPlanner()
        self.diagnostics = BassBrainDiagnostics()
        self.reset()

    def reset(self) -> None:
        self.controller.reset()
        self.phrases.reset()
        self.diagnostics.reset()
        self._section_key: Optional[str] = None
        self._last_position_confidence = 1.0
        self._last_groove_confidence = 0.0
        self._audio_energy: Optional[float] = None
        self._bars = 0
        self.last_intent: Optional[BassMusicalIntent] = None

    def plan_bar(self, ctx: BarContext) -> BassMusicalIntent:
        reasons: List[str] = []
        groove = ctx.groove
        groove_conf = groove.confidence if groove is not None else 0.0
        # Identidade = nome da seção + ocorrência (Refrão#1, Refrão#2...); a MEMÓRIA
        # é pelo nome: o 2º refrão lembra o 1º.
        identity = f"{ctx.section_key}#{ctx.section_occurrence}"
        if identity != self._section_key:
            if self._section_key is not None:
                self.controller.learn(self._last_position_confidence, self._last_groove_confidence)
            profile = self.controller.enter(ctx.section_key, ctx.section_type,
                                            ctx.section_occurrence, groove)
            self.phrases.begin_section(ctx.section_key, profile.phrase_length)
            self._section_key = identity
            if profile.from_memory:
                reasons.append(REASON_SECTION_MEMORY)
        else:
            profile, update_reasons = self.controller.update_bar(groove)
            reasons.extend(update_reasons)
        self._last_position_confidence = ctx.position_confidence
        self._last_groove_confidence = groove_conf
        prior = self.controller.prior(ctx.section_type)
        memory_energy = self.controller.memory_energy(ctx.section_key, ctx.section_type)
        audio_energy = None
        if groove is not None:
            # Um compasso isolado mais forte não muda a intenção: média móvel curta.
            self._audio_energy = (groove.energy if self._audio_energy is None else
                                  self._audio_energy + self.ENERGY_ALPHA * (groove.energy - self._audio_energy))
            audio_energy = self._audio_energy
        # Entrando na seção, o groove suavizado ainda é o da seção ANTERIOR: o prior
        # (e a memória) comandam os primeiros compassos — observar, entender, decidir.
        entry_factor = min(1.0, (profile.bars + 1) / 3.0)
        energy = self.envelope.target(prior, ctx.section_progress, audio_energy,
                                      groove_conf * entry_factor,
                                      memory_energy if profile.occurrence > 1 else None)
        prior_energy = self.envelope.prior_energy(prior, ctx.section_progress)
        if audio_energy is not None and groove_conf >= 0.6 and abs(audio_energy - prior_energy) >= 0.25:
            reasons.append(REASON_AUDIO_OVERRIDES_PRIOR)
        if prior.energy_end - prior.energy_start >= 0.15 and ctx.section_progress >= 0.3:
            reasons.append(REASON_BUILD)

        # Densidade: identidade da seção, com o arco de energia (pré-refrão cresce).
        attacks = profile.target_attacks * (0.8 + 0.6 * energy) / (0.8 + 0.6 * profile.base_energy)
        lo, hi = prior.attacks
        attacks = float(np.clip(attacks, max(0.5, lo * 0.5), hi + 2.0))
        reasons.append(REASON_LOW_ENERGY if energy < 0.35 else
                       REASON_HIGH_ENERGY if energy >= 0.65 else REASON_SECTION_CONTINUITY)

        # Harmonia longa + música calma: segura a nota, respira.
        changes_in_bar = sum(1 for i, c in enumerate(ctx.chords[1:], start=1)
                             if c and c != ctx.chords[i - 1])
        sustain = (changes_in_bar == 0 and ctx.chord_duration_beats >= self.beats_per_bar and
                   energy < 0.45)
        if sustain:
            attacks = min(attacks, 1.0)
            reasons.append(REASON_LONG_CHORD)

        previous = self.last_intent
        if (previous is not None and previous.section_key == ctx.section_key and
                previous.section_occurrence == ctx.section_occurrence):
            # Dentro da seção a densidade SOBE aos poucos (crescer é gradual);
            # acalmar pode ser imediato.
            attacks = min(attacks, previous.attacks_cap + self.MAX_ATTACK_STEP)

        confidence = float(min(ctx.position_confidence, ctx.chord_confidence))
        ceiling = 2 if energy < 0.3 else 3 if energy < 0.5 else 5
        observation = self._bars < self.OBSERVE_BARS
        if observation:
            ceiling = min(ceiling, 2)
            attacks = min(attacks, 2.0)
            reasons.append(REASON_OBSERVE)
        fallback = ""
        if confidence < 0.6:
            ceiling = 1
            attacks = min(attacks, 2.0)
            fallback = "SUSTAINED" if confidence < 0.4 else "ROOT"
            reasons.append(REASON_LOW_CONFIDENCE)
        vocabulary = min(self.creativity, ceiling)

        # Viradas: só nos pontos musicais (fim de frase/seção), nunca aleatórias.
        section_end = ctx.bars_until_section_change <= 1
        phrase_end = self.phrases.bars_left_in_phrase == 1
        policy = profile.fill_policy
        policy_ok = ((policy == "SECTION_END" and section_end) or
                     (policy == "PHRASE_END" and (phrase_end or section_end)) or
                     (policy == "RARE" and section_end and energy >= 0.4))
        fill_allowed = (policy_ok and not observation and not fallback and energy >= 0.3 and
                        vocabulary >= 2 and not ctx.final_bar)
        fill_probability = _FILL_BY_CREATIVITY[self.creativity] if fill_allowed else 0.0
        if fill_allowed:
            reasons.append(REASON_FILL_SECTION_END if section_end else REASON_FILL_PHRASE_END)

        action, mod_reason = self.modulation.plan(
            ctx.beats_until_key_change, self.beats_per_bar, energy, density_level(attacks),
            confidence, vocabulary, ctx.section_type)
        if mod_reason:
            reasons.append(mod_reason)
        if ctx.final_bar:
            reasons.append(REASON_FINAL)

        intent = BassMusicalIntent(
            bar_number=ctx.bar_number, section_type=profile.section_type,
            section_key=ctx.section_key, section_occurrence=ctx.section_occurrence,
            section_progress=ctx.section_progress, current_key=ctx.current_key,
            next_key=ctx.next_key, beats_until_key_change=ctx.beats_until_key_change,
            current_chord=next((c for c in ctx.chords if c), "--") or "--",
            next_chord=ctx.next_chord or "--",
            energy_target=energy, energy_trend=groove.energy_trend if groove else 0.0,
            audio_energy=audio_energy if audio_energy is not None else -1.0,
            density_target=density_level(attacks), attacks_cap=attacks,
            groove_label=profile.groove_family, groove_confidence=groove_conf,
            subdivision=profile.subdivision, syncopation_target=prior.syncopation,
            articulation=profile.articulation, note_length_factor=_LENGTH[profile.articulation],
            register=profile.register, fill_probability=fill_probability,
            transition_fill_allowed=fill_allowed,
            approach_note_policy="DIATONIC" if vocabulary >= 3 and energy >= 0.35 else "NONE",
            push_allowed=vocabulary >= 2 and energy >= 0.35 and not fallback,
            sustain_preferred=sustain, phrase_length_bars=profile.phrase_length,
            bars_left_in_phrase=self.phrases.bars_left_in_phrase,
            variation_budget=profile.variation_budget, complexity_ceiling=ceiling,
            vocabulary_level=vocabulary,
            velocity_base=int(round(72 + 38 * energy)), accent=int(round(6 + 8 * energy)),
            modulation_action=action, fallback_pattern=fallback, observation_mode=observation,
            confidence=confidence, reasons=list(dict.fromkeys(reasons)))
        self._bars += 1
        self.last_intent = intent
        return intent


# ------------------------------------------------------------------ diagnóstico
@dataclass
class DecisionTraceEntry:
    timestamp: float
    bar: int
    beat: int
    section: str
    key: str
    chord: str
    groove: str
    energy: float
    selected_phrase: str
    selected_note: str
    reason_code: str


@dataclass
class _SectionStats:
    bars: int = 0
    beats: int = 0
    notes: int = 0
    roots: int = 0
    fifths: int = 0
    passing: int = 0
    fills: int = 0
    phrase_changes: int = 0
    groove_changes: int = 0
    energies: List[float] = field(default_factory=list)
    velocities: List[int] = field(default_factory=list)
    durations: List[float] = field(default_factory=list)


class BassBrainDiagnostics:
    """Painel 'cérebro do baixo': dados objetivos, decisões e métricas por seção."""

    def __init__(self, trace_size: int = 400):
        self._trace: Deque[DecisionTraceEntry] = deque(maxlen=trace_size)
        self.reset()

    def reset(self) -> None:
        self._trace.clear()
        self._stats: Dict[str, _SectionStats] = {}
        self._order: List[str] = []

    def record_bar(self, intent: BassMusicalIntent, notes: Sequence[Tuple[float, str, str, int, float]],
                   phrase_id: str, decision: str, beats_per_bar: int, fill: bool = False,
                   phrase_changed: bool = False, groove_changed: bool = False) -> None:
        """``notes``: (tempo, nota, papel, velocidade, duração em batidas)."""
        key = (f"{intent.section_key}#{intent.section_occurrence}" if intent.section_key
               else intent.section_type)
        if key not in self._stats:
            self._stats[key] = _SectionStats()
            self._order.append(key)
        stats = self._stats[key]
        stats.bars += 1
        stats.beats += beats_per_bar
        stats.fills += int(fill)
        stats.phrase_changes += int(phrase_changed)
        stats.groove_changes += int(groove_changed)
        stats.energies.append(intent.energy_target)
        for time, note, role, velocity, length in notes:
            stats.notes += 1
            stats.roots += role in ("R", "R8", "PUSH")
            stats.fifths += role == "5"
            stats.passing += role in ("APP",) or role.startswith(("U", "S", "C"))
            stats.velocities.append(int(velocity))
            stats.durations.append(float(length))
            self._trace.append(DecisionTraceEntry(
                timestamp=round(float(time), 3), bar=intent.bar_number, beat=0,
                section=key, key=intent.current_key, chord=intent.current_chord,
                groove=intent.groove_label, energy=round(intent.energy_target, 3),
                selected_phrase=phrase_id, selected_note=note,
                reason_code=intent.reasons[0] if intent.reasons else decision))

    @property
    def trace(self) -> List[DecisionTraceEntry]:
        return list(self._trace)

    def section_metrics(self) -> List[Dict]:
        rows = []
        for key in self._order:
            s = self._stats[key]
            notes = max(1, s.notes)
            rows.append({
                "section": key, "bars": s.bars,
                "notes_per_beat": round(s.notes / max(1, s.beats), 3),
                "notes_per_bar": round(s.notes / max(1, s.bars), 2),
                "root_ratio": round(s.roots / notes, 2), "fifth_ratio": round(s.fifths / notes, 2),
                "passing_ratio": round(s.passing / notes, 2), "fills": s.fills,
                "phrase_changes": s.phrase_changes, "groove_changes": s.groove_changes,
                "energy_variance": round(float(np.var(s.energies)) if s.energies else 0.0, 4),
                "velocity_variance": round(float(np.var(s.velocities)) if s.velocities else 0.0, 1),
                "avg_note_beats": round(float(np.mean(s.durations)) if s.durations else 0.0, 2),
            })
        return rows


def format_brain_report(intent: Optional[BassMusicalIntent], phrase: str = "--",
                        decision: str = "--", phrase_lock: int = 0) -> str:
    """Texto do painel BASS BRAIN (só dados e regras, sem raciocínio privado)."""
    if intent is None:
        return "BASS BRAIN\naguardando a música"
    key_change = (f"{intent.beats_until_key_change:.0f} tempos"
                  if intent.beats_until_key_change >= 0 else "—")
    lines = [
        "BASS BRAIN",
        f"Seção: {intent.section_key or intent.section_type}  ({int(intent.section_progress * 100)}%)",
        f"Tom: {intent.current_key}   Próximo tom: {intent.next_key}   Modulação em: {key_change}",
        f"Groove: {intent.groove_label}  (confiança {int(intent.groove_confidence * 100)}%)",
        f"Energia áudio: {intent.audio_energy:.2f}   Energia alvo: {intent.energy_target:.2f}"
        if intent.audio_energy >= 0 else f"Energia alvo: {intent.energy_target:.2f}",
        f"Densidade: {intent.density_target}  (≤ {intent.attacks_cap:.1f} ataques/compasso)",
        f"Frase: {phrase}   Trava da frase: {phrase_lock} compasso(s)",
        f"Próximo acorde: {intent.next_chord}   Vocabulário: nível {intent.vocabulary_level}",
        f"Decisão: {decision}",
        "Motivos: " + ", ".join(r.replace("REASON_", "") for r in intent.reasons[:4]),
    ]
    return "\n".join(lines)
