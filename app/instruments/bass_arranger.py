"""Arranjador do baixista: transforma cifra alinhada + estilo em uma linha de baixo.

Entrada: ``HarmonyTimeline`` (acorde da cifra por batida do áudio, tempos 1,
intensidade e acentos da levada) e um ``BassStyle``. Saída: ``BassArrangement``,
a lista de notas com instante exato no arquivo.

Como um baixista que conhece a música inteira, o arranjador:
- divide a música em frases (4 compassos, reiniciando em cada seção nova);
- mede a intensidade de cada frase (energia do áudio + tipo de seção) e escolhe
  uma levada do estilo daquele nível, preferindo a que encaixa nos acentos que
  o violão/instrumento está tocando, e respeitando o andamento;
- mantém a levada dentro da frase e varia nos pontos musicais: viradas no fim
  de frase/seção, aproximações para o próximo acorde, antecipações quando a
  banda antecipa, pequenas ornamentações;
- conduz as notas (registro próximo, passagens pela escala do tom);
- é determinístico por ``seed``: a mesma variação toca sempre igual.

A HARMONIA continua sendo a da cifra: toda nota de apoio (R) é o baixo do
acorde da cifra naquele instante (inclusive inversões como C/E).
"""

from dataclasses import dataclass, field
import math
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

from app.analysis.chart_harmony_aligner import HarmonyTimeline, chord_template
from app.instruments.bass_bank import load_bank
from app.analysis.groove_feel import GrooveFeelAnalyzer
from app.instruments.bass_brain import BarRequest, BassBrain
from app.instruments.bass_intent import (APPROACH, DIRECT_CHANGE, PREPARE, BarContext,
                                         BassBrainDiagnostics, BassMusicalIntent,
                                         BassMusicalIntentEngine)
from app.instruments.bass_styles import BassStyle, Fill, Groove, Hit
from app.music.chord_chart import parse_chord
from app.music.constants import BASS_MAX_MIDI_NOTE, BASS_MIN_MIDI_NOTE
from app.music.theory import PITCH_CLASSES, note_to_pc
from app.music.transposition import parse_key_root_mode

_ROOT_LOW, _ROOT_HIGH = BASS_MIN_MIDI_NOTE, 43      # E1..G2: registro das notas de apoio
_MAJOR = (0, 2, 4, 5, 7, 9, 11)
_MINOR = (0, 2, 3, 5, 7, 8, 10)
_SLOTS_PER_BEAT = 4


@dataclass
class ArrangedNote:
    time: float
    midi: int
    velocity: int
    duration: float
    beat_index: int
    bar: int
    beat_in_bar: int
    slot: int                 # semicolcheia dentro do compasso
    chord: str
    role: str
    reason: str

    @property
    def note(self) -> str:
        return f"{PITCH_CLASSES[self.midi % 12]}{self.midi // 12 - 1}"


@dataclass
class BarPlan:
    bar: int
    start_beat: int
    start_time: float
    tier: int
    intensity: float
    groove: str
    fill: str = ""
    variation: str = ""
    source: str = ""          # música do banco que inspirou o compasso (linha real)
    intent: Optional[dict] = None     # intenção musical do compasso (painel do cérebro)
    decision: str = ""
    phrase: str = ""


@dataclass
class BassArrangement:
    notes: List[ArrangedNote]
    bars: List[BarPlan]
    style_label: str
    creativity: int
    seed: int
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    key_label: str = ""            # tom usado para as notas de passagem
    key_source: str = ""           # "cifra" ou "acordes" (quando o tom da cifra não bate)
    diagnostics: Optional[BassBrainDiagnostics] = None

    def __post_init__(self) -> None:
        self.notes.sort(key=lambda note: note.time)
        self.times = np.array([note.time for note in self.notes], dtype=np.float64)

    def first_index_at(self, time: float) -> int:
        return int(np.searchsorted(self.times, time, side="left"))

    def bar_at(self, time: float) -> Optional[BarPlan]:
        current = None
        for plan in self.bars:
            if plan.start_time <= time + 1e-6:
                current = plan
            else:
                break
        return current


@dataclass
class _Bar:
    index: int
    start: int                       # índice da batida do tempo 1 (pode ser < 0 na anacruse)
    active: bool
    intensity: float = 0.5
    section_start: bool = False
    hits: List[Hit] = field(default_factory=list)
    plan: Optional[BarPlan] = None
    band: Optional[np.ndarray] = None      # força dos ataques da banda por semicolcheia (0..~1)
    intent: Optional[BassMusicalIntent] = None
    scale: Optional[Tuple[int, ...]] = None       # escala do tom vigente no compasso
    next_scale: Optional[Tuple[int, ...]] = None  # escala do tom do compasso seguinte
    approach_to_new_key: bool = False             # aproxima o 1º acorde do novo tom


class BassArranger:
    """Gera a linha de baixo de uma música inteira para um estilo."""

    def __init__(self, style: BassStyle):
        self._style = style

    # ------------------------------------------------------------------ API
    def arrange(self, timeline: HarmonyTimeline, creativity: int = 3,
                seed: int = 0) -> BassArrangement:
        """Linha de baixo da música. Com o banco de levadas reais (4/4), decide
        compasso a compasso só com o passado + cifra — como fará ao vivo."""
        bank = load_bank(self._style.name)
        if bank is not None and max(2, timeline.beats_per_bar) == 4:
            return self._arrange_live_simulation(bank, timeline, creativity, seed)
        return self._arrange_with_templates(timeline, creativity, seed)

    def _arrange_live_simulation(self, bank, timeline: HarmonyTimeline, creativity: int,
                                 seed: int) -> BassArrangement:
        creativity = max(1, min(5, int(creativity)))
        self._timeline = timeline
        self._beats = timeline.beat_count
        self._bpb = 4
        self._slots = 16
        self._bass_pc = [self._bass_pc_of(chord) for chord in timeline.chords]
        scale, key_label, key_source = self._choose_scale(timeline)
        self._scale = scale
        engine = BassMusicalIntentEngine(self._style.name, creativity, beats_per_bar=4)
        groove = GrooveFeelAnalyzer(4, calibrated_energy=True)
        brain = BassBrain(bank, creativity, seed, scale, phrase_memory=engine.phrases)
        bars = self._build_bars()
        self._prepare_sections()
        sections = timeline.event_sections
        events = timeline.event_indices
        band_history: List[np.ndarray] = []
        energy_history: List[float] = []
        notes: List[ArrangedNote] = []
        plans: List[BarPlan] = []
        observed_changes = 0
        bpm = self._estimate_bpm(timeline)
        for position, bar in enumerate(bars):
            beats = [bar.start + b for b in range(5)]
            # Aprende se a banda antecipa: só trocas já ouvidas (até o tempo 1 anterior).
            horizon = bar.start - 4
            for k in range(max(1, observed_changes), max(1, horizon + 1)):
                if 0 < k < self._beats and self._bass_pc[k] not in (-1, self._bass_pc[k - 1]):
                    evidence = self._band_anticipates(k - 1)
                    if evidence is not None:
                        brain.observe_anticipation(evidence)
            observed_changes = max(observed_changes, horizon + 1)
            if bar.active:
                next_bar = bars[position + 1] if position + 1 < len(bars) else None
                section_type = ""
                first = next((k for k in range(bar.start, bar.start + 4)
                              if 0 <= k < len(events) and events[k] >= 0), None)
                if first is not None and events[first] < len(sections):
                    section_type = sections[events[first]]
                final = next_bar is None or not next_bar.active
                context = self._bar_context(bar, groove.profile if groove.profile.bars_observed else None,
                                            final)
                intent = engine.plan_bar(context)
                request = BarRequest(
                    bar_number=bar.index + 1,
                    beat_times=[self._boundary_time(k) for k in beats],
                    chords=[self._chord(k) for k in beats[:4]],
                    next_chord=self._chord(bar.start + 4),
                    previous_chord=self._chord(bar.start - 1),
                    band_history=list(band_history[-2:]),
                    energy_history=list(energy_history[-9:]),
                    section_start=bar.section_start, section_type=section_type,
                    final_bar=final, intent=intent,
                    scale=self._scale_at(bar.start), next_scale=self._scale_at(bar.start + 4))
                decision = brain.plan_bar(request)
                plans.append(BarPlan(bar.index + 1, bar.start, self._boundary_time(bar.start),
                                     decision.level, round(decision.intensity, 2),
                                     decision.label, source=decision.source,
                                     intent=intent.to_dict(), decision=decision.decision,
                                     phrase=decision.phrase_id))
                period = (self._boundary_time(bar.start + 4) - self._boundary_time(bar.start)) / 4.0
                engine.diagnostics.record_bar(
                    intent, [(n.time, n.note, n.role, n.velocity, n.duration / max(1e-6, period))
                             for n in decision.notes],
                    decision.phrase_id, decision.decision, 4, fill=decision.fill,
                    phrase_changed=decision.phrase_changed,
                    groove_changed="REASON_GROOVE_TRANSITION" in intent.reasons)
                for planned in decision.notes:
                    k = bar.start + planned.slot // 4
                    notes.append(ArrangedNote(
                        time=planned.time, midi=planned.midi, velocity=planned.velocity,
                        duration=planned.duration, beat_index=k, bar=bar.index + 1,
                        beat_in_bar=planned.slot // 4 + 1, slot=planned.slot,
                        chord=planned.chord, role=planned.role, reason=planned.reason))
            # O compasso terminou: agora a banda dele faz parte do passado.
            band_history.append(self._bar_band_grid(bar))
            energy_history.append(self._bar_energy(bar))
            if bar.active:
                self._observe_groove(groove, bar, bpm)
        return BassArrangement(notes=notes, bars=plans, style_label=self._style.label,
                               creativity=creativity, seed=seed, key_label=key_label,
                               key_source=key_source, diagnostics=engine.diagnostics)

    # ------------------------------------------------------------ intenção / seções
    def _prepare_sections(self) -> None:
        """Para cada batida: rótulo da seção (nome#ocorrência) e o trecho que ela ocupa."""
        timeline = self._timeline
        labels = list(timeline.event_section_labels)
        if not labels:
            # Sem rótulos da cifra: usa tipo + contagem de inícios de seção.
            starts = set(timeline.section_start_events)
            counts: dict = {}
            current = ""
            for e, kind in enumerate(timeline.event_sections):
                if e in starts or not current:
                    counts[kind] = counts.get(kind, 0) + 1
                    current = f"{kind}#{counts[kind]}"
                labels.append(current)
        self._event_labels = labels
        self._beat_label: List[str] = []
        for k in range(self._beats):
            e = timeline.event_indices[k] if k < len(timeline.event_indices) else -1
            self._beat_label.append(labels[e] if 0 <= e < len(labels) else "")
        self._label_span: dict = {}
        for k, label in enumerate(self._beat_label):
            if label:
                first, last = self._label_span.get(label, (k, k))
                self._label_span[label] = (min(first, k), max(last, k))

    def _section_at(self, k: int) -> Tuple[str, int, str, float, int]:
        """(nome, ocorrência, tipo, progresso 0..1, compassos até a próxima seção)."""
        label = ""
        for j in range(k, min(self._beats, k + self._bpb)):
            if 0 <= j < len(self._beat_label) and self._beat_label[j]:
                label = self._beat_label[j]
                break
        if not label:
            return "", 1, "UNKNOWN", 0.0, 99
        name, _, occurrence = label.rpartition("#")
        first, last = self._label_span[label]
        span = max(1, last - first + 1)
        progress = float(np.clip((k - first) / span, 0.0, 1.0))
        bars_left = int(math.ceil(max(0, last + 1 - k) / self._bpb))
        events = self._timeline.event_indices
        e = next((events[j] for j in range(k, min(self._beats, k + self._bpb)) if events[j] >= 0), -1)
        kind = (self._timeline.event_sections[e] if 0 <= e < len(self._timeline.event_sections)
                else "UNKNOWN")
        return name, int(occurrence or 1), kind, progress, bars_left

    def _scale_at(self, k: int) -> Optional[Tuple[int, ...]]:
        """Escala do tom vigente na batida (modulações da cifra); senão, a da música."""
        timeline = self._timeline
        if timeline.event_keys and len(set(timeline.event_keys)) > 1 and 0 <= k < self._beats:
            scale = self._scale_pcs(timeline.key_at_beat(k))
            if scale is not None:
                return scale
        return getattr(self, "_scale", None)

    def _beats_until_key_change(self, k: int) -> Tuple[float, str]:
        timeline = self._timeline
        if not timeline.event_keys or len(set(timeline.event_keys)) <= 1 or not 0 <= k < self._beats:
            return -1.0, "--"
        here = timeline.key_at_beat(k)
        for j in range(k + 1, min(self._beats, k + 8 * self._bpb)):
            if 0 <= timeline.event_indices[j] and timeline.key_at_beat(j) != here:
                return float(j - k), timeline.key_at_beat(j)
        return -1.0, "--"

    def _chord_duration_from(self, k: int) -> float:
        chord = self._chord(k)
        length = 0
        while k + length < self._beats and self._chord(k + length) == chord and length < 4 * self._bpb:
            length += 1
        return float(length)

    def _bar_context(self, bar: "_Bar", groove_profile, final: bool) -> BarContext:
        name, occurrence, kind, progress, bars_left = self._section_at(bar.start)
        beats_until, next_key = self._beats_until_key_change(bar.start)
        timeline = self._timeline
        first = next((j for j in range(bar.start, bar.start + self._bpb) if self._chord(j)), bar.start)
        return BarContext(
            bar_number=bar.index + 1, beats_per_bar=self._bpb,
            section_key=name or kind, section_type=kind, section_occurrence=occurrence,
            section_progress=progress, bars_until_section_change=bars_left,
            chords=[self._chord(bar.start + b) for b in range(self._bpb)],
            next_chord=self._chord(bar.start + self._bpb),
            chord_duration_beats=self._chord_duration_from(first),
            current_key=timeline.key_at_beat(first) if 0 <= first < self._beats else timeline.key,
            next_key=next_key, beats_until_key_change=beats_until, groove=groove_profile,
            # Alinhamento offline aceito (>= 0.55) é posição confiável.
            position_confidence=float(np.clip(0.4 + timeline.confidence, 0.0, 1.0)),
            chord_confidence=1.0, final_bar=final, compound=self._bpb == 6)

    def _observe_groove(self, groove: GrooveFeelAnalyzer, bar: "_Bar", bpm: float) -> None:
        """O compasso acabou de soar: alimenta a escuta do groove (só o passado)."""
        timeline = self._timeline

        def grid_of(values):
            if values is None:
                return None
            grid = np.zeros((self._bpb, 4))
            for b in range(self._bpb):
                k = bar.start + b
                if 0 <= k < len(values) and self._chord(k):
                    grid[b] = values[k]
            return grid

        groove.observe_bar(grid_of(timeline.onsets) if timeline.onsets is not None
                           else np.zeros((self._bpb, 4)),
                           energy=self._bar_energy(bar), bpm=bpm,
                           high_grid=grid_of(timeline.high_onsets),
                           low_grid=grid_of(timeline.low_onsets))

    def _boundary_time(self, k: int) -> float:
        times = self._timeline.beat_times
        period = float(np.median(np.diff(times))) if len(times) > 1 else 0.5
        if k < 0:
            return float(times[0]) + k * period
        if k >= len(times):
            return float(times[-1]) + (k - len(times) + 1) * period
        return float(times[k])

    def _bar_band_grid(self, bar: "_Bar") -> np.ndarray:
        grid = np.zeros(16)
        onsets = self._timeline.onsets
        if onsets is None:
            return grid
        for b in range(4):
            k = bar.start + b
            if 0 <= k < len(onsets) and self._chord(k):
                grid[b * 4:(b + 1) * 4] = onsets[k]
        return grid

    def _bar_energy(self, bar: "_Bar") -> float:
        energy = self._timeline.energy
        if energy is None:
            return 0.0
        values = [energy[k] for k in range(bar.start, bar.start + self._bpb)
                  if 0 <= k < len(energy) and self._chord(k)]
        return float(np.mean(values)) if values else 0.0

    def _arrange_with_templates(self, timeline: HarmonyTimeline, creativity: int = 3,
                                seed: int = 0) -> BassArrangement:
        creativity = max(1, min(5, int(creativity)))
        rng = random.Random(seed * 1_000_003 + creativity * 97 + 11)
        self._timeline = timeline
        self._beats = timeline.beat_count
        self._bpb = max(2, timeline.beats_per_bar)
        self._slots = self._bpb * _SLOTS_PER_BEAT
        self._bpm = self._estimate_bpm(timeline)
        self._bass_pc = [self._bass_pc_of(chord) for chord in timeline.chords]
        self._scale, key_label, key_source = self._choose_scale(timeline)
        grooves = self._grooves_for_meter()
        bars = self._build_bars()
        self._assign_intensity(bars)
        diagnostics = self._plan_intents(bars, creativity)
        phrases = self._build_phrases(bars)
        previous_groove: Optional[Groove] = None
        previous_profile: Optional[np.ndarray] = None
        for phrase_index, phrase in enumerate(phrases):
            profile = self._phrase_profile(phrase)
            self._mark_band(phrase, profile)
            tier = self._phrase_tier(phrase, profile, creativity)
            # O teto vem da MÚSICA (intenção da seção), não da criatividade.
            intent_tiers = [self._intent_tier(bar.intent) for bar in phrase if bar.intent is not None]
            if intent_tiers:
                tier = min(tier, max(intent_tiers))
            same_section = phrase_index > 0 and not phrase[0].section_start
            band_unchanged = (previous_profile is not None and profile is not None and
                              self._similar(previous_profile, profile) >= 0.9)
            keep = (same_section and band_unchanged and previous_groove is not None and
                    previous_groove.tier <= tier + 1 and
                    rng.random() < self._style.keep_groove_prob[creativity])
            groove = previous_groove if keep else self._choose_groove(
                grooves, tier, phrase, profile, previous_groove, creativity, rng, same_section)
            previous_groove, previous_profile = groove, profile
            for position, bar in enumerate(phrase):
                bar.hits = list(groove.hits)
                bar.plan = BarPlan(bar.index + 1, bar.start, self._beat_time(max(0, bar.start)),
                                   groove.tier, round(bar.intensity, 2), groove.label)
                last_in_phrase = position == len(phrase) - 1
                next_bar = self._bar_after(bars, bar)
                intent = bar.intent
                if intent is not None:
                    bar.plan.intent = intent.to_dict()
                if next_bar is None or not next_bar.active:
                    self._make_ending(bar)
                elif intent is not None and intent.transition_fill_allowed:
                    # Virada só nos pontos de transição que a intenção liberou.
                    self._maybe_fill(bar, groove.tier, next_bar.section_start, creativity, rng)
                elif intent is None and last_in_phrase:
                    self._maybe_fill(bar, groove.tier, next_bar.section_start, creativity, rng)
                elif intent is None or intent.vocabulary_level >= 3:
                    self._maybe_ornament(bar, creativity, rng)
                if intent is None or intent.approach_note_policy != "NONE":
                    self._plan_approaches(bar, groove.tier, creativity, rng)
                self._lock_to_band(bar, creativity, rng)
                if intent is not None:
                    self._apply_intent(bar, intent)
        notes = self._resolve(bars, creativity, rng)
        plans = [bar.plan for bar in bars if bar.plan is not None]
        self._record_diagnostics(diagnostics, bars, notes)
        return BassArrangement(notes=notes, bars=plans, style_label=self._style.label,
                               creativity=creativity, seed=seed, key_label=key_label,
                               key_source=key_source, diagnostics=diagnostics)

    # ------------------------------------------------------------ análise
    @staticmethod
    def _estimate_bpm(timeline: HarmonyTimeline) -> float:
        diffs = np.diff(timeline.beat_times)
        period = float(np.median(diffs)) if len(diffs) else 0.5
        return 60.0 / max(1e-3, period)

    @staticmethod
    def _bass_pc_of(chord: Optional[str]) -> int:
        if not chord:
            return -1
        symbol = parse_chord(chord)
        return note_to_pc(symbol.bass_note or symbol.root)

    @staticmethod
    def _scale_pcs(key: str) -> Optional[Tuple[int, ...]]:
        if not key:
            return None
        root, mode = parse_key_root_mode(key)
        tonic = note_to_pc(root)
        if tonic < 0:
            return None
        return tuple((tonic + step) % 12 for step in (_MINOR if mode == "minor" else _MAJOR))

    def _choose_scale(self, timeline: HarmonyTimeline) -> Tuple[Optional[Tuple[int, ...]], str, str]:
        """Escala das notas de passagem: a do tom da cifra, conferida contra os acordes.

        Se o tom informado não explica os acordes (cifra sem "Tom:", tom trocado),
        usa a escala diatônica que melhor explica os acordes da música.
        """
        weights = np.zeros(12)
        for chord in timeline.chords:
            if chord:
                template = chord_template(chord)
                if template is not None:
                    weights += (template > 0).astype(float)
        if weights.sum() == 0:
            scale = self._scale_pcs(timeline.key)
            return scale, timeline.key, "cifra"

        def fit(pcs: Tuple[int, ...]) -> float:
            return float(sum(weights[pc] for pc in pcs) / weights.sum())

        best_tonic = max(range(12), key=lambda tonic: fit(tuple((tonic + s) % 12 for s in _MAJOR)))
        best_scale = tuple((best_tonic + s) % 12 for s in _MAJOR)
        chart_scale = self._scale_pcs(timeline.key)
        if chart_scale is not None and fit(chart_scale) >= fit(best_scale) - 0.03:
            return chart_scale, timeline.key, "cifra"
        return best_scale, f"{PITCH_CLASSES[best_tonic]} Major", "acordes"

    def _beat_time(self, index: int, slot: int = 0) -> float:
        index = max(0, min(self._beats - 1, index))
        start = float(self._timeline.beat_times[index])
        period = float(self._timeline.beat_times[index + 1]) - start
        return start + period * slot / _SLOTS_PER_BEAT

    def _beat_period(self, index: int) -> float:
        index = max(0, min(self._beats - 1, index))
        return float(self._timeline.beat_times[index + 1] - self._timeline.beat_times[index])

    def _chord(self, index: int) -> Optional[str]:
        return self._timeline.chords[index] if 0 <= index < self._beats else None

    def _grooves_for_meter(self) -> Tuple[Groove, ...]:
        if self._bpb == 4:
            return self._style.grooves
        last = self._slots - 1
        each_beat = tuple((b * 4, "R", 100 if b == 0 else 86, 3) for b in range(self._bpb))
        eighths = tuple((s, "R", 102 if s == 0 else 84, 2) for s in range(0, self._slots, 2))
        return (Groove("longa", "Nota longa", 0, ((0, "R", 100, last),)),
                Groove("tempos", "Pulso nos tempos", 1, each_beat),
                Groove("colcheias", "Colcheias", 2, eighths, max_bpm=168),
                Groove("colcheias_fortes", "Colcheias fortes", 3,
                       tuple((s, "R", 106 if s % 4 == 0 else 88, 2) for s in range(0, self._slots, 2)),
                       max_bpm=168))

    def _build_bars(self) -> List[_Bar]:
        phase = self._timeline.downbeat_phase % self._bpb
        first = -1 if phase > 0 else 0
        bars = []
        index = first
        while phase + index * self._bpb < self._beats:
            start = phase + index * self._bpb
            active = any(self._chord(start + b) for b in range(self._bpb))
            bars.append(_Bar(index=index, start=start, active=active))
            index += 1
        starts = set(self._timeline.section_start_events)
        events = self._timeline.event_indices
        previous_event = -1
        for bar in bars:
            for k in range(bar.start, bar.start + self._bpb):
                if 0 <= k < len(events) and events[k] >= 0:
                    if events[k] != previous_event and events[k] in starts:
                        bar.section_start = True
                    previous_event = events[k]
        return bars

    def _assign_intensity(self, bars: List[_Bar]) -> None:
        energy = self._timeline.energy
        active = [bar for bar in bars if bar.active]
        if energy is None or not active:
            return
        values = []
        for bar in active:
            ks = [k for k in range(bar.start, bar.start + self._bpb)
                  if self._chord(k) and 0 <= k < len(energy)]
            values.append(float(np.mean(energy[ks])) if ks else 0.0)
        values = np.asarray(values)
        positive = values[values > 0]
        if len(positive) < 2:
            return
        p10, p90 = np.percentile(positive, 10), np.percentile(positive, 90)
        # Música com dinâmica pequena não deve ganhar contrastes artificiais.
        spread = float(np.clip(math.log2(max(p90, 1e-9) / max(p10, 1e-9)) / 1.5, 0.0, 1.0))
        ranks = np.argsort(np.argsort(values)) / max(1, len(values) - 1)
        sections = self._timeline.event_sections
        events = self._timeline.event_indices
        for bar, rank in zip(active, ranks):
            intensity = 0.55 + (float(rank) - 0.5) * spread
            first = next((k for k in range(bar.start, bar.start + self._bpb)
                          if 0 <= k < len(events) and events[k] >= 0), None)
            if first is not None and events[first] < len(sections):
                section = sections[events[first]]
                intensity += next((bias for name, bias in self._style.section_bias.items()
                                   if name in section), 0.0)
            bar.intensity = float(np.clip(intensity, 0.0, 1.0))

    def _build_phrases(self, bars: List[_Bar]) -> List[List[_Bar]]:
        phrases: List[List[_Bar]] = []
        current: List[_Bar] = []
        previous_active = False
        for bar in bars:
            if not bar.active:
                if current:
                    phrases.append(current)
                current, previous_active = [], False
                continue
            if current and (bar.section_start or not previous_active or
                            len(current) >= self._style.phrase_bars):
                phrases.append(current)
                current = []
            current.append(bar)
            previous_active = True
        if current:
            phrases.append(current)
        return phrases

    @staticmethod
    def _bar_after(bars: List[_Bar], bar: _Bar) -> Optional[_Bar]:
        position = bar.index - bars[0].index + 1
        return bars[position] if position < len(bars) else None

    # ------------------------------------------------------------ escolhas
    def _phrase_tier(self, phrase: List[_Bar], profile: Optional[np.ndarray], creativity: int) -> int:
        """Nível da levada: energia da música E quão cheia a banda está tocando."""
        intensity = float(np.mean([bar.intensity for bar in phrase]))
        energy_tier = 0 if intensity < 0.30 else 1 if intensity < 0.60 else 2 if intensity < 0.85 else 3
        tier = energy_tier
        if profile is not None and profile.max() > 0:
            # ataques da banda por compasso (normalizado para 4 tempos)
            attacks = int(np.sum(profile / profile.max() >= 0.35)) * 16 / self._slots
            density_tier = 0 if attacks <= 2 else 1 if attacks <= 4 else 2 if attacks <= 7 else 3
            # O baixo não fica mais agitado que a banda (nem muito mais calmo).
            tier = int(math.floor((energy_tier + density_tier) / 2 + 0.5))
            tier = min(tier, density_tier + 1)
        # Densidade é da MÚSICA (energia + levada da banda + intenção da seção);
        # a criatividade não libera levadas mais cheias.
        return tier

    @staticmethod
    def _similar(a: np.ndarray, b: np.ndarray) -> float:
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(a @ b) / norm if norm > 0 else 0.0

    def _mark_band(self, phrase: List[_Bar], profile: Optional[np.ndarray]) -> None:
        """Guarda em cada compasso a força dos ataques da banda (relativa à frase)."""
        onsets = self._timeline.onsets
        reference = float(profile.max()) if profile is not None and profile.max() > 0 else 0.0
        for bar in phrase:
            if onsets is None or reference <= 0:
                bar.band = None
                continue
            grid = np.zeros(self._slots)
            for b in range(self._bpb):
                k = bar.start + b
                if 0 <= k < len(onsets):
                    grid[b * 4:(b + 1) * 4] = onsets[k] / reference
            # Média com o padrão da frase: um ataque isolado fraco não some, um
            # ruído isolado não vira ataque.
            bar.band = 0.5 * grid + 0.5 * (profile / reference)

    def _band_strength(self, bar: _Bar, pos: int) -> float:
        if bar.band is None or not (0 <= pos < len(bar.band)):
            return 1.0
        return float(bar.band[pos])

    def _groove_from_band(self, profile: np.ndarray, tier: int) -> Optional[Groove]:
        """Levada do baixo tirada dos acentos da própria banda (como o bumbo no violão)."""
        if profile is None or profile.max() <= 0 or self._bpb != 4:
            return None
        norm = profile / profile.max()
        limits = {0: (1, 2), 1: (2, 4), 2: (4, 8), 3: (6, 10)}[tier]
        order = [s for s in np.argsort(-norm) if norm[s] >= 0.45 and s != 0]
        slots = [0] + [int(s) for s in order[:limits[1] - 1]]
        if len(slots) < limits[0]:
            return None
        slots.sort()
        hits = []
        for i, slot in enumerate(slots):
            following = slots[i + 1] if i + 1 < len(slots) else self._slots
            velocity = int(round(80 + 28 * norm[slot]))
            hits.append((slot, "R", velocity, max(1, following - slot)))
        return Groove("banda", "Levada travada na banda", tier, tuple(hits), weight=1.0)

    def _phrase_profile(self, phrase: List[_Bar]) -> Optional[np.ndarray]:
        onsets = self._timeline.onsets
        if onsets is None:
            return None
        profile = np.zeros(self._slots)
        count = 0
        for bar in phrase:
            for b in range(self._bpb):
                k = bar.start + b
                if 0 <= k < len(onsets) and self._chord(k):
                    profile[b * 4:(b + 1) * 4] += onsets[k]
                    count += 1
        return profile / count if count else None

    def _choose_groove(self, grooves: Sequence[Groove], tier: int, phrase: List[_Bar],
                       profile: Optional[np.ndarray], previous: Optional[Groove], creativity: int,
                       rng: random.Random, same_section: bool = False) -> Groove:
        candidates: List[Groove] = []
        for level in range(tier, -1, -1):
            candidates = [g for g in grooves if g.tier == level and
                          g.min_bpm <= self._bpm <= g.max_bpm]
            if candidates:
                break
        if not candidates:
            candidates = [min(grooves, key=lambda g: g.tier)]
        from_band = self._groove_from_band(profile, tier) if profile is not None else None
        if from_band is not None:
            # Copiar a batida da banda é seguro mas sem personalidade: quanto mais
            # criativo, mais o baixista prefere a linguagem do estilo (desde que caiba).
            from_band = Groove(from_band.name, from_band.label, from_band.tier, from_band.hits,
                               weight=(0, 1.6, 1.3, 0.8, 0.45, 0.3)[creativity])
            candidates = candidates + [from_band]
        weights = []
        for groove in candidates:
            weight = groove.weight
            if profile is not None and profile.max() > 0:
                norm = profile / profile.max()
                vector = np.zeros(self._slots)
                for pos, _, velocity, _ in groove.hits:
                    if pos < self._slots:
                        vector[pos] = velocity / 127.0
                match = self._similar(vector, norm)
                # Atacar onde a banda está em silêncio é o que faz o baixo parecer
                # tocar "outra música": penaliza forte. Tocar só PARTE dos ataques da
                # banda (os principais) é natural para o baixo: penaliza pouco.
                silent = sum(v for pos, v in enumerate(vector) if v > 0 and norm[pos] < 0.3)
                off = silent / max(1e-9, float(vector.sum()))
                if off > 0.35 and groove.name != "banda":
                    weight *= 1e-4          # metade da levada cairia no silêncio da banda
                weight *= math.exp(2.0 * match - 14.0 * off)
            if previous is not None and groove.name == previous.name:
                # Dentro da mesma seção o baixista mantém a levada e varia com
                # viradas/ornamentos; numa seção nova, a levada pode mudar.
                if same_section:
                    weight *= (0.0, 10.0, 8.0, 6.0, 2.5, 1.0)[creativity]
                else:
                    weight *= 1.5 if creativity <= 2 else 1.0 if creativity <= 4 else 0.6
            weights.append(weight)
        return rng.choices(candidates, weights=weights, k=1)[0]

    # ------------------------------------------------------------ intenção (modelos)
    def _plan_intents(self, bars: List[_Bar], creativity: int) -> BassBrainDiagnostics:
        """Intenção de cada compasso, em ordem e só com o passado do áudio."""
        engine = BassMusicalIntentEngine(self._style.name, creativity, beats_per_bar=self._bpb)
        groove = GrooveFeelAnalyzer(self._bpb, calibrated_energy=True)
        self._prepare_sections()
        bpm = self._estimate_bpm(self._timeline)
        active = [bar for bar in bars if bar.active]
        for position, bar in enumerate(active):
            final = position == len(active) - 1
            context = self._bar_context(bar, groove.profile if groove.profile.bars_observed else None,
                                        final)
            bar.intent = engine.plan_bar(context)
            bar.scale = self._scale_at(bar.start)
            bar.next_scale = self._scale_at(bar.start + self._bpb)
            self._observe_groove(groove, bar, bpm)
        self._intent_engine = engine
        return engine.diagnostics

    @staticmethod
    def _intent_tier(intent: BassMusicalIntent) -> int:
        cap = intent.attacks_cap
        return 0 if cap <= 1.6 else 1 if cap <= 3.2 else 2 if cap <= 6.0 else 3

    def _apply_intent(self, bar: _Bar, intent: BassMusicalIntent) -> None:
        """Densidade, sustentação e modulação pedidas pela intenção do compasso."""
        hits = sorted(bar.hits)
        change_slots = {b * 4 for b in range(1, self._bpb)
                        if self._chord(bar.start + b) and
                        self._bass_pc[bar.start + b] != self._bass_pc[bar.start + b - 1]}
        essential = {0} | change_slots
        if intent.sustain_preferred:
            hits = [h for h in hits if h[0] in essential] or [(0, "R", 96, self._slots)]
        cap = max(len(essential), int(round(intent.attacks_cap)))
        if len(hits) > cap:
            def priority(hit):
                pos = hit[0]
                metric = 100.0 if pos == 0 else 60.0 if pos % (self._slots // 2) == 0 else (
                    40.0 if pos % 4 == 0 else 20.0)
                return (pos in essential, metric + 20.0 * self._band_strength(bar, pos))
            keep = {h[0] for h in sorted(hits, key=priority, reverse=True)[:cap]}
            hits = [h for h in hits if h[0] in keep]
        if intent.modulation_action == APPROACH:
            last = self._slots - 2
            hits = [h for h in hits if h[0] < last] + [(last, "S-1", 94, 2)]
            bar.approach_to_new_key = True
        elif intent.modulation_action in (DIRECT_CHANGE, PREPARE):
            hits = [(p, "R" if (r == "PUSH" or r == "APP" or r[0] in "SC") and p >= self._slots - 4
                     else r, v, d) for p, r, v, d in hits]
            if intent.modulation_action == PREPARE:
                hits = [h for h in hits if h[0] <= self._slots // 2]
        hits = self._apply_vocabulary_level(sorted(hits), intent, bar, change_slots)
        # Cada nota segura até a próxima quando a articulação é longa.
        hits = sorted(hits)
        stretched = []
        for i, (pos, role, velocity, duration) in enumerate(hits):
            following = hits[i + 1][0] if i + 1 < len(hits) else self._slots
            if intent.articulation == "LONG" or intent.sustain_preferred:
                duration = following - pos
            stretched.append((pos, role, velocity, max(1, min(duration, following - pos))))
        bar.hits = stretched
        if bar.plan is not None:
            bar.plan.decision = ("Sustenta a fundamental" if intent.sustain_preferred else
                                 f"Modulação: {intent.modulation_action}"
                                 if intent.modulation_action != "KEEP" else
                                 f"Levada do estilo ({intent.density_target})")

    def _apply_vocabulary_level(self, hits, intent: BassMusicalIntent, bar: _Bar, change_slots):
        """Vocabulário pelo nível (mesmo número de notas — só muda O QUE soa):
        1 fundamental · 2 + quinta no tempo forte do meio · 3+ uma aproximação
        discreta para a troca de acorde · 5 + oitava num tempo fraco."""
        level = intent.vocabulary_level
        if level <= 1 or intent.sustain_preferred or intent.fallback_pattern:
            return [(p, "R" if r in ("5", "R8", "APP") else r, v, d) for p, r, v, d in hits]
        middle = (self._bpb // 2) * 4
        result = []
        for pos, role, velocity, duration in hits:
            if pos == middle and role == "R" and pos not in change_slots:
                role = "5"
            result.append((pos, role, velocity, duration))
        if level >= 3 and intent.approach_note_policy != "NONE":
            changes = sorted(change_slots | ({self._slots} if self._changes_after(bar) else set()))
            for change in changes:
                before = [i for i, h in enumerate(result) if 0 < h[0] < change and h[0] not in change_slots
                          and h[1] in ("R", "5") and h[0] != middle]
                if before:
                    i = before[-1]
                    pos, _, velocity, duration = result[i]
                    result[i] = (pos, "APP", velocity, duration)
                    break                       # uma condução por compasso: discreta
        if level >= 5:
            weak = [i for i, h in enumerate(result) if h[1] == "R" and h[0] % 4 == 0 and
                    h[0] not in (0, middle) and h[0] not in change_slots]
            if weak:
                pos, _, velocity, duration = result[weak[0]]
                result[weak[0]] = (pos, "R8", velocity, duration)
        return result

    def _changes_after(self, bar: _Bar) -> bool:
        last, following = bar.start + self._bpb - 1, bar.start + self._bpb
        return (0 <= last < self._beats and following < self._beats and
                self._bass_pc[following] not in (-1, self._bass_pc[last]))

    def _record_diagnostics(self, diagnostics: BassBrainDiagnostics, bars: List[_Bar],
                            notes: List[ArrangedNote]) -> None:
        by_bar: dict = {}
        for note in notes:
            by_bar.setdefault(note.bar, []).append(note)
        previous_groove = None
        for bar in bars:
            if not bar.active or bar.intent is None:
                continue
            bar_notes = by_bar.get(bar.index + 1, [])
            period = self._beat_period(max(0, bar.start))
            groove_name = bar.plan.groove if bar.plan else ""
            diagnostics.record_bar(
                bar.intent, [(n.time, n.note, n.role, n.velocity, n.duration / max(1e-6, period))
                             for n in bar_notes],
                groove_name, bar.plan.decision if bar.plan else "", self._bpb,
                fill=bool(bar.plan and bar.plan.fill),
                phrase_changed=previous_groove is not None and groove_name != previous_groove,
                groove_changed="REASON_GROOVE_TRANSITION" in bar.intent.reasons)
            previous_groove = groove_name

    def _make_ending(self, bar: _Bar) -> None:
        bar.hits = [(0, "R", 96, self._slots - 1)]
        if bar.plan:
            bar.plan.groove = "Final: nota longa"
            bar.plan.tier = 0

    def _maybe_fill(self, bar: _Bar, tier: int, section_end: bool,
                    creativity: int, rng: random.Random) -> None:
        if self._bpb != 4:
            return
        probability = (self._style.fill_section_prob if section_end
                       else self._style.fill_phrase_prob)[creativity]
        if tier == 0 and creativity < 4:
            probability *= 0.3
        if rng.random() >= probability:
            return
        fills = [f for f in self._style.fills if f.max_bpm >= self._bpm and f.min_tier <= tier]
        if bar.band is not None:
            # Virada em semicolcheias só se a banda também toca semicolcheias.
            sixteenths = float(np.max(bar.band[1::2])) >= 0.3
            if not sixteenths:
                fills = [f for f in fills if all(pos % 2 == 0 for pos, *_ in f.hits)]
        if not fills:
            return
        fill: Fill = rng.choices(fills, weights=[f.weight for f in fills], k=1)[0]
        bar.hits = [hit for hit in bar.hits if hit[0] < fill.start] + list(fill.hits)
        if bar.plan:
            bar.plan.fill = fill.label

    def _maybe_ornament(self, bar: _Bar, creativity: int, rng: random.Random) -> None:
        if rng.random() >= self._style.ornament_prob[creativity]:
            return
        weak = [i for i, hit in enumerate(bar.hits) if hit[1] == "R" and hit[0] % 8 in (2, 6)]
        if not weak:
            return
        i = rng.choice(weak)
        pos, _, velocity, duration = bar.hits[i]
        role = rng.choice(("R8", "R8", "5"))
        bar.hits[i] = (pos, role, velocity, duration)
        if bar.plan:
            bar.plan.variation = "oitava" if role == "R8" else "quinta"

    def _plan_approaches(self, bar: _Bar, tier: int, creativity: int, rng: random.Random) -> None:
        """Marca (ou acrescenta) a nota que conduz ao próximo acorde."""
        probability = self._style.approach_prob[creativity]
        if probability <= 0:
            return
        for offset in range(1, self._bpb + 1):
            change = bar.start + offset
            before = change - 1
            if not (0 <= before < self._beats and change < self._beats):
                continue
            if not self._chord(change) or self._bass_pc[change] in (-1, self._bass_pc[before]):
                continue
            beat_slots = range((offset - 1) * 4, offset * 4)
            in_beat = [i for i, hit in enumerate(bar.hits) if hit[0] in beat_slots]
            if self._band_anticipates(before) is True:
                # A banda antecipa a troca: o baixo antecipa junto, em qualquer levada.
                push_pos = (offset - 1) * 4 + 2
                at_push = [i for i in in_beat if bar.hits[i][0] == push_pos]
                if at_push:
                    pos, _, velocity, duration = bar.hits[at_push[0]]
                    bar.hits[at_push[0]] = (pos, "PUSH", velocity, duration)
                else:
                    bar.hits = [hit for hit in bar.hits if hit[0] != push_pos + 1]
                    bar.hits.append((push_pos, "PUSH", 94, 2))
                continue
            if any(bar.hits[i][1] in ("PUSH",) or bar.hits[i][1][0] in "SC" for i in in_beat):
                continue          # já há antecipação ou virada conduzindo
            weak = [i for i in in_beat if bar.hits[i][0] % 4 != 0 and bar.hits[i][1] in ("R", "R8", "5")]
            if weak:
                if rng.random() < probability:
                    i = weak[-1]
                    pos, _, velocity, duration = bar.hits[i]
                    bar.hits[i] = (pos, "APP", velocity, duration)
            elif (tier >= 1 and self._band_strength(bar, (offset - 1) * 4 + 2) >= 0.3 and
                  rng.random() < probability * 0.5):
                bar.hits.append(((offset - 1) * 4 + 2, "APP", 86, 2))

    def _lock_to_band(self, bar: _Bar, creativity: int, rng: random.Random) -> None:
        """Remove ataques onde a banda está em silêncio (o baixo "trava" na banda).

        O tempo 1 sempre fica; a antecipação só fica se a banda também antecipa (ela ataca ali).
        Nos níveis 4–5, ataques nos tempos e viradas podem
        ficar mesmo sem a banda (liberdade criativa); contratempos soltos, nunca.
        """
        if bar.band is None:
            return
        hits = sorted(bar.hits)
        keep_flags = []
        for pos, role, _, _ in hits:
            keep = pos == 0 or self._band_strength(bar, pos) >= 0.3
            if not keep and creativity >= 4 and (pos % 4 == 0 or role[0] in "SC" or role == "APP"):
                keep = rng.random() < (0.5 if creativity == 4 else 0.75)
            keep_flags.append(keep)
        bar.hits = []
        for i, (pos, role, velocity, duration) in enumerate(hits):
            if not keep_flags[i]:
                continue
            # Se o ataque seguinte saiu, a nota sustenta até o próximo que ficou
            # (sem apagar as pausas que a levada já tinha).
            if i + 1 < len(hits) and not keep_flags[i + 1] and role != "PUSH":
                following = next((hits[j][0] for j in range(i + 1, len(hits)) if keep_flags[j]),
                                 self._slots)
                duration = max(duration, following - pos)
            bar.hits.append((pos, role, velocity, duration))

    # ------------------------------------------------------------ notas
    def _resolve(self, bars: List[_Bar], creativity: int, rng: random.Random) -> List[ArrangedNote]:
        notes: List[ArrangedNote] = []
        previous_root = 36
        previous_note = 36
        tied_beat = -1
        global_scale = self._scale
        for bar in bars:
            if not bar.active:
                continue
            # Modulação: o raciocínio de escala usa o tom VIGENTE do compasso.
            self._scale = bar.scale or global_scale
            for pos, role, velocity, duration in sorted(bar.hits):
                k = bar.start + pos // 4
                slot = pos % 4
                chord = self._chord(k)
                if chord is None:
                    continue
                if k == tied_beat and slot == 0:
                    continue          # ligado à antecipação anterior
                target_beat = self._next_change(k)
                target_chord = self._chord(target_beat) if target_beat is not None else None
                reason = ""
                midi = None
                if role == "PUSH":
                    changes_next = (target_beat == k + 1 and slot == 2 and
                                    self._should_push(k, creativity, rng))
                    if changes_next:
                        midi = self._root_midi(self._bass_pc[k + 1], previous_root)
                        previous_root = midi
                        tied_beat = k + 1
                        next_duration = self._first_hit_duration(bars, k + 1)
                        duration += next_duration
                        chord = target_chord
                        reason = f"Antecipação → {target_chord}"
                    else:
                        role = "R"
                if midi is None and role in ("R", "R8", "5", "3"):
                    root = self._root_midi(self._bass_pc[k], previous_root)
                    previous_root = root
                    midi = self._chord_tone(chord, root, role)
                    reason = {"R": f"Fundamental ({chord})", "R8": f"Oitava ({chord})",
                              "5": f"Quinta ({chord})", "3": f"Terça ({chord})"}[role]
                elif midi is None:
                    # Aproximação mira a troca seguinte; virada mira o próximo tempo 1.
                    goal_beat = bar.start + self._bpb
                    if role == "APP" and target_beat is not None:
                        goal_beat = target_beat
                    goal_pc = self._bass_pc[goal_beat] if 0 <= goal_beat < self._beats else -1
                    if goal_pc < 0:
                        goal_pc = self._bass_pc[k]
                    goal_name = self._chord(goal_beat) or chord
                    target = self._root_midi(goal_pc, previous_root)
                    if bar.approach_to_new_key and pos >= self._slots - 2:
                        self._scale = bar.next_scale or self._scale     # escala do DESTINO
                    midi, reason = self._leading_note(role, target, previous_note, goal_name,
                                                      creativity, rng)
                    if bar.approach_to_new_key and pos >= self._slots - 2:
                        reason = f"Aproximação ao novo tom → {goal_name}"
                        self._scale = bar.scale or global_scale
                midi = int(np.clip(midi, BASS_MIN_MIDI_NOTE, BASS_MAX_MIDI_NOTE))
                period = self._beat_period(k)
                intent = bar.intent
                if intent is not None:
                    # Dinâmica da intenção (energia da seção + acento métrico), sem sorteio.
                    vel = intent.velocity_base + (intent.accent if pos == 0 else 0)
                    vel += -5 if pos % 4 else 0
                    vel += int(round(6 * (min(1.0, self._band_strength(bar, pos)) - 0.5)))
                    vel = int(np.clip(vel, 30, 127))
                    length_factor = intent.note_length_factor
                else:
                    level = 0.82 + 0.18 * bar.intensity
                    accent = 0.88 + 0.16 * min(1.0, self._band_strength(bar, pos))
                    vel = int(np.clip(round(velocity * level * accent + rng.uniform(-3, 3)), 30, 127))
                    length_factor = 0.92
                notes.append(ArrangedNote(
                    time=self._beat_time(k, slot), midi=midi, velocity=vel,
                    duration=duration * period / 4.0 * length_factor, beat_index=k, bar=bar.index + 1,
                    beat_in_bar=pos // 4 + 1, slot=pos, chord=chord, role=role, reason=reason))
                previous_note = midi
        self._scale = global_scale
        notes.sort(key=lambda note: note.time)
        cleaned: List[ArrangedNote] = []
        for note in notes:
            if cleaned and note.time - cleaned[-1].time < 0.01:
                continue
            cleaned.append(note)
        return cleaned

    def _next_change(self, k: int) -> Optional[int]:
        current = self._bass_pc[k]
        for j in range(k + 1, min(self._beats, k + 2 * self._bpb + 1)):
            if self._bass_pc[j] == -1:
                return None
            if self._bass_pc[j] != current:
                return j
        return None

    def _first_hit_duration(self, bars: List[_Bar], beat: int) -> int:
        for bar in bars:
            if bar.start <= beat < bar.start + self._bpb:
                for pos, _, _, duration in sorted(bar.hits):
                    if bar.start + pos // 4 == beat and pos % 4 == 0:
                        return duration
        return 2

    def _band_anticipates(self, k: int) -> Optional[bool]:
        """O fim da batida ``k`` já soa o próximo acorde? None = sem evidência."""
        tails = self._timeline.tail_chroma
        current, following = self._chord(k), self._chord(k + 1)
        if tails is None or not (0 <= k < len(tails)) or not current or not following:
            return None
        tail = tails[k]
        if float(np.linalg.norm(tail)) == 0:
            return None
        now_t, next_t = chord_template(current), chord_template(following)
        if now_t is None or next_t is None:
            return None
        difference = float(tail @ next_t) - float(tail @ now_t)
        if difference > 0.04:
            return True
        if difference < -0.08:
            return False
        return None

    def _should_push(self, k: int, creativity: int, rng: random.Random) -> bool:
        if creativity <= 1:
            return False
        evidence = self._band_anticipates(k)
        if evidence is True:
            return True            # a banda antecipa: o baixo antecipa junto
        if evidence is False:
            return False           # a banda segura o acorde: antecipar soaria errado
        return rng.random() < self._style.push_prob[creativity]

    @staticmethod
    def _root_midi(pc: int, previous: int) -> int:
        if pc < 0:
            return previous
        candidates = [m for m in range(_ROOT_LOW, _ROOT_HIGH + 1) if m % 12 == pc]
        return min(candidates, key=lambda m: (abs(m - previous), m))

    @staticmethod
    def _chord_tone(chord: str, root_midi: int, role: str) -> int:
        if role == "R":
            return root_midi
        if role == "R8":
            return root_midi + 12 if root_midi + 12 <= BASS_MAX_MIDI_NOTE else root_midi
        symbol = parse_chord(chord)
        chord_root = note_to_pc(symbol.root)
        if role == "5":
            interval = 6 if symbol.quality == "dim" else 8 if symbol.quality == "aug" else 7
        else:
            interval = 3 if symbol.quality in ("minor", "dim") else 4
        pc = (chord_root + interval) % 12
        midi = root_midi + ((pc - root_midi) % 12)
        if midi > BASS_MAX_MIDI_NOTE:
            midi -= 12
        return midi

    def _leading_note(self, role: str, target: int, previous: int, goal: str,
                      creativity: int, rng: random.Random) -> Tuple[int, str]:
        """Nota que conduz ao alvo. Por padrão SÓ notas da escala do tom; cromatismo
        apenas se o estilo permitir naquele nível de criatividade."""
        chromatic = rng.random() < self._style.chromatic_prob[creativity]
        if role == "APP":
            from_below = previous <= target
            if chromatic:
                return (target - 1 if from_below else target + 1), f"Aproximação cromática → {goal}"
            for direction in ((-1, 1) if from_below else (1, -1)):
                midi = self._scale_step(target, direction)
                if midi != previous:
                    return midi, f"Passagem na escala → {goal}"
            return self._scale_step(target, -1), f"Passagem na escala → {goal}"
        kind, amount = role[0], int(role[1:])
        if kind == "C":
            if chromatic:
                return target + amount, f"Virada (cromática) → {goal}"
            return self._scale_step(target, amount), f"Virada (escala) → {goal}"
        return self._scale_step(target, amount), f"Virada (escala) → {goal}"

    def _scale_step(self, target: int, steps: int) -> int:
        """``steps`` graus da escala abaixo (<0) ou acima (>0) de ``target``."""
        scale = self._scale or tuple((target + s) % 12 for s in _MAJOR)
        direction = 1 if steps > 0 else -1
        midi, remaining = target, abs(steps)
        while remaining:
            midi += direction
            if midi % 12 in scale:
                remaining -= 1
        return midi
