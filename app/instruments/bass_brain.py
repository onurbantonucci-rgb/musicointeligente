"""BassBrain: o baixista que decide COMPASSO A COMPASSO, como ao vivo.

Antes de cada compasso ele sabe apenas:
- o que a banda tocou nos compassos que já passaram (onde atacou, com que
  força) e o andamento;
- o que a CIFRA diz que vem: acordes do compasso, onde trocam, o próximo
  acorde, se começa uma seção nova, o tom.

Com isso ele procura no banco de levadas reais (``BassBank``) os compassos que
baixistas de verdade tocaram em contextos parecidos, escolhe um e o aplica
sobre os acordes da cifra. Quando o contexto continua parecido, prefere seguir
a MESMA linha do baixista real no compasso seguinte — frases coerentes, com as
viradas reais nos fins de frase.

O que é decidido pela MÚSICA (não pela criatividade): densidade e ritmo — vêm
do banco (o que se toca com uma banda assim, nesse andamento) e são presos aos
ataques da banda. O que a CRIATIVIDADE decide: o vocabulário melódico
(nível 1 só fundamental … nível 5 graus da escala, terças, aproximações), o
quanto ele se arrisca em linhas menos comuns e o quanto varia entre frases.

O mesmo código serve para o modo com música gravada (simulando o ao vivo, sem
olhar para frente) e para o modo ao vivo.
"""

from dataclasses import dataclass, field
import math
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

from app.instruments.bass_bank import BankQuery, BassBank, band_lock_penalty, top_candidates
from app.instruments.bass_intent import (APPROACH, DIRECT_CHANGE, PREPARE, BassMusicalIntent,
                                         PhraseMemory)
from app.music.chord_chart import parse_chord
from app.music.constants import BASS_MAX_MIDI_NOTE, BASS_MIN_MIDI_NOTE
from app.music.theory import PITCH_CLASSES, note_to_pc

_ROOT_LOW, _ROOT_HIGH = BASS_MIN_MIDI_NOTE, 43
_MAJOR = (0, 2, 4, 5, 7, 9, 11)

# Vocabulário melódico permitido por nível de criatividade.
ALLOWED_ROLES = {
    1: {"R"},
    2: {"R", "R8", "5"},
    3: {"R", "R8", "5", "3", "APP", "U+1", "U-1"},
    4: {"R", "R8", "5", "3", "APP", "U+1", "U-1", "U+2", "U+3", "U-2", "7"},
    5: {"R", "R8", "5", "3", "APP", "U+1", "U-1", "U+2", "U+3", "U+4", "U+5", "U+6",
        "U-2", "U-3", "7"},
}
# Quando uma função não é permitida no nível, ela cai para a mais simples.
_DOWNGRADE = {"CHR": "APP", "7": "5", "U+4": "5", "U+5": "R8", "U+6": "R8", "U-3": "5",
              "U+3": "5", "U+2": "3", "U-2": "APP", "U+1": "APP", "U-1": "APP",
              "APP": "R", "3": "5", "5": "R", "R8": "R"}
# Preferência por linhas melodicamente ricas (fração de notas que não são a fundamental).
_MELODIC_PREFERENCE = (0, 0.0, 0.0, 0.3, 0.9, 1.5)
_TEMPERATURE = (0, 0.15, 0.25, 0.35, 0.5, 0.7)
_CONTINUITY_BONUS = (0, 1.8, 1.6, 1.4, 1.0, 0.7)
_LOCK_PENALTY = 6.0          # igual em todos os níveis: a densidade é da música


@dataclass
class BarRequest:
    """O que o baixista sabe antes de tocar o compasso."""
    bar_number: int
    beat_times: Sequence[float]                 # bpb+1 fronteiras das batidas do compasso
    chords: Sequence[Optional[str]]             # acorde da cifra em cada tempo
    next_chord: Optional[str]                   # acorde do próximo tempo 1
    previous_chord: Optional[str] = None        # acorde que soava antes do tempo 1
    band_history: Sequence[np.ndarray] = ()     # levadas (16 semicolcheias) dos compassos passados
    energy_history: Sequence[float] = ()
    section_start: bool = False
    section_type: str = ""
    final_bar: bool = False                     # a música acaba / para depois deste compasso
    # Intenção musical do compasso (seção, energia, densidade, viradas, modulação).
    intent: Optional[BassMusicalIntent] = None
    scale: Optional[Tuple[int, ...]] = None     # escala do tom VIGENTE (modulações da cifra)
    next_scale: Optional[Tuple[int, ...]] = None  # escala do tom do próximo compasso


@dataclass
class PlannedNote:
    time: float
    midi: int
    velocity: int
    duration: float
    slot: int
    chord: str
    role: str
    reason: str

    @property
    def note(self) -> str:
        return f"{PITCH_CLASSES[self.midi % 12]}{self.midi // 12 - 1}"


@dataclass
class BarDecision:
    notes: List[PlannedNote]
    label: str
    source: str = ""
    intensity: float = 0.5
    level: int = 1
    variation: str = ""
    decision: str = ""                           # resumo da decisão (painel do cérebro)
    phrase_id: str = ""
    fill: bool = False
    phrase_changed: bool = False


@dataclass
class _Memory:
    last_index: Optional[int] = None
    anticipation: float = 0.0                   # 0..1: a banda costuma antecipar as trocas?
    anticipation_seen: int = 0
    tie_into_next: bool = False
    previous_root: int = 36
    previous_note: int = 36
    history_rhythms: List[Tuple[int, ...]] = field(default_factory=list)


_SUBDIVISION_INDEX = {"QUARTER": 0, "EIGHTH": 1, "SIXTEENTH": 2}
_TARGET_DURATION = {"LONG": 8.0, "MEDIUM": 4.0, "SHORT": 2.0}   # semicolcheias
# Pesos do ENCAIXE com a intenção (somados ao score de contexto da banda do banco):
_DENSITY_FIT = 2.5          # ataques do compasso x ataques que o momento pede
_ARTICULATION_FIT = 0.8     # notas longas/curtas x articulação da seção
_SUBDIVISION_FIT = 1.0      # subdivisão mais fina que a da levada ouvida
_SYNCOPATION_FIT = 1.2      # síncope além do que a seção pede
_TRANSITION_FIT = 0.8       # virada fora de um ponto de transição
# Métrica: quem fica quando é preciso tirar ataques (tempo 1 > 3 > 2/4 > contratempo).
_SLOT_PRIORITY = {0: 100.0, 8: 60.0, 4: 40.0, 12: 40.0}


class BassBrain:
    def __init__(self, bank: BassBank, creativity: int = 3, seed: int = 0,
                 scale: Optional[Tuple[int, ...]] = None, chromatic: bool = False,
                 phrase_memory: Optional[PhraseMemory] = None):
        self._bank = bank
        self._creativity = max(1, min(5, int(creativity)))
        self._rng = random.Random(seed * 7_919 + self._creativity)
        self._scale = scale
        self._chromatic = chromatic
        self._memory = _Memory()
        self._tied_pending = False
        self.phrase_memory = phrase_memory
        self._intent: Optional[BassMusicalIntent] = None

    # ------------------------------------------------------------ aprendizado
    def observe_anticipation(self, anticipated: bool) -> None:
        """A banda antecipou (ou segurou) uma troca que acabou de passar."""
        memory = self._memory
        memory.anticipation_seen += 1
        rate = 0.35 if memory.anticipation_seen > 2 else 0.6
        memory.anticipation += rate * ((1.0 if anticipated else 0.0) - memory.anticipation)

    # ------------------------------------------------------------ decisão
    _CALM_SECTIONS = ("INTRO", "OUTRO", "FINAL", "ENDING", "CODA", "INTERLUDE")

    def plan_bar(self, request: BarRequest) -> BarDecision:
        intent = self._intent = request.intent
        if request.scale is not None:
            self._scale = tuple(request.scale)          # modulação: recontextualiza a escala
        band = self._recent_band(request.band_history)
        if (request.section_start and band is not None and
                any(name in request.section_type.upper() for name in self._CALM_SECTIONS)):
            # A cifra avisa que entra uma parte calma: a levada de antes não vale
            # mais. Toca o essencial até ouvir como a banda vai tocar.
            band = np.zeros(16)
            band[0], band[8] = 1.0, 0.6
        bpm = self._bpm(request.beat_times)
        trend = self._trend(request.energy_history)
        changes = self._changes(request)
        intensity = float(np.clip(0.5 + 0.35 * math.log2(max(0.25, trend)), 0.0, 1.0))
        if request.final_bar:
            return self._ending(request, intensity)
        decision = "Linha do banco"
        if intent is not None and intent.fallback_pattern:
            # Pouca certeza da posição/harmonia: menos risco, não mais notas.
            hits = self._fallback_hits(intent.fallback_pattern, request)
            label, source, decision = f"Seguro: {intent.fallback_pattern}", "", "Fallback seguro"
            self._memory.last_index = None
        elif band is None:
            # Ainda sem ouvir a banda: entra seguro, na fundamental do tempo 1.
            hits = [(0, 16, "R", 100)]
            label, source, decision = "Entrada: escutando a banda", "", "Observando"
            self._memory.last_index = None
        else:
            index = self._choose(BankQuery(band, bpm, trend, changes), band, request)
            if index is None:
                # Nenhuma frase do banco serve a este momento: não usa uma ruim.
                pattern = ("ROOT_FIFTH" if intent is not None and intent.attacks_cap >= 3.5
                           else "ROOT" if intent is None or intent.attacks_cap >= 1.6 else "SUSTAINED")
                hits = self._fallback_hits(pattern, request)
                label, source, decision = f"Sem frase compatível: {pattern}", "", "Fallback do banco"
                self._memory.last_index = None
            else:
                hits = [(n.slot, n.duration, n.role, n.velocity) for n in self._bank.notes(index)]
                self._memory.last_index = index
                label, source = self._label(index), self._bank.song_name(index)
                hits = self._lock(hits, band)
        hits = self._one_per_slot(hits)
        hits = self._apply_vocabulary(hits, request)
        fill = False
        if intent is not None:
            hits, fill, decision = self._fit_to_intent(hits, band, request, decision)
        hits = self._maybe_push(hits, band, request)
        notes = self._resolve(hits, request, band)
        phrase_id, phrase_changed = self._remember_phrase(hits)
        level = 0 if len(notes) <= 2 else 1 if len(notes) <= 4 else 2 if len(notes) <= 7 else 3
        return BarDecision(notes=notes, label=label, source=source, intensity=intensity, level=level,
                           decision=decision, phrase_id=phrase_id, fill=fill,
                           phrase_changed=phrase_changed)

    # ------------------------------------------------------------ intenção
    def _fallback_hits(self, pattern: str, request: BarRequest):
        """Padrões seguros (fundamental sustentada / tônica / tônica e quinta)."""
        change_slots = [b * 4 for b in range(1, min(4, len(request.chords)))
                        if request.chords[b] and request.chords[b] != request.chords[b - 1]]
        if pattern == "SUSTAINED":
            slots = [0] + change_slots
        else:                                   # ROOT e ROOT_FIFTH: tempos 1 e 3 (+ trocas)
            slots = sorted({0, 8} | set(change_slots))
        hits = []
        for i, slot in enumerate(slots):
            following = slots[i + 1] if i + 1 < len(slots) else 16
            role = "5" if pattern == "ROOT_FIFTH" and slot == 8 and slot not in change_slots else "R"
            hits.append((slot, following - slot, role, 100 if slot == 0 else 88))
        return hits

    def _fit_to_intent(self, hits, band, request: BarRequest, decision: str):
        """Ajusta a frase ao que o momento pede: densidade, sustentação, viradas, modulação."""
        intent = self._intent
        change_slots = {b * 4 for b in range(1, min(4, len(request.chords)))
                        if request.chords[b] and request.chords[b] != request.chords[b - 1]}
        essential = {0} | change_slots
        hits = sorted(hits)
        if intent.sustain_preferred:
            # Acorde longo em momento calmo: segura a nota (respira), não repete.
            hits = [h for h in hits if h[0] in essential]
            decision = "Sustenta a fundamental"
        cap = max(len(essential), int(round(intent.attacks_cap)))
        if len(hits) > cap:
            def priority(hit):
                slot = hit[0]
                metric = _SLOT_PRIORITY.get(slot, 25.0 if slot % 2 == 0 else 10.0)
                strength = float(band[slot]) if band is not None else 0.5
                return (slot in essential, metric + 20.0 * strength)
            keep = sorted(sorted(hits, key=priority, reverse=True)[:cap])
            kept_slots = {h[0] for h in keep}
            hits = [h for h in hits if h[0] in kept_slots]
            if decision == "Linha do banco":
                decision = "Frase do banco podada (densidade da seção)"
        present = {h[0] for h in hits}
        for slot in sorted(essential - present):
            hits.append((slot, 4, "R", 96))      # troca de acorde sempre ganha o seu ataque
        hits = sorted(hits)
        # Durações: cada nota segura até a próxima (nada de buracos criados pela poda).
        stretched = []
        for i, (slot, duration, role, velocity) in enumerate(hits):
            following = hits[i + 1][0] if i + 1 < len(hits) else 16
            if intent.articulation == "LONG" or intent.sustain_preferred:
                duration = following - slot
            stretched.append((slot, min(max(duration, 1), following - slot) if following > slot
                              else duration, role, velocity))
        hits = stretched
        fill = False
        # VIRADA = condução com 2+ notas na 2ª metade do compasso: só nos pontos de
        # transição. Uma APROXIMAÇÃO discreta (uma nota levando ao próximo acorde) é
        # vocabulário normal quando a intenção permite aproximações.
        moving = [h for h in hits if h[0] >= 8 and (h[2] == "APP" or h[2].startswith("U"))]
        if len(moving) >= 2 and intent.transition_fill_allowed:
            fill = True
        elif moving:
            keep_last = intent.approach_note_policy != "NONE"
            last_slot = moving[-1][0] if keep_last else -1
            hits = [(s, d, r if (s == last_slot or not (s >= 8 and (r == "APP" or r.startswith("U"))))
                     else ("5" if s == 8 else "R"), v) for s, d, r, v in hits]
        if intent.modulation_action == APPROACH and request.next_chord:
            hits = [h for h in hits if h[0] < 12] + [(12, 2, "R", 96), (14, 2, "KEY_APP", 94)]
            decision = f"Aproximação para o novo tom ({intent.next_key})"
            fill = True
        elif intent.modulation_action in (DIRECT_CHANGE, PREPARE):
            hits = [(s, d, "R" if r in ("APP", "PUSH") and s >= 12 else r, v) for s, d, r, v in hits]
            if intent.modulation_action == PREPARE:
                hits = [h for h in hits if h[0] <= 8]
                decision = f"Prepara a modulação segurando ({intent.next_key})"
            else:
                decision = f"Troca direta de tom ({intent.next_key})"
        return hits, fill, decision

    def _remember_phrase(self, hits) -> Tuple[str, bool]:
        memory = self.phrase_memory
        if memory is None:
            return "", False
        rhythm = np.zeros(16)
        for slot, *_ in hits:
            if 0 <= slot < 16:
                rhythm[slot] = 1.0
        before = memory.phrase_changes
        memory.record(rhythm)
        return memory.motif_id, memory.phrase_changes != before

    def _choose(self, query: BankQuery, band: np.ndarray, request: BarRequest) -> Optional[int]:
        bank, c = self._bank, self._creativity
        intent = self._intent
        scores = bank.score(query)
        scores -= _LOCK_PENALTY * band_lock_penalty(bank.rhythm, band)
        vocabulary = intent.vocabulary_level if intent is not None else c
        scores += _MELODIC_PREFERENCE[vocabulary] * bank.melodic
        if intent is not None:
            density_gap = np.abs(np.log2((bank.notes_per_bar + 1.0) / (intent.attacks_cap + 1.0)))
            scores -= _DENSITY_FIT * density_gap
            target = _TARGET_DURATION.get(intent.articulation, 4.0)
            scores -= _ARTICULATION_FIT * np.abs(np.log2((bank.mean_duration + 1.0) / (target + 1.0)))
            finer = bank.subdivision - _SUBDIVISION_INDEX.get(intent.subdivision, 1)
            scores -= _SUBDIVISION_FIT * np.maximum(0, finer)
            scores -= _SYNCOPATION_FIT * np.maximum(0.0, bank.syncopation - (intent.syncopation_target + 0.15))
            if not intent.transition_fill_allowed:
                # Só VIRADA (2+ notas de condução no fim) é penalizada fora da transição.
                scores -= _TRANSITION_FIT * np.clip(bank.fill_level - 1.0, 0.0, 3.0)
            if self.phrase_memory is not None:
                scores += self.phrase_memory.bonus(bank.rhythm, intent.variation_budget)
        previous = self._memory.last_index
        if previous is not None and not request.section_start:
            following = bank.successor(previous)
            if following is not None:
                scores[following] += _CONTINUITY_BONUS[c]
        candidates = top_candidates(scores, 12)
        if intent is not None and density_gap[candidates[0]] > 1.6:
            # Até a melhor frase tem 3x mais (ou menos) notas do que o momento pede:
            # o banco é vocabulário, não autoridade — usa o padrão seguro.
            return None
        weights = np.exp((scores[candidates] - scores[candidates[0]]) / _TEMPERATURE[c])
        return int(self._rng.choices(list(candidates), weights=list(weights), k=1)[0])

    def _label(self, index: int) -> str:
        notes = self._bank.notes(index)
        roles = {n.role for n in notes}
        count = len(notes)
        rhythm = ("nota longa" if count <= 1 else "mínimas" if count == 2 else
                  "semínimas" if count <= 4 else "colcheias" if count <= 8 else "semicolcheias")
        if roles - {"R", "R8"}:
            return f"Linha real · {rhythm} com condução"
        return f"Linha real · {rhythm}"

    # ------------------------------------------------------------ regras
    @staticmethod
    def _recent_band(history: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        grids = [np.asarray(g, dtype=np.float64) for g in list(history)[-2:]
                 if g is not None and float(np.max(g)) > 0]
        if not grids:
            return None
        normalized = [g / float(np.max(g)) for g in grids]
        return np.mean(normalized, axis=0)

    @staticmethod
    def _bpm(beat_times: Sequence[float]) -> float:
        diffs = np.diff(np.asarray(beat_times, dtype=np.float64))
        return 60.0 / float(np.median(diffs)) if len(diffs) and np.median(diffs) > 0 else 120.0

    @staticmethod
    def _trend(energy: Sequence[float]) -> float:
        values = [float(e) for e in energy if e is not None]
        if len(values) < 2:
            return 1.0
        recent, before = values[-1], values[-9:-1]
        mean = float(np.mean(before)) if before else 0.0
        return recent / mean if mean > 0 else 1.0

    @staticmethod
    def _changes(request: BarRequest) -> int:
        """Máscara das trocas de acorde no compasso (bit 4*b = troca no tempo b+1)."""
        mask = 0
        for beat, chord in enumerate(request.chords[:4]):
            before = request.chords[beat - 1] if beat > 0 else request.previous_chord
            if chord and chord != before:
                mask |= 1 << (beat * 4)
        return mask

    @staticmethod
    def _one_per_slot(hits):
        seen, result = set(), []
        for hit in sorted(hits, key=lambda h: h[0]):
            if hit[0] not in seen:
                seen.add(hit[0])
                result.append(hit)
        return result

    def _lock(self, hits, band):
        """Nenhum ataque onde a banda está em silêncio (o tempo 1 fica), em todos os níveis."""
        if band is None:
            return hits
        kept = []
        for i, (slot, duration, role, velocity) in enumerate(hits):
            if slot == 0 or band[slot] >= 0.3:
                kept.append([slot, duration, role, velocity])
            elif kept:
                # A nota anterior sustenta no lugar do ataque removido.
                following = hits[i + 1][0] if i + 1 < len(hits) else 16
                kept[-1][1] = max(kept[-1][1], following - kept[-1][0])
        if not kept or kept[0][0] != 0:
            # Sem ataque no tempo 1 (o baixista real ligou do compasso anterior):
            # mantém a ligação só se for o caso; senão, o tempo 1 é do baixo.
            if not self._memory.tie_into_next:
                first = kept[0][0] if kept else 16
                kept.insert(0, [0, first, "R", 100])
        return [tuple(h) for h in kept]

    def _apply_vocabulary(self, hits, request: BarRequest):
        # Criatividade = vocabulário POSSÍVEL; o momento (energia/confiança) limita o teto.
        level = self._intent.vocabulary_level if self._intent is not None else self._creativity
        allowed = ALLOWED_ROLES[max(1, min(5, level))]
        result = []
        for slot, duration, role, velocity in hits:
            while role not in allowed:
                role = _DOWNGRADE.get(role, "R")
            beat = min(len(request.chords) - 1, slot // 4)
            # Tempo 1: só a nota do baixo da cifra (outra nota ali viraria uma inversão
            # que a cifra não pede). Tempo 3: fundamental ou quinta.
            if slot == 0 and role not in ("R", "R8"):
                role = "R"
            elif slot == 8 and role not in ("R", "R8", "5"):
                role = "R"
            # Aproximação só faz sentido se há troca à frente no compasso (ou no próximo).
            if role == "APP" and self._target_after(request, beat) is None:
                role = "5" if "5" in allowed else "R"
            result.append((slot, duration, role, velocity))
        return result

    def _maybe_push(self, hits, band, request: BarRequest):
        """Antecipa a troca do próximo tempo 1 se a banda costuma antecipar."""
        if (self._creativity < 2 or band is None or request.next_chord is None or
                (self._intent is not None and (not self._intent.push_allowed or
                                               self._intent.modulation_action != "KEEP"))):
            self._memory.tie_into_next = False
            return hits
        last_chord = request.chords[-1] if request.chords else None
        changes = (request.next_chord != last_chord and
                   self._bass_pc(request.next_chord) != self._bass_pc(last_chord))
        if not (changes and self._memory.anticipation >= 0.6 and band[14] >= 0.3):
            self._memory.tie_into_next = False
            return hits
        hits = [h for h in hits if h[0] < 14] + [(14, 2, "PUSH", 98)]
        self._memory.tie_into_next = True
        return hits

    # ------------------------------------------------------------ alturas
    def _resolve(self, hits, request: BarRequest, band) -> List[PlannedNote]:
        memory = self._memory
        notes: List[PlannedNote] = []
        beat_times = request.beat_times
        beats = len(beat_times) - 1
        tied_from_previous = self._tied_pending
        self._tied_pending = memory.tie_into_next
        for slot, duration, role, velocity in sorted(hits):
            beat = min(beats - 1, slot // 4)
            chord = request.chords[beat]
            if not chord:
                continue
            if slot == 0 and tied_from_previous:
                continue                                   # ligada da antecipação anterior
            if role == "PUSH":
                chord = request.next_chord
                midi = self._root_midi(self._bass_pc(chord), memory.previous_root)
                memory.previous_root = midi
                reason = f"Antecipação com a banda → {chord}"
            elif role in ("R", "R8", "5", "3", "7"):
                root = self._root_midi(self._bass_pc(chord), memory.previous_root)
                memory.previous_root = root
                midi = self._chord_tone(chord, root, role)
                reason = {"R": f"Fundamental ({chord})", "R8": f"Oitava ({chord})",
                          "5": f"Quinta ({chord})", "3": f"Terça ({chord})",
                          "7": f"Sétima ({chord})"}[role]
            elif role == "KEY_APP":
                # Aproximação ao primeiro acorde do NOVO tom, pela escala do destino.
                target_chord = request.next_chord or chord
                target = self._root_midi(self._bass_pc(target_chord), memory.previous_root)
                midi = self._approach(target, memory.previous_note, request.next_scale)
                reason = f"Aproximação ao novo tom → {target_chord}"
            elif role == "APP":
                target_chord = self._target_after(request, beat) or chord
                target = self._root_midi(self._bass_pc(target_chord), memory.previous_root)
                midi = self._approach(target, memory.previous_note)
                reason = f"Passagem na escala → {target_chord}"
            else:                                           # U+n / U-n: graus a partir da fundamental
                root = self._root_midi(self._bass_pc(chord), memory.previous_root)
                steps = int(role[1:])
                midi = self._scale_step(root, steps)
                reason = f"Linha pela escala ({chord})"
            midi = int(np.clip(midi, BASS_MIN_MIDI_NOTE, BASS_MAX_MIDI_NOTE))
            start = beat_times[beat]
            period = beat_times[beat + 1] - beat_times[beat]
            time = start + period * (slot % 4) / 4.0
            strength = min(1.0, float(band[slot])) if band is not None else 0.6
            intent = self._intent
            if intent is not None:
                # Dinâmica pela INTENÇÃO (energia da seção + acento métrico + frase),
                # sem sorteio: humanização por fraseado, não por ruído.
                vel = intent.velocity_base + (intent.accent if slot == 0 else 0)
                vel += 3 if slot == 8 else -5 if slot % 4 else 0
                vel += 4 if (intent.bars_left_in_phrase == intent.phrase_length_bars and slot == 0) else 0
                vel += int(round(6 * (strength - 0.5)))
                vel = int(np.clip(vel, 30, 127))
                length = duration * period / 4.0 * intent.note_length_factor
            else:
                accent = 0.88 + 0.16 * strength
                vel = int(np.clip(round(velocity * accent + self._rng.uniform(-3, 3)), 30, 127))
                length = duration * period / 4.0 * 0.92
            if role == "PUSH":
                length += period * 0.9                      # ligada ao tempo 1 seguinte
            notes.append(PlannedNote(time, midi, vel, length, slot, chord, role, reason))
            memory.previous_note = midi
        return notes

    def _ending(self, request: BarRequest, intensity: float) -> BarDecision:
        chord = next((c for c in request.chords if c), None)
        self._memory.tie_into_next = False
        self._tied_pending = False
        if chord is None:
            return BarDecision([], "Silêncio", intensity=intensity, level=0)
        midi = self._root_midi(self._bass_pc(chord), self._memory.previous_root)
        start, end = request.beat_times[0], request.beat_times[-1]
        note = PlannedNote(start, midi, 96, (end - start) * 0.95, 0, chord, "R",
                           f"Final: nota longa ({chord})")
        return BarDecision([note], "Final: nota longa", intensity=intensity, level=0)

    def _target_after(self, request: BarRequest, beat: int) -> Optional[str]:
        current = request.chords[beat]
        for later in request.chords[beat + 1:]:
            if later and self._bass_pc(later) != self._bass_pc(current):
                return later
        if request.next_chord and self._bass_pc(request.next_chord) != self._bass_pc(current):
            return request.next_chord
        return None

    @staticmethod
    def _bass_pc(chord: Optional[str]) -> int:
        if not chord:
            return -1
        symbol = parse_chord(chord)
        return note_to_pc(symbol.bass_note or symbol.root)

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
        elif role == "3":
            interval = 3 if symbol.quality in ("minor", "dim") else 4
        else:  # sétima: só a que o acorde tem; sem sétima na cifra, usa a quinta
            extension = symbol.extension or ""
            if "maj7" in extension or "7M" in extension:
                interval = 11
            elif "7" in extension or "9" in extension:
                interval = 10
            else:
                interval = 7
        pc = (chord_root + interval) % 12
        midi = root_midi + ((pc - root_midi) % 12)
        return midi - 12 if midi > BASS_MAX_MIDI_NOTE else midi

    def _approach(self, target: int, previous: int,
                  scale: Optional[Tuple[int, ...]] = None) -> int:
        from_below = previous <= target
        for direction in ((-1, 1) if from_below else (1, -1)):
            midi = self._scale_step(target, direction, scale)
            if midi != previous:
                return midi
        return self._scale_step(target, -1, scale)

    def _scale_step(self, start: int, steps: int,
                    scale: Optional[Tuple[int, ...]] = None) -> int:
        """``steps`` graus da escala do tom acima (>0) ou abaixo (<0) de ``start``."""
        scale = scale or self._scale or tuple((start + s) % 12 for s in _MAJOR)
        direction = 1 if steps > 0 else -1
        midi, remaining = start, abs(steps)
        while remaining:
            midi += direction
            if midi % 12 in scale:
                remaining -= 1
        return midi
