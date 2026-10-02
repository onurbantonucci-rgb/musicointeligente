"""Monta o banco de levadas de baixo reais a partir de arquivos MIDI.

Fonte: Lakh MIDI Dataset (clean_midi), Colin Raffel, CC-BY 4.0 —
https://colinraffel.com/projects/lmd/ ("Learning-Based Methods for Comparing
Sequences, with Applications to Audio-to-MIDI Alignment and Matching", 2016).

Para cada compasso 4/4 de uma música com baixo, guarda:
- CONTEXTO causal (o que um baixista ao vivo já sabe antes do compasso):
  levada da banda nos 2 compassos anteriores (força dos ataques por
  semicolcheia), andamento, tendência de intensidade, e onde os acordes trocam
  dentro do compasso (isso vem da cifra);
- o que o baixista REAL tocou: semicolcheia, duração, intensidade e a FUNÇÃO
  de cada nota em relação ao acorde (fundamental, oitava, quinta, terça,
  aproximação do próximo acorde, grau da escala acima/abaixo da fundamental).

Como as notas são guardadas como função, o padrão pode ser tocado sobre
qualquer acorde da cifra, em qualquer tom.

Uso:  python -m tools.bass_bank.build_bank <pasta clean_midi> <saída .npz> [estilo]
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.bass_bank.midi_reader import MidiNote, MidiSong, read_midi

ROLES = ["R", "R8", "5", "3", "7", "APP", "U+1", "U+2", "U+3", "U+4", "U+5", "U+6",
         "U-1", "U-2", "U-3", "CHR"]
ROLE_ID = {role: i for i, role in enumerate(ROLES)}

POOLS: Dict[str, Sequence[str]] = {
    # Sonoridade do worship atual: balada pop, pop rock, CCM e soul/gospel.
    "GOSPEL_WORSHIP": (
        "ABBA", "Air Supply", "Amy Grant", "Anita Baker", "Babyface", "Backstreet Boys",
        "Bee Gees", "Billy Joel", "Bon Jovi", "Bonnie Tyler", "Boyz II Men", "Boyzone",
        "Brian McKnight", "Bryan Adams", "Celine Dion", "Cher", "Chicago", "Christopher Cross",
        "dc Talk", "Eagles", "Elton John", "Enya", "Eric Clapton", "Fleetwood Mac", "Foreigner",
        "Garth Brooks", "George Michael", "Goo Goo Dolls", "Jars of Clay", "John Elton",
        "John Lennon", "Journey", "Kenny Loggins", "LeAnn Rimes", "Lionel Richie",
        "Luther Vandross", "Mahalia Jackson", "Mariah Carey", "Michael Bolton",
        "Natalie Imbruglia", "Oasis", "Paul McCartney", "Peter Cetera", "Peter Gabriel",
        "Phil Collins", "REO Speedwagon", "Richard Marx", "RICHIE LIONEL", "Robbie Williams",
        "Rod Stewart", "Roxette", "Savage Garden", "Shania Twain", "Sheryl Crow", "Simple Minds",
        "Simply Red", "Spandau Ballet", "Stevie Wonder", "Sting", "Take That", "Tears for Fears",
        "The Beatles", "The Corrs", "Tina Turner", "Toni Braxton", "Toto", "Twila Paris", "U2",
        "Westlife", "Whitney Houston",
    ),
}

_MAJOR = (0, 2, 4, 5, 7, 9, 11)
_MINOR = (0, 2, 3, 5, 7, 8, 10)
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _triad_templates() -> Tuple[np.ndarray, List[Tuple[int, str]]]:
    templates, labels = [], []
    for root in range(12):
        for quality, third in (("maj", 4), ("min", 3)):
            t = np.zeros(12)
            t[root], t[(root + third) % 12], t[(root + 7) % 12] = 1.0, 0.8, 0.8
            templates.append(t / np.linalg.norm(t))
            labels.append((root, quality))
    return np.array(templates), labels


_TEMPLATES, _LABELS = _triad_templates()


def _is_four_four(song: MidiSong) -> bool:
    return all((num, den) == (4, 4) for _, num, den in song.time_signatures)


def _bass_notes(song: MidiSong) -> List[MidiNote]:
    best: List[MidiNote] = []
    for track in song.tracks:
        notes = [n for n in track.notes if n.channel != 9]
        bassy = [n for n in notes if 32 <= n.program <= 39]
        if not bassy and "bass" in track.name.lower():
            bassy = notes
        if len(bassy) > len(best) and np.median([n.pitch for n in bassy]) < 55:
            best = bassy
    # Monofônico: em ataques simultâneos fica a nota mais grave.
    by_start: Dict[int, MidiNote] = {}
    for note in best:
        current = by_start.get(note.start)
        if current is None or note.pitch < current.pitch:
            by_start[note.start] = note
    return sorted(by_start.values(), key=lambda n: n.start)


def _estimate_key(notes: Sequence[MidiNote]) -> Tuple[int, Tuple[int, ...]]:
    hist = np.zeros(12)
    for n in notes:
        hist[n.pitch % 12] += n.end - n.start
    best, scale = -1e9, _MAJOR
    tonic = 0
    for root in range(12):
        for profile, steps in ((_KS_MAJOR, _MAJOR), (_KS_MINOR, _MINOR)):
            score = float(np.corrcoef(hist, np.roll(profile, root))[0, 1])
            if score > best:
                best, tonic, scale = score, root, steps
    return tonic, tuple((tonic + s) % 12 for s in scale)


class _SongBars:
    """Grade de semicolcheias de uma música em 4/4."""

    def __init__(self, song: MidiSong, bass: List[MidiNote]):
        self.song = song
        self.slot = song.ticks_per_beat / 4.0
        self.bar_ticks = song.ticks_per_beat * 4
        bass_lines = {(n.track, n.channel) for n in bass}
        others = [n for t in song.tracks for n in t.notes if (n.track, n.channel) not in bass_lines]
        self.bass = bass
        self.harmony = [n for n in others if n.channel != 9]
        self.drums = [n for n in others if n.channel == 9]
        end = max(n.end for t in song.tracks for n in t.notes)
        self.bars = int(end // self.bar_ticks) + 1
        if self.bars > 600:
            # MIDI corrompido (evento num tick absurdo): descarta antes de alocar grades.
            raise ValueError("música longa demais")
        self._band_grids = self._compute_band_grids()
        self._energy = self._compute_energy()
        self._chords = self._compute_beat_chords()

    def _compute_band_grids(self) -> np.ndarray:
        grids = np.zeros((self.bars, 16))
        per_track: Dict[Tuple[int, int, int], float] = {}
        for n in self.harmony + self.drums:
            pos = n.start / self.slot
            slot = int(round(pos))
            if abs(pos - slot) > 0.34:
                continue
            bar, s = divmod(slot, 16)
            if bar >= self.bars:
                continue
            if n.channel == 9:
                weight = 1.0 if n.pitch in (35, 36, 38, 40) else 0.5
            else:
                weight = 0.7
            key = (n.track if n.channel != 9 else 1000 + n.pitch, bar, s)
            per_track[key] = max(per_track.get(key, 0.0), weight)
        for (_, bar, s), weight in per_track.items():
            grids[bar, s] += weight
        peak = grids.max(axis=1, keepdims=True)
        return np.divide(grids, peak, out=np.zeros_like(grids), where=peak > 0)

    def _compute_energy(self) -> np.ndarray:
        energy = np.zeros(self.bars)
        for n in self.harmony + self.drums:
            bar = int(n.start // self.bar_ticks)
            if bar < self.bars:
                energy[bar] += n.velocity
        return energy

    def _compute_beat_chords(self) -> List[Optional[Tuple[int, str]]]:
        beats = self.bars * 4
        tpb = self.song.ticks_per_beat
        hist = np.zeros((beats, 12))
        for n in self.harmony:
            first, last = int(n.start // tpb), int((n.end - 1) // tpb)
            for b in range(first, min(last, beats - 1) + 1):
                overlap = min(n.end, (b + 1) * tpb) - max(n.start, b * tpb)
                if overlap > 0:
                    hist[b, n.pitch % 12] += overlap * n.velocity
        for n in self.bass:
            b = int(n.start // tpb)
            if b < beats:
                hist[b, n.pitch % 12] += 1.5 * tpb * n.velocity
        chords: List[Optional[Tuple[int, str]]] = []
        for b in range(beats):
            norm = np.linalg.norm(hist[b])
            if norm == 0:
                chords.append(chords[-1] if chords else None)
                continue
            scores = _TEMPLATES @ (hist[b] / norm)
            best = int(np.argmax(scores))
            chords.append(_LABELS[best] if scores[best] >= 0.55 else (chords[-1] if chords else None))
        return chords

    def bar_records(self, key_scale: Tuple[int, ...]):
        """Gera (contexto, notas) de cada compasso aproveitável."""
        by_bar: Dict[int, List[MidiNote]] = defaultdict(list)
        for n in self.bass:
            by_bar[int(n.start // self.bar_ticks)].append(n)
        for bar in range(2, self.bars):
            notes = by_bar.get(bar)
            if not notes:
                continue
            slots, off_grid = [], 0
            for n in notes:
                pos = (n.start - bar * self.bar_ticks) / self.slot
                slot = int(round(pos))
                if abs(pos - slot) > 0.2:
                    off_grid += 1
                slots.append(min(15, max(0, slot)))
            if off_grid > 0.25 * len(notes):
                continue                  # swing/quiáltera: fora do vocabulário em semicolcheias
            beat_chords = self._chords[bar * 4:bar * 4 + 4]
            if any(c is None for c in beat_chords):
                continue
            previous_chord = self._chords[bar * 4 - 1]
            changes = 0
            for b, chord in enumerate(beat_chords):
                before = previous_chord if b == 0 else beat_chords[b - 1]
                if chord != before:
                    changes |= 1 << (b * 4)
            next_chord = self._chords[bar * 4 + 4] if bar * 4 + 4 < len(self._chords) else None
            roles = self._roles(notes, slots, beat_chords, next_chord, key_scale)
            if roles is None:
                continue
            durations = [max(1, min(16, int(round((n.end - n.start) / self.slot)))) for n in notes]
            peak_vel = max(n.velocity for n in notes) or 1
            history = self._energy[max(0, bar - 9):bar - 1]
            trend = (self._energy[bar - 1] / history.mean()) if history.size and history.mean() > 0 else 1.0
            context = {
                "prev_grid": (self._band_grids[bar - 1] + self._band_grids[bar - 2]) / 2.0,
                "cur_grid": self._band_grids[bar],
                "bpm": self.song.bpm_at(bar * self.bar_ticks),
                "trend": float(trend),
                "changes": changes,
            }
            hits = [(s, d, ROLE_ID[r], int(round(127 * n.velocity / peak_vel)))
                    for s, d, r, n in zip(slots, durations, roles, notes)]
            yield bar, context, hits

    @staticmethod
    def _roles(notes, slots, beat_chords, next_chord, scale) -> Optional[List[str]]:
        roles: List[str] = []
        base_pitch: Optional[int] = None
        for i, (n, slot) in enumerate(zip(notes, slots)):
            beat = slot // 4
            root, quality = beat_chords[beat]
            interval = (n.pitch - root) % 12
            third = 3 if quality == "min" else 4
            # Próxima troca de acorde depois desta nota (para aproximações).
            target = None
            for b in range(beat + 1, 4):
                if beat_chords[b] != beat_chords[beat]:
                    target = beat_chords[b]
                    break
            if target is None and next_chord is not None and next_chord != beat_chords[beat]:
                target = next_chord if beat == 3 else None
            if interval == 0:
                if base_pitch is not None and n.pitch >= base_pitch + 12:
                    roles.append("R8")
                else:
                    roles.append("R")
                    base_pitch = n.pitch if base_pitch is None else min(base_pitch, n.pitch)
                continue
            if interval == 7:
                roles.append("5")
                continue
            if interval == third:
                roles.append("3")
                continue
            if target is not None and slot % 4 != 0:
                distance = (target[0] - n.pitch % 12) % 12
                if distance in (1, 2, 10, 11):
                    roles.append("APP")
                    continue
            if interval in (10, 11):
                roles.append("7")
                continue
            if n.pitch % 12 not in scale:
                roles.append("CHR")
                continue
            # Grau da escala acima da fundamental (1..6) ou abaixo (oitava de baixo).
            root_pc = root
            steps_up = 0
            pc = root_pc
            while pc != n.pitch % 12 and steps_up < 7:
                pc = (pc + 1) % 12
                if pc in scale:
                    steps_up += 1
            if steps_up == 0 or steps_up > 6:
                return None
            roles.append(f"U+{steps_up}" if steps_up <= 3 or interval < 9 else f"U-{7 - steps_up}")
        return [r if r in ROLE_ID else "CHR" for r in roles]


def build(source_dir: str, out_path: str, style: str = "GOSPEL_WORSHIP",
          limit: int = 0) -> Dict[str, int]:
    artists = POOLS[style]
    files = []
    for artist in artists:
        folder = os.path.join(source_dir, artist)
        if os.path.isdir(folder):
            files += [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                      if f.lower().endswith(".mid")]
    contexts, notes_out, bar_meta, song_names = [], [], [], []
    stats = defaultdict(int)
    seen = {}
    if limit:
        files = files[::max(1, len(files) // limit)][:limit]
    for number, path in enumerate(files, start=1):
        if number % 250 == 0:
            print(f"  {number}/{len(files)} músicas, {stats['compassos']} compassos", file=sys.stderr,
                  flush=True)
        song = read_midi(path)
        if song is None or not _is_four_four(song):
            stats["descartada_formato"] += 1
            continue
        bass = _bass_notes(song)
        if len(bass) < 32:
            stats["sem_baixo"] += 1
            continue
        _, scale = _estimate_key([n for t in song.tracks for n in t.notes if n.channel != 9])
        try:
            bars = _SongBars(song, bass)
        except (ValueError, IndexError):
            stats["erro"] += 1
            continue
        song_id = len(song_names)
        song_names.append(os.path.relpath(path, source_dir))
        stats["musicas"] += 1
        for bar, context, hits in bars.bar_records(scale):
            prev_q = np.round(context["prev_grid"] * 4).astype(np.uint8)
            signature = (tuple(prev_q), context["changes"], tuple((h[0], h[1], h[2]) for h in hits),
                         int(context["bpm"] // 10))
            if signature in seen:
                bar_meta[seen[signature]][3] += 1        # mesmo padrão: conta repetições
                stats["repetidos"] += 1
                continue
            seen[signature] = len(bar_meta)
            bar_meta.append([song_id, bar, len(notes_out), 1])
            contexts.append((context["prev_grid"], context["cur_grid"], context["bpm"],
                             context["trend"], context["changes"]))
            notes_out.append(hits)
            stats["compassos"] += 1
    count = len(contexts)
    flat = [(i, *hit) for i, hits in enumerate(notes_out) for hit in hits]
    np.savez_compressed(
        out_path,
        roles=np.array(ROLES),
        song_names=np.array(song_names),
        song_id=np.array([m[0] for m in bar_meta], dtype=np.int32),
        bar_index=np.array([m[1] for m in bar_meta], dtype=np.int32),
        weight=np.array([m[3] for m in bar_meta], dtype=np.int32),
        prev_grid=np.round(np.array([c[0] for c in contexts]) * 255).astype(np.uint8).reshape(count, 16),
        cur_grid=np.round(np.array([c[1] for c in contexts]) * 255).astype(np.uint8).reshape(count, 16),
        bpm=np.array([c[2] for c in contexts], dtype=np.float32),
        trend=np.array([c[3] for c in contexts], dtype=np.float32),
        changes=np.array([c[4] for c in contexts], dtype=np.uint16),
        note_bar=np.array([f[0] for f in flat], dtype=np.int32),
        note_slot=np.array([f[1] for f in flat], dtype=np.uint8),
        note_dur=np.array([f[2] for f in flat], dtype=np.uint8),
        note_role=np.array([f[3] for f in flat], dtype=np.uint8),
        note_vel=np.array([f[4] for f in flat], dtype=np.uint8),
        source=np.array("Lakh MIDI Dataset (clean_midi), Colin Raffel, CC-BY 4.0, "
                        "https://colinraffel.com/projects/lmd/"),
    )
    return dict(stats)


if __name__ == "__main__":
    source, output = sys.argv[1], sys.argv[2]
    style_name = sys.argv[3] if len(sys.argv) > 3 else "GOSPEL_WORSHIP"
    max_songs = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    print(build(source, output, style_name, max_songs))
