"""Arranjo criativo do baixo por estilo (modo ESTILO MUSICAL)."""
import unittest

import numpy as np

from app.analysis.chart_harmony_aligner import (BeatChroma, HarmonyTimeline, align_chart_to_beats,
                                                 chord_template)
from app.instruments.bass_arranger import BassArranger
from app.instruments.bass_model import BassHarmonySource, BassPatternType
from app.instruments.bass_player import BassPlayer
from app.instruments.bass_styles import GOSPEL_WORSHIP
from app.music.chord_chart import parse_chord
from app.music.musical_context import MusicalContext
from app.music.theory import note_to_pc

FILL_ROLES = {"S-3", "S-2", "S-1", "S+3", "S+2", "S+1", "C-1", "C+1"}


def make_timeline(plan, bpm=90.0, push_into=(), hold_tail=False, start=0.0):
    """plan: [(acorde, batidas, energia, seção)]. ``push_into``: índices (no plano) de
    acordes cuja entrada a banda antecipa no fim da batida anterior."""
    chords, energy, events, sections, starts, onsets = [], [], [], [], [], []
    previous_section = None
    for index, (chord, beats, level, section) in enumerate(plan):
        if section != previous_section:
            starts.append(index)
            previous_section = section
        sections.append(section)
        for beat in range(beats):
            chords.append(chord)
            energy.append(level)
            events.append(index)
            # A banda acompanha a intensidade: mínimas no calmo, semínimas no médio,
            # colcheias no cheio.
            if level < 0.3:
                onsets.append([1.0 if beat % 2 == 0 else 0.05, 0.05, 0.05, 0.05])
            elif level < 0.7:
                onsets.append([1.0, 0.05, 0.1, 0.05])
            else:
                onsets.append([1.0, 0.1, 0.7, 0.1])
    count = len(chords)
    period = 60.0 / bpm
    onsets = np.asarray(onsets)
    tails = np.array([chord_template(chord) for chord in chords])
    change_beats = np.cumsum([beats for _, beats, _, _ in plan])[:-1]
    for index, beat in enumerate(change_beats, start=1):
        if index in push_into:
            tails[beat - 1] = chord_template(plan[index][0])
        elif hold_tail:
            tails[beat - 1] = chord_template(plan[index - 1][0])
    return HarmonyTimeline(
        beat_times=start + period * np.arange(count + 1), chords=chords, event_indices=events,
        beats_per_bar=4, downbeat_phase=0, confidence=0.95, energy=np.asarray(energy, float),
        onsets=onsets, tail_chroma=tails, key="G Major", event_sections=sections,
        section_start_events=[s for s in starts if s > 0])


def bass_pc(chord):
    symbol = parse_chord(chord)
    return note_to_pc(symbol.bass_note or symbol.root)


SONG = ([("G", 8, .2, "INTRO"), ("D", 8, .2, "INTRO")] +
        [("G", 8, .5, "VERSE"), ("D", 8, .5, "VERSE"), ("Em", 4, .5, "VERSE"),
         ("D", 4, .5, "VERSE"), ("C", 8, .5, "VERSE")] +
        [("C", 4, 1.0, "CHORUS"), ("D", 4, 1.0, "CHORUS"), ("Em", 4, 1.0, "CHORUS"),
         ("G/B", 4, 1.0, "CHORUS")] * 3 + [("G", 8, .2, "OUTRO")])


class TestBassArranger(unittest.TestCase):
    def arrange(self, timeline=None, creativity=3, seed=0):
        return BassArranger(GOSPEL_WORSHIP).arrange(timeline or make_timeline(SONG), creativity, seed)

    def test_supporting_notes_are_the_chart_bass_including_minor_and_inversions(self):
        for creativity in (1, 3, 5):
            with self.subTest(creativity=creativity):
                arrangement = self.arrange(creativity=creativity)
                for note in arrangement.notes:
                    if note.role in ("R", "R8"):
                        self.assertEqual(note.midi % 12, bass_pc(note.chord), note)
                    elif note.role == "PUSH" and note.reason.startswith("Antecipação"):
                        self.assertEqual(note.midi % 12, bass_pc(note.chord), note)
                self.assertTrue(any(n.chord == "Em" and n.role == "R" and n.midi % 12 == 4
                                    for n in arrangement.notes))
                self.assertTrue(any(n.chord == "G/B" and n.role == "R" and n.midi % 12 == 11
                                    for n in arrangement.notes))

    def test_notes_sit_on_the_audio_grid_and_in_bass_range(self):
        timeline = make_timeline(SONG, bpm=84.0, start=0.37)
        period = 60.0 / 84.0
        for note in self.arrange(timeline, creativity=5).notes:
            expected = timeline.beat_times[note.beat_index] + (note.slot % 4) * period / 4
            self.assertAlmostEqual(note.time, expected, places=6)
            self.assertTrue(28 <= note.midi <= 55)

    def test_simple_creativity_has_no_ornaments(self):
        roles = {note.role for note in self.arrange(creativity=1).notes}
        self.assertTrue(roles <= {"R", "5"}, roles)

    def test_intensity_drives_density(self):
        # A banda (e não a criatividade) decide a densidade: refrão cheio > intro calma.
        arrangement = self.arrange(creativity=3)
        intro = [n for n in arrangement.notes if n.bar <= 4]
        chorus = [n for n in arrangement.notes if 15 <= n.bar <= 24]
        self.assertLess(len(intro) / 4, len(chorus) / 10)

    def test_variation_is_reproducible_and_seed_changes_it(self):
        first = [(n.time, n.midi) for n in self.arrange(creativity=4, seed=1).notes]
        again = [(n.time, n.midi) for n in self.arrange(creativity=4, seed=1).notes]
        other = [(n.time, n.midi) for n in self.arrange(creativity=4, seed=2).notes]
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_long_chorus_is_not_the_same_bar_over_and_over(self):
        arrangement = self.arrange(creativity=3)
        rhythms = {}
        for note in arrangement.notes:
            if 13 <= note.bar <= 24:
                rhythms.setdefault(note.bar, []).append((note.slot, note.midi))
        self.assertGreater(len({tuple(v) for v in rhythms.values()}), 1)

    def test_fast_tempo_avoids_sixteenth_notes(self):
        arrangement = self.arrange(make_timeline(SONG, bpm=172.0), creativity=5)
        self.assertFalse([n for n in arrangement.notes if n.slot % 2])

    def test_anticipates_only_when_the_band_anticipates(self):
        pushed = make_timeline(SONG, push_into=range(1, len(SONG)))
        notes = self.arrange(pushed, creativity=3).notes
        anticipations = [n for n in notes if n.reason.startswith("Antecipação")]
        self.assertGreaterEqual(len(anticipations), 10)
        for push in anticipations:
            # Ligada: nenhum ataque separado no tempo seguinte.
            self.assertFalse([n for n in notes if n.beat_index == push.beat_index + 1 and n.slot % 4 == 0])
        held = make_timeline(SONG, hold_tail=True)
        for seed in range(3):
            notes = self.arrange(held, creativity=3, seed=seed).notes
            self.assertLessEqual(len([n for n in notes if n.reason.startswith("Antecipação")]), 2)

    def test_creativity_changes_vocabulary_not_density(self):
        simple = self.arrange(creativity=1).notes
        rich = self.arrange(creativity=5).notes
        self.assertEqual({n.role for n in simple}, {"R"})
        melodic = [n for n in rich if n.role not in ("R", "R8", "PUSH")]
        self.assertGreater(len(melodic), 0.2 * len(rich))
        # Mais criatividade não é mais notas.
        self.assertLess(len(rich), 1.25 * len(simple))

    def test_song_ends_on_a_long_note(self):
        arrangement = self.arrange(creativity=5)
        last_bar = max(n.bar for n in arrangement.notes)
        final = [n for n in arrangement.notes if n.bar == last_bar]
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0].midi % 12, bass_pc("G"))


class TestBassFollowsBandAndKey(unittest.TestCase):
    G_MAJOR = {7, 9, 11, 0, 2, 4, 6}

    def arrange(self, timeline, creativity, seed=0):
        return BassArranger(GOSPEL_WORSHIP).arrange(timeline, creativity, seed)

    def test_bass_attacks_only_where_the_band_attacks(self):
        # Ao vivo o baixista só conhece a levada dos compassos que passaram:
        # admite-se 1 compasso de reação quando a banda muda sem a cifra avisar.
        timeline = make_timeline(SONG)
        for creativity in (1, 3, 5):
            with self.subTest(creativity=creativity):
                for seed in range(3):
                    notes = self.arrange(timeline, creativity, seed).notes
                    locked = [timeline.onsets[n.beat_index, n.slot % 4] >= 0.3 for n in notes]
                    self.assertGreaterEqual(sum(locked) / len(locked), 0.95)

    def test_decisions_only_use_the_past(self):
        # Mudar o que a banda toca DEPOIS do compasso 10 não muda nada até o 10.
        base = make_timeline(SONG)
        changed = make_timeline(SONG)
        cut = int(changed.beat_count * 0.6)
        changed.onsets[cut:] = np.tile([1.0, 0.9, 0.9, 0.9], (changed.beat_count - cut, 1))
        changed.energy[cut:] = 3.0
        first = [(n.time, n.midi) for n in self.arrange(base, 4, 1).notes
                 if n.beat_index < cut - 4]
        second = [(n.time, n.midi) for n in self.arrange(changed, 4, 1).notes
                  if n.beat_index < cut - 4]
        self.assertEqual(first, second)

    def test_real_bass_lines_come_from_the_bank(self):
        arrangement = self.arrange(make_timeline(SONG), 3)
        sources = {plan.source for plan in arrangement.bars if plan.source}
        self.assertGreater(len(sources), 2)

    def test_passing_notes_stay_in_the_key_of_the_chart(self):
        timeline = make_timeline(SONG)
        for creativity in (3, 4, 5):
            for seed in range(4):
                for note in self.arrange(timeline, creativity, seed).notes:
                    tones = {i for i, v in enumerate(chord_template(note.chord)) if v > 0}
                    self.assertIn(note.midi % 12, self.G_MAJOR | tones, note)

    def test_wrong_chart_key_falls_back_to_the_key_of_the_chords(self):
        timeline = make_timeline(SONG)
        timeline.key = "C# Major"           # tom que não explica G D Em C
        arrangement = self.arrange(timeline, 5)
        self.assertEqual(arrangement.key_source, "acordes")
        self.assertEqual(arrangement.key_label, "G Major")
        for note in arrangement.notes:
            tones = {i for i, v in enumerate(chord_template(note.chord)) if v > 0}
            self.assertIn(note.midi % 12, self.G_MAJOR | tones, note)

    def test_chart_key_is_used_when_it_explains_the_chords(self):
        arrangement = self.arrange(make_timeline(SONG), 3)
        self.assertEqual((arrangement.key_source, arrangement.key_label), ("cifra", "G Major"))


class TestSoundingKeyWithCapo(unittest.TestCase):
    """Cifra Club: "Tom:" é o tom SOANTE; os acordes são as formas com capotraste."""

    def align(self, sounding_sequence):
        chroma = np.array([chord_template(c) for c in sounding_sequence])
        beat_chroma = BeatChroma(beat_times=0.5 * np.arange(len(sounding_sequence) + 1),
                                 chroma=chroma, energy=np.full(len(sounding_sequence), 0.5),
                                 duration=0.5 * len(sounding_sequence))
        # Formas em F# (B9 C#4 D#m F#) com capo 2 -> soa G# (Tom: G#)
        return align_chart_to_beats(beat_chroma, ["B9", "C#4", "D#m", "F#"], 4,
                                    capo_semitones=2, key="G# Major")

    def test_audio_with_capo_keeps_the_chart_key(self):
        timeline = self.align(["C#9"] * 8 + ["D#sus4"] * 8 + ["Fm"] * 8 + ["G#"] * 8)
        self.assertEqual(timeline.transposition, 2)
        self.assertEqual(timeline.key, "G# Major")

    def test_audio_in_written_shapes_uses_the_shape_key(self):
        timeline = self.align(["B9"] * 8 + ["C#sus4"] * 8 + ["D#m"] * 8 + ["F#"] * 8)
        self.assertEqual(timeline.transposition, 0)
        self.assertEqual(timeline.key, "F# Major")


class TestHalfBeatSlipRepair(unittest.TestCase):
    def test_tracker_that_slipped_to_offbeats_is_moved_back(self):
        from app.analysis.tempo_detector import OnsetTempoDetector
        period = 0.6
        truth = 0.3 + period * np.arange(60)
        # 25 batidas no contratempo, escorregão gradual (2 intervalos curtos), depois no tempo
        slipped = np.concatenate([truth[:25] + period / 2,
                                  [truth[25] + 0.2 * period, truth[26]],
                                  truth[27:]])
        repaired = OnsetTempoDetector._repair_half_beat_slips(slipped, truth.copy())
        errors = [np.min(np.abs(truth - b)) for b in repaired]
        self.assertLess(max(errors), 0.15 * period)
        self.assertGreaterEqual(len(repaired), 58)

    def test_gradual_tempo_change_is_left_alone(self):
        from app.analysis.tempo_detector import OnsetTempoDetector
        intervals = np.linspace(0.8, 0.72, 80)        # 75 -> 83 BPM
        beats = np.concatenate([[0.3], 0.3 + np.cumsum(intervals)])
        rigid = 0.3 + 0.76 * np.arange(81)             # referência sem a deriva
        np.testing.assert_allclose(
            OnsetTempoDetector._repair_half_beat_slips(beats, rigid), beats)


class TestBassPlayerStyleMode(unittest.TestCase):
    def context(self, timestamp):
        context = MusicalContext(timestamp=timestamp, chord="G", chord_confidence=.9, bar=1, beat=1,
                                 beat_position=0.0, bpm=90, meter="4/4",
                                 tempo_tracking_state="TRACKING")
        context.chart_available = True
        context.chart_published_chord = "G"
        return context

    def test_schedules_arrangement_notes_and_new_variation_regenerates(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.STYLE,
                          harmony_source=BassHarmonySource.CHART)
        bass.creativity = 4
        bass.set_harmony_timeline(make_timeline(SONG))
        bass.on_musical_context(self.context(0.0))
        arrangement = bass.arrangement
        now = next(n.time for n in arrangement.notes if n.time >= 30.0) - 0.2
        bass.on_musical_context(self.context(now))
        scheduled = [e for e in bass.synthesizer.scheduled_events if e.scheduled_time >= now - 0.001]
        self.assertTrue(scheduled)
        expected = [n for n in arrangement.notes if now - 0.001 <= n.time < now + 0.45]
        self.assertEqual([round(e.scheduled_time, 6) for e in scheduled],
                         [round(n.time, 6) for n in expected])
        self.assertTrue(bass.synthesizer.strict_mono)
        self.assertTrue(bass.style_status(now))
        bass.new_variation()
        bass.on_musical_context(self.context(now))
        self.assertIsNot(bass.arrangement, arrangement)

    def test_without_analysis_style_mode_still_plays(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.STYLE,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(0.25)
        context.clock_bar, context.clock_beat = 1, 4
        context.beat_position = 0.5
        event = bass.on_musical_context(context)
        # Sem a música analisada toca como AUTO guiado pela intenção (no começo,
        # prudente: tempo 1 e 3), sempre o acorde da cifra.
        self.assertIsNotNone(event)
        self.assertIn(event.note[0], ("G", "D"))


class TestSectionBoundaryFromEnergy(unittest.TestCase):
    def test_section_starting_on_same_chord_enters_where_the_band_opens_up(self):
        # Verso termina em C (fraco) e o refrão começa em C (forte): a fronteira só
        # é visível pela energia.
        sequence = ["G"] * 8 + ["C"] * 12 + ["D"] * 4
        energy = np.array([0.2] * 16 + [1.0] * 8)
        chroma = np.array([chord_template(c) for c in sequence])
        beat_chroma = BeatChroma(beat_times=0.5 * np.arange(len(sequence) + 1), chroma=chroma,
                                 energy=energy, duration=0.5 * len(sequence))
        timeline = align_chart_to_beats(beat_chroma, ["G", "C", "C", "D"], 4, section_starts=[2])
        self.assertEqual(timeline.event_indices[15], 1)
        self.assertEqual(timeline.event_indices[16], 2)


if __name__ == "__main__":
    unittest.main()
