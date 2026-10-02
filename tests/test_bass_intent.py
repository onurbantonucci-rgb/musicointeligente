"""Intenção musical do baixista: seção, groove, memória, confiança e modulação."""
import copy
import unittest

import numpy as np

from app.analysis.groove_feel import GrooveProfile
from app.instruments.bass_arranger import BassArranger
from app.instruments.bass_bank import load_bank
from app.instruments.bass_brain import BarRequest, BassBrain
from app.instruments.bass_intent import (APPROACH, DIRECT_CHANGE, BarContext, BassMusicalIntentEngine,
                                         PhraseMemory, REASON_GROOVE_TRANSITION, REASON_LOW_CONFIDENCE,
                                         REASON_SECTION_MEMORY)
from app.instruments.bass_styles import GOSPEL_WORSHIP
from tests.test_bass_arranger import make_timeline


def groove(energy=0.3, density=0.12, label="SPACIOUS", confidence=0.9, syncopation=0.05):
    return GrooveProfile(tempo_bpm=68.0, onset_density=density, energy=energy, label=label,
                         confidence=confidence, syncopation=syncopation,
                         subdivision="QUARTER" if density < 0.25 else "EIGHTH", bars_observed=8)


def context(bar, section="Verso", kind="VERSE", occurrence=1, progress=0.0, bars_left=8, g=None,
            chords=("G",) * 4, duration=8.0, position=0.95, beats_until_key=-1.0, next_key="--"):
    return BarContext(bar_number=bar, section_key=section, section_type=kind,
                      section_occurrence=occurrence, section_progress=progress,
                      bars_until_section_change=bars_left, chords=list(chords), next_chord="D",
                      chord_duration_beats=duration, current_key="G Major", next_key=next_key,
                      beats_until_key_change=beats_until_key, groove=g, position_confidence=position)


# Worship lento: intro, verso calmo com acordes de 2 compassos, pré-refrão, refrão.
WORSHIP_68 = ([("G", 8, .2, "INTRO")] +
              [("G", 8, .35, "VERSE"), ("D", 8, .35, "VERSE"), ("Em", 8, .35, "VERSE"),
               ("C", 8, .35, "VERSE")] +
              [("Am", 4, .5, "PRE_CHORUS"), ("C", 4, .55, "PRE_CHORUS"), ("D", 8, .65, "PRE_CHORUS")] +
              [("G", 4, .9, "CHORUS"), ("D", 4, .9, "CHORUS"), ("Em", 4, .9, "CHORUS"),
               ("C", 4, .9, "CHORUS")] * 2 + [("G", 8, .2, "OUTRO")])


def arrange(plan=WORSHIP_68, creativity=3, seed=0, bpm=68.0, **kwargs):
    timeline = make_timeline(plan, bpm=bpm, **kwargs)
    return timeline, BassArranger(GOSPEL_WORSHIP).arrange(timeline, creativity, seed)


def metrics(arrangement):
    return {row["section"]: row for row in arrangement.diagnostics.section_metrics()}


class TestWorshipRestraint(unittest.TestCase):
    def test_slow_worship_verse_has_few_long_notes_and_no_fills(self):     # TEST 1
        for creativity in (1, 5):
            with self.subTest(creativity=creativity):
                _, arrangement = arrange(creativity=creativity)
                verse = metrics(arrangement)["VERSE#1"]
                self.assertLessEqual(verse["notes_per_bar"], 2.0)
                self.assertEqual(verse["fills"], 0)
                self.assertGreaterEqual(verse["avg_note_beats"], 1.5)

    def test_advanced_is_not_more_notes(self):                          # TEST 2
        _, simple = arrange(creativity=1)
        _, advanced = arrange(creativity=5)
        for section in ("VERSE#1", "CHORUS#1"):
            s, a = metrics(simple)[section], metrics(advanced)[section]
            self.assertLessEqual(a["notes_per_bar"], s["notes_per_bar"] * 1.15 + 0.25, section)
        # ...mas usa vocabulário mais rico onde a música abre espaço.
        rich = [n for n in advanced.notes if n.role not in ("R", "R8", "PUSH")]
        plain = [n for n in simple.notes if n.role not in ("R", "R8", "PUSH")]
        self.assertGreater(len(rich), len(plain))

    def test_long_chord_is_held_not_repeated(self):                     # TEST 11
        timeline, arrangement = arrange(creativity=5)
        verse_notes = [n for n in arrangement.notes if timeline.chords[n.beat_index] and
                       2 <= n.bar <= 9]
        bars = {n.bar for n in verse_notes}
        self.assertLessEqual(len(verse_notes), len(bars) * 2)
        holds = [p for p in arrangement.bars if 2 <= p.bar <= 9 and p.intent["sustain_preferred"]]
        self.assertTrue(holds)

    def test_chorus_is_more_present_than_verse(self):
        _, arrangement = arrange(creativity=3)
        m = metrics(arrangement)
        self.assertGreater(m["CHORUS#1"]["notes_per_bar"], m["VERSE#1"]["notes_per_bar"])
        chorus = [n.velocity for n in arrangement.notes if 18 <= n.bar <= 21]
        verse = [n.velocity for n in arrangement.notes if 2 <= n.bar <= 9]
        self.assertGreater(np.mean(chorus), np.mean(verse))

    def test_fills_only_at_transitions(self):                           # TEST 12
        for seed in range(3):
            _, arrangement = arrange(creativity=5, seed=seed)
            for plan in arrangement.bars:
                if plan.fill:
                    self.assertTrue(plan.intent["transition_fill_allowed"], plan)
            middle_of_verse = [p for p in arrangement.bars if 3 <= p.bar <= 8]
            self.assertFalse(any(p.intent["transition_fill_allowed"] for p in middle_of_verse))


class TestSectionIdentity(unittest.TestCase):
    def engine(self, creativity=3):
        return BassMusicalIntentEngine("GOSPEL_WORSHIP", creativity)

    def test_isolated_spike_does_not_change_the_verse(self):            # TEST 3
        engine = self.engine()
        intents = []
        for bar in range(8):
            g = groove(energy=0.85, density=0.6, label="BUSY") if bar == 3 else groove()
            intents.append(engine.plan_bar(context(bar + 1, progress=bar / 8, bars_left=8 - bar, g=g)))
        families = {i.groove_label for i in intents}
        self.assertEqual(families, {"SPACIOUS"})
        caps = [i.attacks_cap for i in intents[2:]]
        self.assertLess(max(caps) - min(caps), 1.0)
        self.assertNotIn(REASON_GROOVE_TRANSITION, intents[3].reasons)

    def test_persistent_change_is_followed_gradually(self):            # TEST 4
        engine = self.engine()
        intents = []
        for bar in range(10):
            g = groove(energy=0.8, density=0.55, label="DRIVING") if bar >= 4 else groove()
            intents.append(engine.plan_bar(context(bar + 1, progress=bar / 10, bars_left=10 - bar, g=g)))
        self.assertEqual(intents[4].groove_label, "SPACIOUS")      # 1 compasso não basta
        transition = next(i for i, x in enumerate(intents) if REASON_GROOVE_TRANSITION in x.reasons)
        self.assertGreaterEqual(transition, 5)
        self.assertEqual(intents[-1].groove_label, "DRIVING")
        self.assertGreater(intents[-1].attacks_cap, intents[3].attacks_cap)
        jumps = np.diff([i.attacks_cap for i in intents])
        self.assertLess(max(jumps), 2.5)                            # transição gradual

    def test_pre_chorus_builds(self):                                   # TEST 5
        engine = self.engine()
        flat = groove(energy=0.45, density=0.2, label="STEADY")
        energies = []
        for bar in range(4):
            engine.plan_bar(context(bar + 1, progress=bar / 4, bars_left=4 - bar, g=flat))
        for bar in range(4):
            intent = engine.plan_bar(context(5 + bar, "Pré-refrão", "PRE_CHORUS",
                                             progress=bar / 4, bars_left=4 - bar, g=flat))
            energies.append(intent.energy_target)
        self.assertTrue(all(b >= a for a, b in zip(energies, energies[1:])))
        self.assertGreater(energies[-1], energies[0] + 0.05)

    def test_section_end_can_prepare_the_chorus(self):                  # TEST 13
        engine = self.engine(creativity=4)
        g = groove(energy=0.55, density=0.3, label="STEADY")
        for bar in range(3):
            engine.plan_bar(context(bar + 1, "Pré-refrão", "PRE_CHORUS", progress=bar / 4,
                                    bars_left=4 - bar, g=g))
        last = engine.plan_bar(context(4, "Pré-refrão", "PRE_CHORUS", progress=0.9, bars_left=1, g=g))
        self.assertTrue(last.transition_fill_allowed)
        self.assertGreater(last.fill_probability, 0)

    def test_chorus_and_verse_memory(self):                             # TESTS 6, 7
        engine = self.engine()
        chorus_groove = groove(energy=0.8, density=0.45, label="DRIVING")
        for bar in range(4):
            engine.plan_bar(context(bar + 1, g=groove()))
        first_chorus = [engine.plan_bar(context(5 + b, "Refrão", "CHORUS", progress=b / 4,
                                                bars_left=4 - b, g=chorus_groove)) for b in range(4)]
        verse2 = engine.plan_bar(context(9, occurrence=2, g=chorus_groove))     # banda ainda forte
        self.assertIn(REASON_SECTION_MEMORY, verse2.reasons)
        self.assertEqual(verse2.groove_label, "SPACIOUS")                       # lembra o verso 1
        for b in range(1, 4):
            engine.plan_bar(context(9 + b, occurrence=2, progress=b / 4, bars_left=4 - b, g=groove()))
        chorus2 = engine.plan_bar(context(13, "Refrão", "CHORUS", occurrence=2, g=groove()))
        self.assertIn(REASON_SECTION_MEMORY, chorus2.reasons)
        self.assertEqual(chorus2.groove_label, first_chorus[-1].groove_label)
        self.assertAlmostEqual(chorus2.attacks_cap, first_chorus[-1].attacks_cap, delta=1.5)

    def test_low_confidence_reduces_complexity(self):                   # TEST 14
        engine = self.engine(creativity=5)
        for bar in range(3):
            engine.plan_bar(context(bar + 1, "Refrão", "CHORUS", g=groove(0.9, 0.5, "DRIVING")))
        unsure = engine.plan_bar(context(4, "Refrão", "CHORUS", g=groove(0.9, 0.5, "DRIVING"),
                                         position=0.3))
        self.assertEqual(unsure.fallback_pattern, "SUSTAINED")
        self.assertEqual(unsure.vocabulary_level, 1)
        self.assertLessEqual(unsure.attacks_cap, 2.0)
        self.assertFalse(unsure.transition_fill_allowed)
        self.assertIn(REASON_LOW_CONFIDENCE, unsure.reasons)


class TestPhraseMemory(unittest.TestCase):
    def test_similar_rhythms_are_preferred_inside_a_phrase(self):      # TEST 8
        memory = PhraseMemory()
        memory.begin_section("Verso", 4)
        motif = np.zeros(16)
        motif[[0, 8]] = 1
        memory.record(motif)
        other = np.zeros(16)
        other[::2] = 1
        bonus = memory.bonus(np.array([motif, other]), variation_budget=0.2)
        self.assertGreater(bonus[0], bonus[1])

    def test_motif_persists_in_the_arrangement(self):
        _, arrangement = arrange(creativity=4)
        chorus = metrics(arrangement)["CHORUS#1"]
        self.assertLessEqual(chorus["phrase_changes"], chorus["bars"] // 2)


class TestModulation(unittest.TestCase):
    def test_slow_worship_changes_key_directly(self):                   # TEST 21
        engine = BassMusicalIntentEngine("GOSPEL_WORSHIP", 5)
        for bar in range(3):
            engine.plan_bar(context(bar + 1, g=groove()))
        intent = engine.plan_bar(context(4, g=groove(), beats_until_key=4.0, next_key="A Major"))
        self.assertEqual(intent.modulation_action, DIRECT_CHANGE)
        self.assertFalse(intent.transition_fill_allowed)

    def test_energetic_section_can_approach_the_new_key(self):          # TEST 22
        engine = BassMusicalIntentEngine("GOSPEL_WORSHIP", 4)
        loud = groove(energy=0.9, density=0.5, label="DRIVING")
        for bar in range(3):
            engine.plan_bar(context(bar + 1, "Refrão", "CHORUS", g=loud))
        intent = engine.plan_bar(context(4, "Refrão", "CHORUS", g=loud, beats_until_key=4.0,
                                         next_key="A Major"))
        self.assertEqual(intent.modulation_action, APPROACH)

    def modulated_timeline(self):
        plan = [("G", 4, .9, "CHORUS"), ("D", 4, .9, "CHORUS"), ("Em", 4, .9, "CHORUS"),
                ("C", 4, .9, "CHORUS")] * 2 + [("A", 4, .9, "CHORUS"), ("E", 4, .9, "CHORUS"),
                                               ("F#m", 4, .9, "CHORUS"), ("D", 4, .9, "CHORUS")] * 2
        timeline = make_timeline(plan, bpm=84.0)
        timeline.event_keys = ["G Major"] * 8 + ["A Major"] * 8
        return timeline

    def test_groove_survives_the_key_change_and_notes_follow_the_new_key(self):   # TEST 23
        timeline = self.modulated_timeline()
        arrangement = BassArranger(GOSPEL_WORSHIP).arrange(timeline, 4, 0)
        change = 8 * 4
        before = [p for p in arrangement.bars if p.start_beat < change][-2:]
        after = [p for p in arrangement.bars if p.start_beat >= change][:2]
        self.assertEqual({p.intent["groove_label"] for p in before},
                         {p.intent["groove_label"] for p in after})
        self.assertEqual(after[0].intent["current_key"], "A Major")
        a_major = {9, 11, 1, 2, 4, 6, 8}
        for note in arrangement.notes:
            if note.beat_index >= change and note.role not in ("R", "R8", "5", "3", "PUSH"):
                self.assertIn(note.midi % 12, a_major, note)
        last_before = [p for p in arrangement.bars if p.start_beat == change - 4][0]
        self.assertIn(last_before.intent["modulation_action"], ("APPROACH", "PREPARE", "DIRECT_CHANGE"))
        approaches = [n for n in arrangement.notes if n.reason.startswith("Aproximação ao novo tom")]
        for note in approaches:
            self.assertIn(note.midi % 12, a_major)


class TestBankFallback(unittest.TestCase):
    def test_no_compatible_bank_phrase_falls_back_to_a_safe_pattern(self):   # TEST 30
        bank = load_bank("GOSPEL_WORSHIP")
        if bank is None:
            self.skipTest("banco não instalado")
        crowded = copy.copy(bank)
        crowded.notes_per_bar = np.full(len(bank), 16.0)       # todo compasso do banco "lotado"
        engine = BassMusicalIntentEngine("GOSPEL_WORSHIP", 5)
        for bar in range(3):
            engine.plan_bar(context(bar + 1, g=groove()))
        intent = engine.plan_bar(context(4, g=groove(), duration=4.0, chords=("G", "G", "D", "D")))
        brain = BassBrain(crowded, 5, 0)
        band = np.zeros(16)
        band[[0, 8]] = 1.0
        beat_times = [i * 0.88 for i in range(5)]
        decision = brain.plan_bar(BarRequest(bar_number=4, beat_times=beat_times,
                                             chords=["G", "G", "D", "D"], next_chord="Em",
                                             band_history=[band, band], energy_history=[0.3] * 4,
                                             intent=intent))
        self.assertTrue(decision.decision.startswith("Fallback"), decision.decision)
        self.assertLessEqual(len(decision.notes), 3)
        self.assertEqual({n.role for n in decision.notes} - {"R", "5"}, set())


if __name__ == "__main__":
    unittest.main()
