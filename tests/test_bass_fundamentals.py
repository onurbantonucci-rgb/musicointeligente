"""Tônica única e divisões musicais do novo padrão do baixista."""
import unittest
import tempfile
import tkinter as tk
import numpy as np

from app.analysis.audio_analyzer import AudioAnalyzer
from app.instruments.bass_decision import BassDecisionEngine
from app.instruments.bass_model import BassNoteValue, BassPatternType, BassHarmonySource
from app.instruments.bass_pattern import BassPatternGenerator
from app.instruments.bass_player import BassPlayer
from app.music.musical_context import MusicalContext
from app.song.song import Song
from app.song.song_session import SongSession
from app.ui.main_window import MainWindow


class TestBassFundamentals(unittest.TestCase):
    def context(self, timestamp=.75, beat=2, phase=.5, chord="C", bpm=120):
        return MusicalContext(timestamp=timestamp, chord=chord, chord_confidence=.9,
                              bar=1, beat=beat, beat_position=phase, bpm=bpm,
                              meter="4/4", tempo_tracking_state="TRACKING",
                              follow_confidence_level="MEDIUM", position_confidence=.9)

    def test_only_tonic_even_for_inverted_chord(self):
        context = self.context(chord="G/B")
        context.chord_root = "G"
        context.bass_note = "B"
        context.inversion = "slash"
        decision = BassDecisionEngine().decide(
            context, pattern_override=BassPatternType.FUNDAMENTALS)
        self.assertTrue(decision.root_note.startswith("G"))
        steps = BassPatternGenerator().generate_pattern(
            decision, note_value=BassNoteValue.EIGHTH)
        self.assertEqual(len(steps), 8)
        self.assertTrue(all(note.startswith("G") for _, note, _, _ in steps))

    def test_note_values_define_attack_counts_in_four_four(self):
        decision = BassDecisionEngine().decide(
            self.context(), pattern_override=BassPatternType.FUNDAMENTALS)
        for value, count in ((BassNoteValue.WHOLE, 1), (BassNoteValue.HALF, 2),
                             (BassNoteValue.QUARTER, 4), (BassNoteValue.EIGHTH, 8),
                             (BassNoteValue.SIXTEENTH, 16)):
            with self.subTest(value=value):
                steps = BassPatternGenerator().generate_pattern(decision, note_value=value)
                self.assertEqual(len(steps), count)
                self.assertTrue(all(midi == decision.root_midi for _, _, midi, _ in steps))

    def test_half_note_is_scheduled_on_beat_three_at_120_bpm(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF)
        event = bass.on_musical_context(self.context())
        self.assertEqual((event.bar, event.beat), (1, 3))
        self.assertAlmostEqual(event.start_time, 1.0)
        self.assertAlmostEqual(event.duration, .85)
        self.assertEqual(len(bass.synthesizer.scheduled_events), 1)

    def test_eighth_and_sixteenth_follow_bpm_and_phase(self):
        for value, expected in ((BassNoteValue.EIGHTH, .75),
                                (BassNoteValue.SIXTEENTH, .625)):
            with self.subTest(value=value):
                bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                                  note_value=value)
                event = bass.on_musical_context(self.context(timestamp=.55, beat=2, phase=.1))
                self.assertAlmostEqual(event.start_time, expected)
                self.assertEqual(event.note[0], "C")

    def test_uses_clock_bar_while_chart_waits_for_audio(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF)
        context = self.context(timestamp=5.75, beat=1, phase=.5)
        context.chart_available = True
        context.clock_bar = 3
        context.clock_beat = 4
        event = bass.on_musical_context(context)
        self.assertEqual((event.bar, event.beat), (4, 1))
        self.assertEqual(event.note[0], "C")

    def test_tempo_change_retimes_same_note_without_duplicate_attack(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.QUARTER)
        context = self.context(timestamp=1.75, beat=4)
        first = bass.on_musical_context(context)
        self.assertAlmostEqual(first.start_time, 2.0)
        context.timestamp = 1.78
        context.bpm = 100
        updated = bass.on_musical_context(context)
        self.assertGreater(updated.start_time, first.start_time)
        self.assertEqual(len(bass.synthesizer.scheduled_events), 1)
        self.assertEqual(bass.synthesizer.scheduled_events[0].midi_note,
                         updated.midi_note)

    def test_phase_correction_does_not_retrigger_beat_already_in_audio(self):
        for pattern in (BassPatternType.FUNDAMENTALS, BassPatternType.ROOT):
            with self.subTest(pattern=pattern):
                bass = BassPlayer(sample_rate=1000, pattern=pattern,
                                  note_value=BassNoteValue.QUARTER)
                context = self.context(timestamp=.75, beat=2, phase=.5)
                first = bass.on_musical_context(context)
                self.assertEqual((first.bar, first.beat), (1, 3))
                bass.synthesizer.render_chunk(300, 1000, .75)
                self.assertTrue(bass.synthesizer.was_rendered((1, 3)))
                context.timestamp = 1.03
                context.beat_position = .7
                self.assertIsNone(bass.on_musical_context(context))
                self.assertEqual(bass.synthesizer.scheduled_events, [])

    def test_safe_chart_lookahead_uses_reference_chord(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS)
        context = self.context(timestamp=1.75, beat=4)
        context.chart_available = True
        context.clock_bar = 1
        context.clock_beat = 4
        context.next_expected_chord = "Am"
        context.next_change_bar = 2
        context.audio_activity = .1
        context.smoothed_detected_chord = "C"
        context.stable_chord_confidence = .9
        context.harmonic_event_root = "C"
        context.follow_confidence_level = "HIGH"
        context.phase_confidence = .9
        self.assertTrue(bass.on_musical_context(context).note.startswith("A"))
        bass.reset()
        context.harmonic_event_root = "G"
        context.harmonic_event_confidence = .9
        context.current_chord_elapsed_beats = 1.0
        self.assertIsNone(bass.on_musical_context(context))
        self.assertEqual(bass.synthesizer.scheduled_events, [])
        context.confirmed_variation_chord = "G"
        self.assertTrue(bass.on_musical_context(context).note.startswith("G"))

    def test_stable_audio_can_confirm_next_chord_or_silence_wrong_chart_root(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS)
        context = self.context(timestamp=1.25, beat=3)
        context.chart_available = True
        context.clock_bar = 1
        context.clock_beat = 3
        context.next_expected_chord = "G"
        context.audio_activity = .1
        context.stable_chord_confidence = .9
        context.stable_chord_duration = .4
        context.smoothed_detected_chord = "F#"
        self.assertIsNone(bass.on_musical_context(context))
        context.smoothed_detected_chord = "G"
        event = bass.on_musical_context(context)
        self.assertTrue(event.note.startswith("G"))
        self.assertEqual(bass.synthesizer.scheduled_events[0].source,
                         "audio-chart-confirmed")

    def test_uncertain_reference_cancels_prepared_subdivision(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.EIGHTH)
        context = self.context(timestamp=.55, beat=2, phase=.1)
        context.chart_available = True
        context.clock_bar = 1
        context.clock_beat = 2
        context.follow_confidence_level = "LOW"
        context.audio_activity = .1
        context.harmonic_event_root = "C"
        context.harmonic_event_confidence = .8
        self.assertIsNotNone(bass.on_musical_context(context))
        self.assertEqual(len(bass.synthesizer.scheduled_events), 1)
        context.harmonic_event_confidence = .2
        self.assertIsNone(bass.on_musical_context(context))
        self.assertEqual(bass.synthesizer.scheduled_events, [])

    def test_chart_only_uses_playalong_position_not_clock_or_detected_chord(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=.25, beat=1, phase=.5, chord="C")
        context.chart_available = True
        context.chart_clock_chord = "G"
        context.chart_next_clock_chord = "Am"
        context.clock_bar = 1
        context.clock_beat = 1
        context.smoothed_detected_chord = "F#"
        context.stable_chord_confidence = .95
        context.stable_chord_duration = .5
        context.confirmed_variation_chord = "F#"
        context.audio_activity = .1
        context.follow_confidence_level = "LOW"
        context.tracking_state = "LOST"
        event = bass.on_musical_context(context)
        self.assertTrue(event.note.startswith("C"))
        self.assertEqual(bass.synthesizer.scheduled_events[0].source, "chart-only")

    def test_chart_only_holds_playalong_chord_when_clock_enters_next_bar(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=1.75, beat=4, phase=.5, chord="C")
        context.chart_available = True
        context.chart_clock_chord = "C"
        context.chart_next_clock_chord = "Am"
        context.clock_bar = 1
        context.clock_beat = 4
        context.smoothed_detected_chord = "G"
        context.audio_activity = .1
        event = bass.on_musical_context(context)
        self.assertEqual((event.bar, event.beat), (2, 1))
        self.assertTrue(event.note.startswith("C"))

    def test_chart_only_replaces_pending_note_when_playalong_position_changes(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=.75, beat=2, phase=.5, chord="C")
        context.chart_available = True
        context.clock_bar = 1
        context.clock_beat = 2
        bass.on_musical_context(context)
        self.assertEqual(bass.synthesizer.scheduled_events[0].note, "C2")
        context.chord = "G"
        context.timestamp = .77
        context.beat_position = .54
        bass.on_musical_context(context)
        self.assertEqual(len(bass.synthesizer.scheduled_events), 1)
        self.assertTrue(bass.synthesizer.scheduled_events[0].note.startswith("G"))

    def test_half_note_attacks_at_chart_change_without_waiting_one_beat(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=.25, beat=1, phase=.5, chord="C")
        context.chart_available = True
        context.clock_bar = 1
        context.clock_beat = 1
        bass.on_musical_context(context)
        self.assertAlmostEqual(bass.synthesizer.scheduled_events[0].scheduled_time, 1.0)

        context.chord = "G"
        context.timestamp = .55
        context.beat = 2
        context.clock_beat = 2
        context.beat_position = .1
        changed = bass.on_musical_context(context)
        self.assertTrue(changed.note.startswith("G"))
        self.assertAlmostEqual(changed.start_time, .56)
        self.assertEqual(len(bass.synthesizer.scheduled_events), 1)

        bass.synthesizer.render_chunk(100, 1000, .55)
        self.assertTrue(bass.synthesizer.playing_event.note.startswith("G"))
        context.timestamp = .8
        context.beat_position = .6
        following = bass.on_musical_context(context)
        self.assertEqual((following.bar, following.beat), (1, 4))
        self.assertAlmostEqual(following.start_time, 1.55)

    def test_same_root_chart_quality_change_does_not_add_an_attack(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=.25, beat=1, phase=.5, chord="C")
        context.chart_available = True
        context.clock_bar = 1
        context.clock_beat = 1
        bass.on_musical_context(context)
        context.chord = "C7"
        context.timestamp = .55
        context.beat = 2
        context.clock_beat = 2
        context.beat_position = .1
        event = bass.on_musical_context(context)
        self.assertEqual((event.bar, event.beat), (1, 3))
        self.assertAlmostEqual(event.start_time, 1.0)

    def test_chart_change_during_recovery_enters_immediately(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=.25, beat=1, phase=.5, chord="C")
        context.chart_available = True
        context.clock_bar = 1
        context.clock_beat = 1
        bass.on_musical_context(context)
        context.chord = "G"
        context.timestamp = .55
        context.beat = 2
        context.clock_beat = 2
        context.beat_position = .1
        context.performance_state = "RECOVERING"
        event = bass.on_musical_context(context)
        self.assertTrue(event.note.startswith("G"))
        self.assertLess(event.start_time, 1.0)

    def test_chart_change_stays_ahead_of_known_output_buffer(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=.25, beat=1, phase=.5, chord="C")
        context.chart_available = True
        context.clock_bar = 1
        context.clock_beat = 1
        bass.on_musical_context(context)
        context.chord = "G"
        context.timestamp = .55
        context.beat = 2
        context.clock_beat = 2
        context.beat_position = .1
        context.output_latency = 46.0
        event = bass.on_musical_context(context)
        self.assertAlmostEqual(event.start_time, .600)
        bass.synthesizer.render_chunk(46, 1000, .55)
        self.assertIsNone(bass.synthesizer.playing_event)
        bass.synthesizer.render_chunk(46, 1000, .596)
        self.assertTrue(bass.synthesizer.playing_event.note.startswith("G"))

    def test_late_analysis_schedules_new_chart_root_after_rendered_audio(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=.25, beat=1, phase=.5, chord="C")
        context.chart_available = True
        context.clock_bar = 1
        context.clock_beat = 1
        bass.on_musical_context(context)

        # A interface demorou a despachar a nova posição: a saída já passou
        # pelo horário que o agendador antigo escolheria para a nota G.
        bass.synthesizer.render_chunk(300, 1000, .55)
        self.assertAlmostEqual(bass.synthesizer.rendered_until_time, .85)
        context.chord = "G"
        context.timestamp = .55
        context.beat = 2
        context.clock_beat = 2
        context.beat_position = .1
        context.output_latency = 46.0
        changed = bass.on_musical_context(context)
        self.assertTrue(changed.note.startswith("G"))
        self.assertAlmostEqual(changed.start_time, .854)
        bass.synthesizer.render_chunk(50, 1000, .85)
        self.assertTrue(bass.synthesizer.playing_event.note.startswith("G"))

    def test_chart_transition_cancels_old_root_before_confirmed_new_root(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=.65, beat=1, phase=.3, chord="B")
        context.chart_available = True
        context.clock_bar = 2
        context.clock_beat = 1
        context.chart_published_chord = "B"
        old = bass.on_musical_context(context)
        self.assertTrue(old.note.startswith("B"))
        self.assertEqual(len(bass.synthesizer.scheduled_events), 1)

        # O relógio alcança o sucessor antes que o áudio confirme o cursor.
        context.timestamp = .9
        context.beat_position = .8
        context.next_expected_chord = "F#"
        context.next_change_bar = 2
        context.chart_published_chord = "F#"
        context.chord_candidate = "F#"
        context.chord_candidate_confidence = .7
        changed = bass.on_musical_context(context)
        self.assertTrue(changed.note.startswith("F#"))
        self.assertTrue(all(event.note.startswith("F#")
                            for event in bass.synthesizer.scheduled_events))

        context.chord = "F#"
        context.timestamp = 1.05
        context.clock_beat = 2
        context.beat_position = .1
        new = bass.on_musical_context(context)
        self.assertTrue(new.note.startswith("F#"))
        context.next_expected_chord = "G"
        context.chart_published_chord = "F#"
        context.chord_candidate = "G"
        context.timestamp = 1.06
        context.beat_position = .12
        self.assertIsNone(bass.on_musical_context(context))
        bass.synthesizer.render_chunk(100, 1000, .90)
        self.assertTrue(bass.synthesizer.playing_event.note.startswith("F#"))

    def test_chart_mode_does_not_advance_from_clock_without_published_change(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=1.9, beat=4, phase=.8, chord="B")
        context.chart_available = True
        context.bar = 7
        context.clock_bar = 11
        context.clock_beat = 4
        context.next_expected_chord = "D#m"
        context.next_change_bar = 8
        context.chart_clock_chord = "C#"
        event = bass.on_musical_context(context)
        self.assertTrue(event.note.startswith("B"))

    def test_first_chart_chord_is_not_a_change_from_missing_chord(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=.75, beat=2, phase=.5, chord="G")
        context.chart_available = True
        context.clock_bar = 1
        context.clock_beat = 2
        event = bass.on_musical_context(context)
        self.assertEqual((event.bar, event.beat), (1, 3))
        self.assertAlmostEqual(event.start_time, 1.0)

    def test_clock_hint_alone_does_not_silence_confirmed_current_root(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=.75, beat=2, phase=.5, chord="B")
        context.chart_available = True
        context.clock_bar = 2
        context.clock_beat = 2
        context.next_expected_chord = "F#"
        context.next_change_bar = 2
        context.chart_clock_chord = "F#"
        context.smoothed_detected_chord = "B"
        context.stable_chord_confidence = .9
        context.stable_chord_duration = .4
        context.audio_activity = .1
        event = bass.on_musical_context(context)
        self.assertTrue(event.note.startswith("B"))

    def test_clock_hint_and_stable_audio_do_not_veto_published_root(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=.75, beat=2, phase=.5, chord="B")
        context.chart_available = True
        context.clock_bar = 2
        context.clock_beat = 2
        context.next_expected_chord = "F#"
        context.next_change_bar = 2
        context.chart_clock_chord = "F#"
        context.smoothed_detected_chord = "G#m"
        context.stable_chord_confidence = .8
        context.stable_chord_duration = .3
        context.audio_activity = .1
        event = bass.on_musical_context(context)
        self.assertTrue(event.note.startswith("B"))

    def test_audio_candidate_does_not_change_published_root(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=.75, beat=2, phase=.5, chord="B")
        context.chart_available = True
        context.clock_bar = 2
        context.clock_beat = 2
        context.next_expected_chord = "F#"
        context.next_change_bar = 2
        context.chart_clock_chord = "B"
        context.chord_candidate = "F#"
        context.chord_candidate_confidence = .65
        event = bass.on_musical_context(context)
        self.assertTrue(event.note.startswith("B"))

    def test_chart_only_legacy_pattern_also_uses_playalong_chord(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.ROOT,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=1.75, beat=4, phase=.5, chord="C")
        context.chart_available = True
        context.clock_bar = 1
        context.clock_beat = 4
        context.chart_next_clock_chord = "G"
        event = bass.on_musical_context(context)
        self.assertEqual((event.bar, event.beat), (2, 1))
        self.assertTrue(event.note.startswith("C"))

    def test_short_scheduling_lead_does_not_skip_nearby_beat(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=.48, beat=1, phase=.96)
        context.chart_available = True
        context.chart_clock_chord = "C"
        context.chart_next_clock_chord = "G"
        context.clock_bar = 1
        context.clock_beat = 1
        event = bass.on_musical_context(context)
        self.assertAlmostEqual(event.start_time, .5)

    def test_session_publishes_chart_at_clock_bar_while_audio_cursor_waits(self):
        session = SongSession(Song(title="Timeline", bpm=120,
                                   chart_text="[Intro]\nC G Am"))
        source = MusicalContext(audio_activity=.1)
        session.update_audio_tick(0.0, source_context=source)
        session.update_audio_tick(2.1, source_context=source)
        self.assertEqual(session.context.chart_clock_chord, "G")
        self.assertEqual(session.context.chart_next_clock_chord, "Am")

    def test_chart_only_prepares_first_tonic_at_playback_start(self):
        session = SongSession(Song(title="Start", bpm=120,
                                   chart_text="[Intro]\nG/B C"))
        session.start()
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          harmony_source=BassHarmonySource.CHART)
        event = bass.prepare_chart_start(session.context)
        self.assertAlmostEqual(event.start_time, 0.0)
        self.assertTrue(event.note.startswith("G"))
        self.assertEqual(bass.synthesizer.scheduled_events[0].source, "chart-only")

    def test_unmeasured_analyzer_tempo_does_not_replace_chart_bpm(self):
        session = SongSession(Song(title="Chart tempo", bpm=165,
                                   chart_text="[Intro]\nC G"))
        analyzer = AudioAnalyzer()
        context = analyzer.analyze_chunk(np.zeros(4096, dtype=np.float32),
                                         44100, 0.0, session=session)
        self.assertAlmostEqual(context.bpm, 165.0)

    def test_new_attacks_do_not_overlap_old_voice(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS)
        synth = bass.synthesizer
        synth.schedule_note((1, 1), .1, 36, 100, .5)
        synth.schedule_note((1, 2), .2, 43, 100, .5)
        synth.render_chunk(400, 1000, 0.0)
        self.assertLessEqual(len(synth._voices), 1)
        self.assertEqual(synth.scheduled_events, [])

    def test_displayed_playing_note_is_the_rendered_note_not_future_note(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS)
        synth = bass.synthesizer
        synth.schedule_note((1, 1), .1, 36, 100, .5, note="C2")
        synth.schedule_note((1, 2), .5, 43, 100, .5, note="G2")
        self.assertIsNone(synth.playing_event)
        synth.render_chunk(200, 1000, 0.0)
        self.assertEqual(synth.playing_event.note, "C2")
        self.assertEqual(synth.scheduled_events[0].note, "G2")
        synth.render_chunk(400, 1000, .2)
        self.assertEqual(synth.playing_event.note, "G2")

    def test_late_attack_is_discarded_instead_of_sounding_wrong_note(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS)
        synth = bass.synthesizer
        synth.schedule_note((1, 1), .1, 36, 100, .5, note="C2")
        synth.render_chunk(100, 1000, .12)
        self.assertIsNone(synth.playing_event)
        self.assertEqual(synth.scheduled_events, [])

    def test_chart_change_cancels_old_pending_note_and_ignores_audio(self):
        """A posição destacada é a autoridade, mesmo com áudio contraditório."""
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF,
                          harmony_source=BassHarmonySource.CHART)
        context = self.context(timestamp=.25, beat=1, phase=.5, chord="F#")
        context.chart_available = True
        context.chart_published_chord = "C"
        context.clock_bar = context.clock_beat = 1
        bass.on_musical_context(context)
        self.assertTrue(bass.synthesizer.scheduled_events[0].note.startswith("C"))

        context.timestamp = .55
        context.beat = context.clock_beat = 2
        context.beat_position = .1
        context.chart_published_chord = "Dm"
        context.smoothed_detected_chord = "F#"
        context.stable_chord_confidence = .99
        context.harmonic_event_chord = "F#"
        context.follow_confidence_level = "LOW"
        changed = bass.on_musical_context(context)
        self.assertTrue(changed.note.startswith("D"))
        self.assertTrue(all(event.note.startswith("D")
                            for event in bass.synthesizer.scheduled_events))

    def test_chart_change_is_immediate_in_uncertain_and_recovering(self):
        for state in ("UNCERTAIN", "RECOVERING"):
            with self.subTest(state=state):
                bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                                  note_value=BassNoteValue.HALF,
                                  harmony_source=BassHarmonySource.CHART)
                context = self.context(timestamp=.25, beat=1, phase=.5, chord="C")
                context.chart_available = True
                context.chart_published_chord = "C"
                context.clock_bar = context.clock_beat = 1
                bass.on_musical_context(context)
                context.timestamp = .55
                context.beat = context.clock_beat = 2
                context.beat_position = .1
                context.performance_state = state
                context.chart_published_chord = "D"
                event = bass.on_musical_context(context)
                self.assertTrue(event.note.startswith("D"))
                self.assertLess(event.start_time, 1.0)

    def test_fundamentals_share_d_root_and_inversions(self):
        for chord in ("D", "Dm", "Dm7", "Dm9", "Dm/F"):
            with self.subTest(chord=chord):
                decision = BassDecisionEngine().decide(
                    self.context(chord=chord), pattern_override=BassPatternType.FUNDAMENTALS)
                self.assertTrue(decision.root_note.startswith("D"))


class TestBassChartModeUI(unittest.TestCase):
    def test_chart_source_is_selectable_in_bassist_tab(self):
        with tempfile.TemporaryDirectory() as folder:
            try:
                root = tk.Tk()
            except tk.TclError as exc:
                self.skipTest(f"Tk indisponível neste ambiente: {exc}")
            root.withdraw()
            try:
                window = MainWindow(root, project_file_path=f"{folder}/project.json")
                self.assertEqual(window.combo_bass_harmony_source.get(),
                                 "Seguir sempre a cifra")
                self.assertEqual(window.analyzer.bass_player.harmony_source,
                                 BassHarmonySource.CHART)
                window.combo_bass_harmony_source.set("Cifra + confirmação do áudio")
                window._on_bass_harmony_source_changed()
                self.assertEqual(window.analyzer.bass_player.harmony_source,
                                 BassHarmonySource.FOLLOW)
            finally:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
