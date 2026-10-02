"""Relógio elástico, observações reais e eventos de baixo agendados."""
import unittest

import numpy as np

from app.analysis.tempo_detector import OnsetTempoDetector, TempoResult
from app.analysis.audio_analyzer import AudioAnalyzer
from app.instruments.bass_model import BassPatternType
from app.instruments.bass_player import BassPlayer
from app.instruments.bass_synthesizer import BassSynthesizer
from app.instruments.registry import VirtualPlayerRegistry
from app.music.musical_clock import MusicalClock
from app.music.musical_context import MusicalContext
from app.song.song import Song
from app.song.song_session import SongSession


class TestAdaptiveTempo(unittest.TestCase):
    def pulse(self, clock, timestamp):
        clock.update(timestamp, beat_timestamp=timestamp, observation_confidence=0.95)

    def test_constant_bpm_and_phase(self):
        clock = MusicalClock(120)
        for n in range(18):
            self.pulse(clock, n * 0.5)
        self.assertAlmostEqual(clock.current_bpm, 120, delta=0.5)
        self.assertAlmostEqual(clock.beat_phase, 0, delta=0.02)
        self.assertLess(abs(clock.phase_error_ms), 10)
        self.assertEqual(clock.tracking_state, "TRACKING")

    def test_acceleration_deceleration_and_human_swing(self):
        clock = MusicalClock(120)
        timestamp = 0.0
        self.pulse(clock, timestamp)
        for bpm in [120] * 4 + [122, 125, 127, 130] * 3:
            timestamp += 60 / bpm
            self.pulse(clock, timestamp)
        self.assertGreater(clock.current_bpm, 122)
        self.assertLess(clock.current_bpm, 132)
        fastest = clock.current_bpm
        for bpm in [130, 127, 124, 121, 118] * 3:
            timestamp += 60 / bpm
            self.pulse(clock, timestamp)
        self.assertLess(clock.current_bpm, fastest)
        self.assertGreater(clock.current_bpm, 115)
        self.assertLess(abs(clock.phase_error_ms), 160)

    def test_single_wrong_observation_does_not_set_147_bpm(self):
        clock = MusicalClock(120)
        self.pulse(clock, 0)
        self.pulse(clock, .5)
        clock.update(.5 + 60 / 147, beat_timestamp=.5 + 60 / 147,
                     observation_confidence=.95)
        self.assertAlmostEqual(clock.target_bpm, 120, delta=1)
        self.pulse(clock, 1.0)
        self.assertLess(abs(clock.target_bpm - 120), 2)

    def test_single_first_attack_cannot_move_the_phase(self):
        clock = MusicalClock(120)
        clock.update(.19, beat_timestamp=.19, observation_confidence=.9)
        self.assertEqual(clock.tracking_state, "UNCERTAIN")
        self.assertAlmostEqual(clock.beat_phase, .38, delta=.01)
        self.assertEqual(clock.phase_confidence, 0)

    def test_phase_error_decreases_on_consistent_shifted_pulses(self):
        clock = MusicalClock(120)
        clock.update(0)
        errors = []
        for n in range(1, 12):
            clock.update(n * .5 + .055, beat_timestamp=n * .5 + .055,
                         observation_confidence=.95)
            if n > 1:
                errors.append(abs(clock.phase_error_ms))
        self.assertLess(errors[-1], errors[0])
        self.assertLess(errors[-1], 15)

    def test_holdover_and_recovery_carry_phase(self):
        clock = MusicalClock(120)
        self.pulse(clock, 0)
        self.pulse(clock, .5)
        clock.update(4.0)
        self.assertEqual(clock.tracking_state, "HOLDOVER")
        self.assertEqual(clock.bar, 3)
        previous = clock.current_bpm
        self.pulse(clock, 4.5)
        self.assertEqual(clock.tracking_state, "TRACKING")
        self.assertAlmostEqual(clock.current_bpm, previous, delta=2)

    def test_precomputed_pulse_has_exact_timestamp_and_fallback_is_not_observation(self):
        detector = OnsetTempoDetector()
        detector._bpm = 120
        fallback = detector.get_tempo_at_time(.5)
        self.assertTrue(fallback.is_beat)
        self.assertIsNone(fallback.beat_timestamp)
        detector._beat_times = np.array([0, .48, .97], dtype=np.float32)
        self.assertAlmostEqual(detector.get_tempo_at_time(.5).beat_timestamp, .48, places=4)

    def test_live_attack_is_observed_only_once(self):
        detector = OnsetTempoDetector()
        silence = np.zeros(4096, dtype=np.float32)
        attack = np.ones(4096, dtype=np.float32) * .3
        detector.observe_chunk(silence, 44100, 0)
        detector.observe_chunk(attack, 44100, .5)
        self.assertEqual(detector.get_tempo_at_time(.5).beat_timestamp, .5)
        detector.observe_chunk(attack, 44100, .53)
        self.assertEqual(detector.get_tempo_at_time(.53).beat_timestamp, .5)

    def test_session_publishes_adaptive_tempo_separately_from_chart_position(self):
        session = SongSession(Song(title="Tempo", chart_text="C\nG\nAm", bpm=120))
        for n in range(5):
            t = n * .5
            session.update_audio_tick(t, tempo_result=TempoResult(
                bpm=120, confidence=.9, is_beat=True, beat_timestamp=t))
        self.assertEqual(session.clock.tracking_state, "TRACKING")
        self.assertEqual(session.context.tempo_tracking_state, "TRACKING")
        self.assertEqual(session.current_bar, session.chart_position.current_bar)
        self.assertIn("PHASE ERROR", session.format_tempo_diagnostics())

    def test_bass_is_scheduled_at_next_beat_instead_of_triggered_on_context(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.ROOT)
        ctx = MusicalContext(timestamp=.25, chord="C", bpm=120, bar=1,
                             beat=1, beat_position=.5, tempo_tracking_state="TRACKING")
        event = bass.on_musical_context(ctx)
        self.assertEqual(event.beat, 2)
        self.assertAlmostEqual(event.start_time, .5)
        self.assertEqual(len(bass.synthesizer._voices), 0)
        audio = bass.synthesizer.render_chunk(200, 1000, .4)
        self.assertTrue(np.allclose(audio[:99], 0))
        self.assertGreater(np.max(abs(audio[110:])), 0)

    def test_missed_events_are_discarded_without_burst(self):
        synth = BassSynthesizer(sample_rate=1000)
        synth.schedule_note((1, 1), .5, 48, 100, .2)
        synth.schedule_note((1, 2), 1.0, 50, 100, .2)
        audio = synth.render_chunk(100, 1000, 1.4)
        self.assertTrue(np.allclose(audio, 0))
        self.assertFalse(synth._scheduled)

    def test_next_bar_uses_predicted_chord(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.ROOT)
        ctx = MusicalContext(timestamp=1.75, chord="C", next_expected_chord="G",
                             chord_root="C", bpm=120, bar=1, beat=4,
                             beat_position=.5, tempo_tracking_state="TRACKING")
        event = bass.on_musical_context(ctx)
        self.assertEqual((event.bar, event.beat), (2, 1))
        self.assertEqual(bass.current_decision.chord, "G")
        self.assertTrue(event.note.startswith("G"))

    def test_audio_analyzer_connects_observed_pulses_session_and_mixer_bass(self):
        analyzer = AudioAnalyzer()
        analyzer.tempo_detector._bpm = 120
        analyzer.tempo_detector._beat_times = np.array([0, .5, 1.0], dtype=np.float32)
        bass = analyzer.bass_player
        bass.pattern = BassPatternType.ROOT
        registry = VirtualPlayerRegistry(bass_player=bass)
        session = SongSession(Song(title="Integration", chart_text="C\nG", bpm=120))
        silence = np.zeros(4096, dtype=np.float32)
        analyzer.analyze_chunk(silence, 44100, 0.0, session=session, players=registry)
        ctx = analyzer.analyze_chunk(silence, 44100, .5, session=session, players=registry)
        self.assertIs(ctx, session.context)
        self.assertEqual(session.clock.tracking_state, "TRACKING")
        self.assertIs(registry.bass_player, analyzer.bass_player)
        # Só pulsos (contagem/bateria) e silêncio: a música ainda não começou —
        # o baixo espera o primeiro acorde da cifra.
        self.assertFalse(session.music_started)
        self.assertFalse(bass.synthesizer._scheduled)

    def test_session_reset_restores_initial_tempo_and_clears_pulse_history(self):
        session = SongSession(Song(title="Reset", chart_text="C\nG", bpm=120))
        session.update_audio_tick(0, tempo_result=TempoResult(confidence=.9, beat_timestamp=0))
        session.update_audio_tick(.48, tempo_result=TempoResult(confidence=.9, beat_timestamp=.48))
        self.assertEqual(session.clock.tracking_state, "TRACKING")
        session.reset()
        self.assertEqual(session.clock.tracking_state, "HOLDOVER")
        self.assertEqual(session.clock.current_bpm, 120)
        self.assertEqual(session.clock.target_bpm, 120)
        self.assertEqual(session.clock.bar, 1)


if __name__ == "__main__":
    unittest.main()
