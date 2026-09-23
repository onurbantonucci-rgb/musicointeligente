"""Estados de performance: silêncio do músico, recuperação e fim natural."""
import unittest

from app.instruments.bass_model import BassPatternType
from app.instruments.bass_player import BassPlayer
from app.instruments.registry import VirtualPlayerRegistry
from app.music.musical_context import MusicalContext
from app.song.song import Song
from app.song.song_session import PerformanceState, SongSession


class TestPerformanceState(unittest.TestCase):
    def session(self, chart="[Intro]\nC\nG\nAm\nF\n[Verse]\nDm\nG\nC"):
        return SongSession(Song(title="Performance state", chart_text=chart, bpm=120))

    @staticmethod
    def source(activity=0.0, note="--", note_confidence=0.0):
        return MusicalContext(audio_activity=activity, note=note,
                              note_confidence=note_confidence)

    @staticmethod
    def bass_registry():
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.ROOT)
        return bass, VirtualPlayerRegistry(bass_player=bass)

    def make_playing(self, session):
        session.update_audio_tick(0.0, detected_chord="C", detected_confidence=.9,
                                  source_context=self.source(.1))
        self.assertEqual(session.performance_state, PerformanceState.RECOVERING.value)
        session.update_audio_tick(2.0, detected_chord="C", detected_confidence=.9,
                                  source_context=self.source(.1))
        self.assertEqual(session.performance_state, PerformanceState.PLAYING.value)

    def test_start_keeps_intro_prior_when_refrain_is_identical(self):
        chart = "[Intro]\nC\nG\nAm\nF\n[Refrain]\nC\nG\nAm\nF"
        session = self.session(chart)
        for timestamp, note in [(0.0, "C4"), (.2, "E4"), (.5, "G4"), (2.0, "C4")]:
            session.update_audio_tick(timestamp, source_context=self.source(.1, note, .95))
        self.assertLessEqual(session.current_bar, 2)
        self.assertEqual(session.current_section, "Intro")
        self.assertEqual(session.performance_state, PerformanceState.PLAYING.value)

    def test_three_ambiguous_notes_are_candidates_not_a_remote_jump(self):
        session = self.session("[Intro]\nC\nG\nAm\nF\n[Refrain]\nC\nG\nAm\nF")
        for timestamp, note in [(0.0, "C4"), (.2, "E4"), (.5, "G4"), (2.0, "C4")]:
            session.update_audio_tick(timestamp, source_context=self.source(.1, note, .95))
        self.assertLessEqual(session.current_bar, 2)
        self.assertGreaterEqual(session.position_confidence, .85)

    def test_short_pause_enters_holding_without_ending_or_new_bass_attacks(self):
        session = self.session()
        bass, registry = self.bass_registry()
        self.make_playing(session)
        registry.dispatch_context(session.context)
        self.assertTrue(bass.synthesizer.scheduled_events)
        session.update_audio_tick(4.6, source_context=self.source())
        self.assertEqual(session.performance_state, PerformanceState.HOLDING.value)
        self.assertIsNone(registry.dispatch_context(session.context).get("bass"))
        self.assertFalse(bass.synthesizer.scheduled_events)
        self.assertNotEqual(session.performance_state, PerformanceState.ENDED.value)

    def test_brief_unobserved_gap_is_uncertain_before_holding(self):
        session = self.session()
        self.make_playing(session)
        session.update_audio_tick(2.4, source_context=self.source())
        self.assertEqual(session.performance_state, PerformanceState.UNCERTAIN.value)
        self.assertEqual(session.current_section, "Intro")

    def test_activity_after_brief_gap_resumes_playing_without_downbeat_wait(self):
        session = self.session()
        self.make_playing(session)
        session.update_audio_tick(2.4, source_context=self.source())
        self.assertEqual(session.performance_state, PerformanceState.UNCERTAIN.value)
        session.update_audio_tick(2.5, detected_chord="G", detected_confidence=.9,
                                  source_context=self.source(.1))
        self.assertEqual(session.performance_state, PerformanceState.PLAYING.value)

    def test_long_pause_holds_position_and_waits_in_middle_without_ending(self):
        session = self.session()
        self.make_playing(session)
        session.update_audio_tick(2.0, detected_chord="Am", detected_confidence=.9,
                                  source_context=self.source(.1))
        held_bar = session.current_bar
        session.update_audio_tick(4.6, source_context=self.source())
        self.assertEqual(session.performance_state, PerformanceState.HOLDING.value)
        session.update_audio_tick(6.2, source_context=self.source())
        self.assertEqual(session.performance_state, PerformanceState.WAITING.value)
        self.assertEqual(session.current_bar, held_bar)
        session.update_audio_tick(12.0, source_context=self.source())
        self.assertEqual(session.performance_state, PerformanceState.WAITING.value)

    def test_return_recovers_and_prepares_next_downbeat_before_playing(self):
        session = self.session()
        bass, registry = self.bass_registry()
        self.make_playing(session)
        session.update_audio_tick(4.6, source_context=self.source())
        session.update_audio_tick(6.2, source_context=self.source())
        self.assertEqual(session.performance_state, PerformanceState.WAITING.value)
        session.update_audio_tick(7.0, detected_chord="Dm", detected_confidence=.9,
                                  source_context=MusicalContext(audio_activity=.1,
                                      smoothed_detected_chord="Dm", stable_chord_confidence=.9))
        self.assertEqual(session.performance_state, PerformanceState.RECOVERING.value)
        event = registry.dispatch_context(session.context)["bass"]
        self.assertEqual(event.beat, 1)
        self.assertGreater(event.start_time, session.context.timestamp)
        session.update_audio_tick(8.0, detected_chord="Dm", detected_confidence=.9,
                                  source_context=MusicalContext(audio_activity=.1,
                                      smoothed_detected_chord="Dm", stable_chord_confidence=.9))
        self.assertEqual(session.performance_state, PerformanceState.PLAYING.value)

    def test_end_requires_silence_near_chart_end_and_keeps_session_for_restart(self):
        session = self.session("[Intro]\nC\nG")
        bass, registry = self.bass_registry()
        self.make_playing(session)
        session.update_audio_tick(2.0, detected_chord="G", detected_confidence=.9,
                                  source_context=self.source(.1))
        registry.dispatch_context(session.context)
        session.update_audio_tick(4.6, source_context=self.source())
        session.update_audio_tick(6.2, source_context=self.source())
        session.update_audio_tick(10.1, source_context=self.source())
        self.assertEqual(session.performance_state, PerformanceState.ENDED.value)
        self.assertIsNone(registry.dispatch_context(session.context).get("bass"))
        self.assertFalse(bass.synthesizer.scheduled_events)
        self.assertEqual(session.current_chord, "G")
        session.start()
        self.assertEqual(session.performance_state, PerformanceState.WAITING.value)
        session.update_audio_tick(0.0, detected_chord="C", detected_confidence=.9,
                                  source_context=self.source(.1))
        self.assertEqual(session.performance_state, PerformanceState.RECOVERING.value)


if __name__ == "__main__":
    unittest.main()
