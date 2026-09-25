"""Fontes independentes de BPM e tom por música."""

import unittest

from app.analysis.tempo_detector import TempoResult
from app.music.musical_context import MusicalContext
from app.song.song import PerformanceSettings, Song
from app.song.song_session import SongSession


class TestReferenceSources(unittest.TestCase):
    def test_sources_are_persisted(self):
        settings = PerformanceSettings(bpm_source="CHART", key_source="AUDIO")
        restored = PerformanceSettings.from_dict(settings.to_dict())
        self.assertEqual(restored.bpm_source, "CHART")
        self.assertEqual(restored.key_source, "AUDIO")

    def test_chart_bpm_rejects_audio_tempo_and_pulse(self):
        song = Song(bpm=132.0, chart_text="BPM: 132\n[Intro]\nC")
        song.performance_settings.bpm_source = "CHART"
        session = SongSession(song)
        heard = MusicalContext(audio_activity=.1)
        session.update_audio_tick(
            1.0, source_context=heard,
            tempo_result=TempoResult(bpm=96.0, confidence=.95, beat_timestamp=1.0))
        self.assertAlmostEqual(session.clock.bpm, 132.0)

    def test_audio_bpm_accepts_measured_tempo(self):
        song = Song(bpm=132.0, chart_text="BPM: 132\n[Intro]\nC")
        song.performance_settings.bpm_source = "AUDIO"
        session = SongSession(song)
        session.update_audio_tick(
            1.0, source_context=MusicalContext(audio_activity=.1),
            tempo_result=TempoResult(bpm=96.0, confidence=.95))
        self.assertAlmostEqual(session.clock.bpm, 96.0)

    def test_audio_key_is_not_overwritten_by_chart_key(self):
        song = Song(key="G Major", chart_text="Tom: G\n[Intro]\nG D")
        song.performance_settings.key_source = "AUDIO"
        session = SongSession(song)
        session.update_audio_tick(
            1.0, detected_key="C Major",
            source_context=MusicalContext(key="C Major", key_confidence=.9))
        self.assertEqual(session.context.key, "C Major")


if __name__ == "__main__":
    unittest.main()
