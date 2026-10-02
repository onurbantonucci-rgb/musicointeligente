"""Início da música e avanço só para frente (cifra e baixo).

Caso real: a bateria dá uma contagem sozinha, há ruído de fundo, e só depois a
banda entra no primeiro acorde da cifra. Nada antes desse acorde é música.
"""
import unittest

import numpy as np

from app.analysis.chart_harmony_aligner import BeatChroma, align_chart_to_beats, chord_template
from app.music.harmonic_rhythm import HarmonicRhythmEvent
from app.music.musical_context import MusicalContext
from app.music.position_estimator import TrackingState
from app.song.song import Song
from app.song.song_session import PerformanceState, SongSession


class TestAlignerIgnoresCountIn(unittest.TestCase):
    def test_drum_count_in_and_noise_are_not_the_first_chord(self):
        rng = np.random.default_rng(1)
        count_in = 8
        noise = np.abs(rng.normal(0.5, 0.25, (count_in, 12)))     # bateria/ruído: cromagrama espalhado
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        song = ["G"] * 8 + ["D"] * 8 + ["Em"] * 8 + ["C"] * 8
        chroma = np.vstack([noise, [chord_template(c) for c in song]])
        energy = np.full(len(chroma), 0.4)
        beat_chroma = BeatChroma(beat_times=0.75 * np.arange(len(chroma) + 1), chroma=chroma,
                                 energy=energy, duration=0.75 * len(chroma))
        timeline = align_chart_to_beats(beat_chroma, ["G", "D", "Em", "C"], 4)
        self.assertEqual(timeline.chords[:count_in], [None] * count_in)
        self.assertEqual(timeline.chords[count_in], "G")


class TestLiveStartGate(unittest.TestCase):
    CHART = "[Intro]\nC\nG\nAm\nF\n[Verso]\nDm\nG\nC\nF"

    def session(self):
        return SongSession(Song(title="Start gate", chart_text=self.CHART, bpm=120))

    @staticmethod
    def source(chord="--", onset=0.0, activity=0.1):
        return MusicalContext(audio_activity=activity, smoothed_detected_chord=chord,
                              stable_chord_confidence=.9 if chord != "--" else 0.0,
                              chord_start_time=onset)

    def test_drums_and_noise_do_not_start_the_song(self):
        session = self.session()
        for t in np.arange(0.0, 4.0, 0.1):
            # pulsos da bateria + ruído, sem acorde
            from app.analysis.tempo_detector import TempoResult
            pulse = TempoResult(bpm=120, confidence=.9, beat_timestamp=float(t)) if abs(t % 0.5) < 1e-6 else None
            session.update_audio_tick(float(t), source_context=self.source(), tempo_result=pulse)
        self.assertFalse(session.music_started)
        self.assertEqual(session.performance_state, PerformanceState.WAITING.value)
        self.assertEqual(session.position_estimator.chart_cursor_index, 0)

    def test_a_chord_that_is_not_the_first_one_is_ignored(self):
        session = self.session()
        for t in (1.0, 1.2, 1.4, 1.6):
            session.update_audio_tick(t, detected_chord="Am", detected_confidence=.9,
                                      source_context=self.source("Am", 1.0))
        self.assertFalse(session.music_started)
        self.assertEqual(session.current_chord, "C")

    def test_first_chord_starts_playing_with_downbeat_on_its_onset(self):
        session = self.session()
        for t in (0.5, 1.0, 1.5, 2.0):          # contagem
            session.update_audio_tick(t, source_context=self.source())
        for t in (2.55, 2.75, 2.95):             # C soou em 2.5 e foi confirmado depois
            session.update_audio_tick(t, detected_chord="C", detected_confidence=.9,
                                      source_context=self.source("C", 2.5))
        self.assertTrue(session.music_started)
        self.assertEqual(session.performance_state, PerformanceState.PLAYING.value)
        self.assertEqual(session.current_chord, "C")
        # O tempo 1 do compasso 1 é a batida em que o C soou.
        self.assertAlmostEqual(session.clock.beats_at(2.5), session.clock.downbeat_offset, delta=0.15)

    def test_arpeggiated_start_on_the_first_chord_notes(self):
        session = self.session()
        for t, note in ((1.0, "C3"), (1.15, "E3"), (1.3, "G3"), (1.45, "C4")):
            context = MusicalContext(audio_activity=.1, note=note, note_confidence=.9)
            session.update_audio_tick(t, source_context=context)
        self.assertTrue(session.music_started)


class TestNeverGoesBackToReplay(unittest.TestCase):
    def make(self, chart):
        session = SongSession(Song(title="Forward", chart_text=chart, bpm=120))
        estimator = session.position_estimator
        return session, estimator

    def test_lost_position_does_not_jump_back_into_the_middle_of_a_section(self):
        chart = "[Intro]\nC Am F G\n[Verso]\nDm G C F E\n[Refrão]\nEm C G D"
        session, estimator = self.make(chart)
        session.update_audio_tick(19.0, detected_chord="Em", detected_confidence=.9)
        before = estimator.chart_cursor_index
        estimator._tracking_state = TrackingState.LOST
        estimator._startup_lock = False
        # Progressão do MEIO do Verso (sem passar pelo início "Dm"): não é repetição de seção.
        for index, chord in enumerate(("G", "C", "F", "E")):
            event = HarmonicRhythmEvent(chord, index * 4.0, 4.0, 4.0, .95, "TEST")
            estimator._recent_harmonic_rhythm.append(((event.symbol, event.start_beat), event))
        for i, chord in enumerate(("C", "F", "E")):
            session.update_audio_tick(19.1 + i * .05, detected_chord=chord, detected_confidence=.95)
        self.assertGreaterEqual(estimator.chart_cursor_index, before)


if __name__ == "__main__":
    unittest.main()
