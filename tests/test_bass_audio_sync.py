"""Sincronia do baixo com o áudio: tempo 1 do relógio, fase das batidas e harmonia da cifra."""
import unittest

import numpy as np

from app.analysis.audio_analyzer import AudioAnalyzer
from app.analysis.chart_harmony_aligner import (BeatChroma, MIN_CONFIDENCE, align_chart_to_beats,
                                                 chord_template, complete_beat_grid)
from app.analysis.tempo_detector import OnsetTempoDetector
from app.instruments.bass_model import BassNoteValue, BassPatternType, BassHarmonySource
from app.instruments.bass_player import BassPlayer
from app.music.musical_clock import MusicalClock
from app.music.musical_context import MusicalContext


class TestClockDownbeatAlignment(unittest.TestCase):
    def test_align_downbeat_renumbers_bars_from_music_start(self):
        clock = MusicalClock(bpm=120.0, meter="4/4")
        clock.update(2.1)
        clock.align_downbeat_at(2.0)
        self.assertEqual(clock.downbeat_offset, 4.0)
        self.assertEqual((clock.bar, clock.beat), (1, 1))
        clock.update(4.05)
        self.assertEqual((clock.bar, clock.beat), (2, 1))

    def test_snap_phase_puts_observed_attack_exactly_on_beat(self):
        clock = MusicalClock(bpm=120.0, meter="4/4")
        clock.update(2.1)
        clock.align_downbeat_at(2.08, snap_phase=True)
        self.assertAlmostEqual(clock.beats_at(2.08), clock.downbeat_offset, places=6)
        # O ajuste sobrevive às atualizações seguintes (não é recalculado do zero).
        clock.update(2.58)
        self.assertEqual((clock.bar, clock.beat), (1, 2))
        self.assertLess(clock.beat_position, 0.01)

    def test_seek_keeps_downbeat_offset(self):
        clock = MusicalClock(bpm=120.0, meter="4/4")
        clock.update(2.1)
        clock.align_downbeat_at(2.0)
        clock.update(6.0)
        clock.update(1.0)  # seek para trás
        self.assertEqual(clock.downbeat_offset, 4.0)
        clock.update(4.02)
        self.assertEqual((clock.bar, clock.beat), (2, 1))

    def test_reset_clears_downbeat_offset(self):
        clock = MusicalClock(bpm=120.0, meter="4/4")
        clock.update(2.1)
        clock.align_downbeat_at(2.0)
        clock.reset()
        self.assertEqual(clock.downbeat_offset, 0.0)


class TestBeatGridPhase(unittest.TestCase):
    @staticmethod
    def _click_track(sr, beats, duration):
        audio = np.zeros(int(sr * duration), dtype=np.float32)
        rng = np.random.default_rng(3)
        burst = (rng.standard_normal(int(sr * 0.03)) *
                 np.exp(-np.linspace(0, 8, int(sr * 0.03)))).astype(np.float32)
        for t in beats:
            start = int(t * sr)
            audio[start:start + len(burst)] += burst
        return audio

    def test_refine_beat_phase_moves_late_grid_onto_attacks(self):
        sr = 22050
        true_beats = np.arange(0.5, 9.5, 0.5)
        audio = self._click_track(sr, true_beats, 10.0)
        late_grid = (true_beats + 0.03).astype(np.float32)
        refined = OnsetTempoDetector._refine_beat_phase(audio, sr, late_grid)
        self.assertLess(float(np.max(np.abs(refined - true_beats))), 0.008)

    def test_refine_beat_phase_without_evidence_keeps_grid(self):
        sr = 22050
        grid = np.arange(0.5, 9.5, 0.5).astype(np.float32)
        refined = OnsetTempoDetector._refine_beat_phase(
            np.zeros(sr * 10, dtype=np.float32), sr, grid)
        np.testing.assert_array_equal(refined, grid)

    def test_next_beat_extrapolates_before_first_detected_beat(self):
        # O beat_track costuma pular o primeiro ataque: a grade é estendida para trás.
        detector = OnsetTempoDetector()
        detector._bpm = 120.0
        detector._beat_times = np.arange(0.8, 10.0, 0.5, dtype=np.float32)
        self.assertAlmostEqual(detector.get_tempo_at_time(0.25).next_beat_timestamp, 0.3, places=5)
        self.assertAlmostEqual(detector.get_tempo_at_time(0.31).next_beat_timestamp, 0.3, places=5)
        self.assertAlmostEqual(detector.get_tempo_at_time(1.0).next_beat_timestamp, 1.3, places=5)


def _beat_chroma(sequence, period=0.5, start=0.0, silent_beats=()):
    """BeatChroma sintético: uma batida por acorde em ``sequence`` (None = silêncio)."""
    count = len(sequence)
    chroma = np.zeros((count, 12))
    energy = np.full(count, 0.2)
    for k, chord in enumerate(sequence):
        if chord is None or k in silent_beats:
            energy[k] = 0.0
            continue
        chroma[k] = chord_template(chord)
    times = start + period * np.arange(count + 1)
    return BeatChroma(beat_times=times, chroma=chroma, energy=energy,
                      duration=float(times[-1]))


class TestChartToAudioAlignment(unittest.TestCase):
    def test_aligns_chart_without_durations_to_real_chord_lengths(self):
        # Cifra do Cifra Club: C Em Am F (sem durações). Áudio: 2 compassos cada,
        # depois de 2 batidas de silêncio (anacruse/introdução).
        sequence = [None, None] + ["C"] * 8 + ["Em"] * 8 + ["Am"] * 8 + ["F"] * 8
        timeline = align_chart_to_beats(_beat_chroma(sequence), ["C", "Em", "Am", "F"], 4)
        self.assertGreaterEqual(timeline.confidence, MIN_CONFIDENCE)
        self.assertEqual(timeline.chords[:2], [None, None])
        self.assertEqual(timeline.chords[2:], sequence[2:])
        # O tempo 1 cai onde a música começa (batida 2), não em t=0.
        self.assertEqual(timeline.downbeat_phase, 2)
        self.assertEqual(timeline.beat_position(10), (3, 1))

    def test_half_bar_changes_keep_order_of_chart(self):
        sequence = (["C"] * 2 + ["G"] * 2 + ["Am"] * 2 + ["F"] * 2) * 2
        timeline = align_chart_to_beats(_beat_chroma(sequence), ["C", "G", "Am", "F"] * 2, 4)
        self.assertEqual(timeline.chords, sequence)

    def test_capo_transposes_chart_shapes_to_sounding_chords(self):
        # Shapes C Em com capotraste na 2ª casa soam D F#m.
        sequence = ["D"] * 8 + ["F#m"] * 8
        timeline = align_chart_to_beats(_beat_chroma(sequence), ["C", "Em"], 4, capo_semitones=2)
        self.assertEqual(timeline.transposition, 2)
        self.assertEqual(timeline.chords, sequence)

    def test_chart_that_does_not_match_audio_has_low_confidence(self):
        sequence = ["C#"] * 8 + ["F#"] * 8 + ["G#"] * 8
        timeline = align_chart_to_beats(_beat_chroma(sequence), ["Am", "Dm"], 4)
        self.assertTrue(timeline is None or timeline.confidence < MIN_CONFIDENCE)

    def test_complete_beat_grid_extends_edges_and_fills_gaps(self):
        grid = complete_beat_grid([1.0, 1.5, 2.5, 3.0], bpm=120.0, duration=4.2)
        np.testing.assert_allclose(grid, np.arange(0.0, 4.75, 0.5))


class TestBassOnAlignedChart(unittest.TestCase):
    def context(self, timestamp, generation=0):
        context = MusicalContext(timestamp=timestamp, chord="C", chord_confidence=.9,
                                 bar=1, beat=1, beat_position=0.0, bpm=120,
                                 meter="4/4", tempo_tracking_state="TRACKING",
                                 follow_confidence_level="MEDIUM", position_confidence=.9)
        context.chart_available = True
        # O cursor publicado ainda mostra o acorde anterior: não deve importar.
        context.chart_published_chord = "C"
        context.position_generation = generation
        return context

    def bass(self, sequence, note_value=BassNoteValue.HALF, **kwargs):
        timeline = align_chart_to_beats(_beat_chroma(sequence, **kwargs),
                                        list(dict.fromkeys(sequence)), 4)
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=note_value, harmony_source=BassHarmonySource.CHART)
        bass.set_harmony_timeline(timeline)
        return bass, timeline

    def test_minor_chord_gets_its_own_root_exactly_on_its_audio_downbeat(self):
        bass, _ = self.bass(["C"] * 4 + ["Em"] * 4)
        bass.on_musical_context(self.context(1.7))
        events = bass.synthesizer.scheduled_events
        self.assertEqual([(round(e.scheduled_time, 3), e.note[0]) for e in events], [(2.0, "E")])

    def test_only_the_chosen_subdivision_is_played(self):
        # Troca no tempo 2 (fora da grade de mínimas): nenhum ataque extra; o
        # tempo 3 já sai com a tônica nova.
        bass, _ = self.bass(["C"] + ["G"] * 7)
        times = []
        for step in range(0, 80):
            now = step * 0.05
            bass.on_musical_context(self.context(now))
            for event in bass.synthesizer.scheduled_events:
                if now <= event.scheduled_time < now + 0.05:
                    times.append((round(event.scheduled_time, 3), event.note[0]))
            bass.synthesizer.render_chunk(50, 1000, now)
        self.assertEqual(sorted(set(times)), [(0.0, "C"), (1.0, "G"), (2.0, "G"), (3.0, "G")])

    def test_follows_audio_beats_not_nominal_tempo(self):
        # Batidas reais a 0.52 s (o relógio acha 120 BPM): o pulso segue o áudio.
        bass, timeline = self.bass(["Am"] * 8, period=0.52)
        bass.on_musical_context(self.context(0.9))
        self.assertAlmostEqual(bass.synthesizer.scheduled_events[0].scheduled_time, 1.04)

    def test_silence_in_audio_schedules_nothing(self):
        bass, _ = self.bass(["C"] * 4 + ["C"] * 4, silent_beats=(4, 5, 6, 7))
        bass.on_musical_context(self.context(1.8))
        self.assertEqual(bass.synthesizer.scheduled_events, [])

    def test_session_relocalization_does_not_cut_the_sounding_note(self):
        bass, _ = self.bass(["C"] * 8)
        bass.on_musical_context(self.context(0.0))
        bass.synthesizer.render_chunk(100, 1000, 0.0)
        self.assertIsNotNone(bass.synthesizer.playing_event)
        bass.on_musical_context(self.context(0.1, generation=1))
        self.assertIsNotNone(bass.synthesizer.playing_event)

    def test_low_confidence_timeline_falls_back_to_published_chart(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.FUNDAMENTALS,
                          note_value=BassNoteValue.HALF, harmony_source=BassHarmonySource.CHART)
        bass.set_harmony_timeline(align_chart_to_beats(
            _beat_chroma(["C#"] * 8 + ["F#"] * 8), ["Am", "Dm"], 4))
        context = self.context(0.25)
        context.clock_bar = context.clock_beat = 1
        context.beat_position = 0.5
        event = bass.on_musical_context(context)
        self.assertEqual(event.note[0], "C")
        self.assertEqual(bass.synthesizer.scheduled_events[0].source, "chart-only")


class TestTrackAnalysisSurvivesStop(unittest.TestCase):
    def test_stop_keeps_beat_grid_and_beat_chroma(self):
        analyzer = AudioAnalyzer()
        detector = OnsetTempoDetector()
        detector._bpm = 120.0
        detector._beat_times = np.arange(0.0, 10.0, 0.5, dtype=np.float32)
        analyzer.apply_tempo_analysis(detector)
        beat_chroma = _beat_chroma(["C"] * 8)
        analyzer.set_beat_chroma(beat_chroma)
        analyzer.reset_musical_history(keep_track_analysis=True)
        self.assertEqual(analyzer.tempo_detector.bpm, 120.0)
        self.assertEqual(len(analyzer.tempo_detector.beat_times), 20)
        self.assertIs(analyzer._beat_chroma, beat_chroma)
        analyzer.reset_musical_history()
        self.assertEqual(analyzer.tempo_detector.bpm, 0.0)
        self.assertIsNone(analyzer._beat_chroma)


if __name__ == "__main__":
    unittest.main()
