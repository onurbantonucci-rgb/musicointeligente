"""Escuta do groove: o baixista ouve COMO a banda toca, não só o BPM."""
import unittest

import numpy as np

from app.analysis.groove_feel import (BUSY, SPACIOUS, SYNCOPATED, GrooveFeelAnalyzer,
                                      LiveOnsetGrid, bar_features)
from app.analysis.tempo_detector import OnsetTempoDetector

SPACIOUS_BAR = np.array([[1.0, 0, 0, 0], [0.05, 0, 0, 0], [0.7, 0, 0, 0], [0.05, 0, 0, 0]])
SYNCOPATED_BAR = np.array([[1.0, 0, 0.8, 0.6], [0.1, 0, 0.9, 0], [0.2, 0.7, 0.8, 0], [0.1, 0, 0.9, 0.5]])
BUSY_BAR = np.full((4, 4), 0.9)


def feed(grids, energies=None, bpm=68.0):
    analyzer = GrooveFeelAnalyzer(4)
    energies = energies if energies is not None else [0.5] * len(grids)
    for grid, energy in zip(grids, energies):
        profile = analyzer.observe_bar(grid, energy, bpm)
    return analyzer, profile


class TestGrooveFeel(unittest.TestCase):
    def test_same_bpm_different_feel(self):                        # TEST 9
        _, spacious = feed([SPACIOUS_BAR] * 6)
        _, syncopated = feed([SYNCOPATED_BAR] * 6)
        self.assertEqual(spacious.tempo_bpm, syncopated.tempo_bpm)
        self.assertEqual(spacious.label, SPACIOUS)
        self.assertEqual(syncopated.label, SYNCOPATED)
        self.assertGreater(syncopated.syncopation, spacious.syncopation + 0.3)
        self.assertGreater(spacious.space_ratio, syncopated.space_ratio)
        self.assertEqual(spacious.subdivision, "QUARTER")
        self.assertIn(syncopated.subdivision, ("EIGHTH", "SIXTEENTH"))

    def test_spacious_track_is_sparse(self):                       # TEST 10
        _, profile = feed([SPACIOUS_BAR] * 6)
        self.assertIn(profile.density_level, ("VERY_LOW", "LOW"))
        self.assertGreater(profile.confidence, 0.5)

    def test_label_does_not_flip_flop(self):                       # TEST 29
        analyzer, _ = feed([SPACIOUS_BAR, BUSY_BAR] * 8)
        self.assertLessEqual(analyzer.label_changes, 1)

    def test_persistent_change_is_heard(self):
        analyzer, profile = feed([SPACIOUS_BAR] * 6 + [BUSY_BAR] * 6)
        self.assertEqual(profile.label, BUSY)
        self.assertEqual(analyzer.label_changes, 1)

    def test_energy_trend(self):
        _, rising = feed([SPACIOUS_BAR] * 10, energies=[0.2] * 6 + [0.6] * 4)
        self.assertGreater(rising.energy_trend, 0.3)

    def test_half_time_feel(self):
        half = np.array([[1.0, 0, 0, 0], [0.1, 0, 0, 0], [1.0, 0, 0, 0], [0.1, 0, 0, 0]])
        features = bar_features(half)
        self.assertGreater(features.half_time, 0.5)

    def test_live_grid_closes_bars(self):
        grid = LiveOnsetGrid(4)
        closed = None
        for i in range(32):
            beat = i // 2 % 4 + 1
            closed = grid.observe(1 + i // 8, beat, 0.0 if i % 2 == 0 else 0.5,
                                  flux=1.0 if i % 2 == 0 else 0.1, energy=0.3) or closed
        self.assertIsNotNone(closed)
        bar_grid, energy = closed
        self.assertEqual(bar_grid.shape, (4, 4))
        self.assertAlmostEqual(energy, 0.3)


class TestCompoundMeterFromAudio(unittest.TestCase):
    def test_six_eight_is_detected_from_the_onset_envelope(self):
        # Colcheias a 160/min com acento a cada 3 (semínima pontuada) — 6/8.
        fps, eighth = 43.07, 60.0 / 160.0
        frames = int(60 * fps)
        envelope = np.zeros(frames)
        for i in range(int(60 / eighth)):
            frame = int(round(i * eighth * fps))
            if frame < frames:
                envelope[frame] = 1.0 if i % 3 == 0 else 0.25
        # O rastreador "escolhe" um pulso de 1,5 colcheia (o erro observado na prática).
        beats = np.arange(0, frames, 1.5 * eighth * fps).astype(int)
        bpm = OnsetTempoDetector._compound_eighth_bpm(envelope, 22050, beats)
        self.assertIsNotNone(bpm)
        self.assertAlmostEqual(bpm, 160.0, delta=6.0)

    def test_simple_meter_is_left_alone(self):
        fps, beat = 43.07, 0.5
        frames = int(60 * fps)
        envelope = np.zeros(frames)
        for i in range(int(60 / (beat / 2))):
            frame = int(round(i * beat / 2 * fps))
            if frame < frames:
                envelope[frame] = 1.0 if i % 2 == 0 else 0.5
        beats = np.arange(0, frames, beat * fps).astype(int)
        self.assertIsNone(OnsetTempoDetector._compound_eighth_bpm(envelope, 22050, beats))


if __name__ == "__main__":
    unittest.main()
