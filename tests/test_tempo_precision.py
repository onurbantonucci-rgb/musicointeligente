"""Precisão de andamento e gerador de áudio em BPMs arbitrários."""

import os
import tempfile
import unittest

import numpy as np

from app.analysis.tempo_detector import OnsetTempoDetector
from app.audio.audio_loader import FileAudioSource
from app.utils.audio_generator import generate_test_song


class TestTempoPrecision(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _detect(self, bpm: float) -> float:
        path = os.path.join(self._tmp.name, f"{bpm:g}.wav")
        generate_test_song(path, bpm=bpm, total_cycles=3)
        src = FileAudioSource(path)
        try:
            return OnsetTempoDetector().analyze_audio(src.mono_data, src.get_sample_rate())
        finally:
            src.close()

    def test_generator_works_at_non_120_bpm(self):
        # Antes quebrava com ValueError de broadcast em BPMs como 92 e 138.
        for bpm in (92.0, 138.0):
            self.assertGreater(self._detect(bpm), 0.0)

    def test_bpm_is_not_quantized(self):
        # O tempo bruto do librosa é quantizado (117.45 para 120); o refinado não deriva.
        for bpm in (92.0, 120.0, 138.0):
            self.assertAlmostEqual(self._detect(bpm), bpm, delta=0.5)

    def test_refine_rejects_irregular_grid(self):
        # Grade irregular demais: mantém o valor bruto em vez de uma inclinação distorcida.
        beats = np.array([0.0, 0.5, 1.0, 1.5, 3.5, 4.0, 4.5, 5.0, 7.5, 8.0], dtype=np.float32)
        self.assertEqual(OnsetTempoDetector._refine_bpm(beats, 120.0), 120.0)

    def test_refine_needs_enough_beats(self):
        self.assertEqual(OnsetTempoDetector._refine_bpm(np.array([0.0, 0.5]), 117.45), 117.45)


if __name__ == "__main__":
    unittest.main()
