"""Testes automatizados da camada de andamento (BPM), batidas e tonalidade (v0.1-D e E)."""

import os
import unittest
from unittest.mock import Mock, patch
import numpy as np

from app.analysis.tempo_detector import OnsetTempoDetector
from app.analysis.key_detector import KrumhanslSchmucklerKeyDetector
from app.audio.audio_loader import FileAudioSource
from app.music.theory import PITCH_CLASSES


class TestTempoAndKeyAnalysis(unittest.TestCase):
    """Validação de BPM, grade de batidas e estimativa de tonalidade."""

    @classmethod
    def setUpClass(cls):
        cls.sr = 44100
        cls.tempo_detector = OnsetTempoDetector()
        cls.key_detector = KrumhanslSchmucklerKeyDetector()

        # Usar o áudio sintético padrão na raiz se existir
        cls.test_wav = os.path.abspath("test_song_120bpm.wav")

    def test_bpm_detection_120(self):
        """Valida se o OnsetTempoDetector identifica o andamento de 120 BPM."""
        if not os.path.exists(self.test_wav):
            self.skipTest("Arquivo test_song_120bpm.wav não encontrado na raiz.")

        source = FileAudioSource(self.test_wav)
        bpm = self.tempo_detector.analyze_audio(source.mono_data, source.get_sample_rate())
        source.close()

        # Tolerância razoável para andamento musical
        self.assertAlmostEqual(bpm, 120.0, delta=4.0)
        self.assertGreater(len(self.tempo_detector.beat_times), 15)

    def test_numpy_fallback_keeps_tempo_when_primary_backend_fails(self):
        source = FileAudioSource(self.test_wav)
        detector = OnsetTempoDetector()
        unavailable = Mock()
        unavailable.beat.beat_track.side_effect = RuntimeError("backend indisponível")
        with patch("app.analysis.tempo_detector.librosa", unavailable):
            bpm = detector.analyze_audio(source.mono_data, source.get_sample_rate())
        source.close()
        self.assertAlmostEqual(bpm, 120.0, delta=4.0)
        self.assertGreater(len(detector.beat_times), 15)

    def test_beat_pulse_matching(self):
        """Valida se o tracker detecta pulso quando o timestamp coincide com a batida."""
        if len(self.tempo_detector.beat_times) == 0:
            # Simular batidas a cada 0.5s (120 BPM)
            self.tempo_detector._bpm = 120.0
            self.tempo_detector._beat_times = np.arange(0.0, 10.0, 0.5, dtype=np.float32)

        first_beat = float(self.tempo_detector.beat_times[0])
        res_exact = self.tempo_detector.get_tempo_at_time(first_beat)
        self.assertTrue(res_exact.is_beat)
        self.assertGreater(res_exact.bpm, 0.0)

        # Meio tempo (fora do clique) não deve ser beat
        off_beat = first_beat + 0.25
        res_off = self.tempo_detector.get_tempo_at_time(off_beat)
        self.assertFalse(res_off.is_beat)

    def test_krumhansl_schmuckler_c_major(self):
        """Valida que a escala diatônica de Dó Maior é identificada como C Major ou A Minor."""
        self.key_detector.reset()

        # Simular notas da tonalidade de Dó Maior (C, D, E, F, G, A, B)
        c_chroma = np.zeros(12, dtype=np.float32)
        for note in ["C", "D", "E", "F", "G", "A", "B"]:
            idx = PITCH_CLASSES.index(note)
            c_chroma[idx] = 1.0

        # Alimentar o detector por múltiplos ciclos para permitir convergência do acumulador
        key_res = None
        for _ in range(8):
            key_res = self.key_detector.estimate_key(c_chroma, timestamp=1.0)

        self.assertIsNotNone(key_res)
        # O resultado deve ser C Major ou sua relativa A Minor
        self.assertIn(key_res.key_name, ["C Major", "A Minor"])
        self.assertGreater(key_res.confidence, 0.65)


if __name__ == "__main__":
    unittest.main()
