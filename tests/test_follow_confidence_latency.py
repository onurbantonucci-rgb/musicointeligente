"""Confiança consolidada e compensação prudente de latência no baixo."""
import unittest

import numpy as np

from app.analysis.audio_analyzer import AudioAnalyzer
from app.instruments.bass_model import BassPatternType
from app.instruments.bass_player import BassPlayer
from app.music.follow_confidence import calculate_follow_confidence
from app.music.musical_context import MusicalContext


class TestFollowConfidenceAndLatency(unittest.TestCase):
    def predictive_context(self, latency_ms: float, level: str = "HIGH") -> MusicalContext:
        return MusicalContext(
            timestamp=1.25, chord="C", next_expected_chord="Am",
            bar=1, beat=3, beat_position=.5, bpm=120, meter="4/4",
            tempo_tracking_state="TRACKING", chart_available=True,
            position_confidence=.95, follow_confidence_level=level,
            confidence=.90, total_estimated_latency=latency_ms,
        )

    def test_global_follow_confidence_uses_gates_instead_of_blind_average(self):
        high = calculate_follow_confidence(
            tempo=.95, phase=.91, position=.93, harmonic=.90,
            chart_alignment=.96, stability=.88)
        low = calculate_follow_confidence(
            tempo=.96, phase=.94, position=.12, harmonic=.92,
            chart_alignment=.12, stability=.90)
        self.assertEqual(high.level, "HIGH")
        self.assertGreater(high.score, .75)
        self.assertEqual(low.level, "LOW")
        self.assertLess(low.score, high.score)

    def test_known_pipeline_parts_are_reported_without_driver_guessing(self):
        analyzer = AudioAnalyzer(sample_rate=10000, chunk_size=1000)
        analyzer.update_audio_format(10000, 1000, capture_latency_ms=20, output_latency_ms=50)
        ctx = analyzer.analyze_chunk(np.zeros(1000, dtype=np.float32), 10000, 0.0)
        self.assertAlmostEqual(ctx.capture_latency, 20.0)
        self.assertAlmostEqual(ctx.analysis_latency, 50.0)
        self.assertAlmostEqual(ctx.output_latency, 50.0)
        self.assertGreaterEqual(ctx.total_estimated_latency, 120.0)

    def test_chart_events_compensate_20_to_200ms_and_keep_musical_target(self):
        for latency_ms in (20, 50, 100, 200):
            with self.subTest(latency_ms=latency_ms):
                bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.ROOT)
                event = bass.on_musical_context(self.predictive_context(latency_ms))
                queued = bass.synthesizer.scheduled_events[0]
                self.assertAlmostEqual(event.start_time, queued.musical_time, places=4)
                self.assertAlmostEqual(queued.scheduled_time + latency_ms / 1000.0,
                                       queued.musical_time, places=4)
                self.assertEqual(queued.source, "chart")

    def test_medium_confidence_uses_root_and_fifth_without_latency_anticipation(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.AUTO)
        context = self.predictive_context(200, level="MEDIUM")
        context.beat = 4                    # próximo ataque: tempo 1 (todo padrão o toca)
        bass.on_musical_context(context)
        queued = bass.synthesizer.scheduled_events[0]
        # Confiança média: no máximo tônica e quinta. A INTENÇÃO pode tocar menos
        # (começo da música, trecho calmo), nunca arriscar mais.
        self.assertIn(bass.current_decision.pattern_type,
                      (BassPatternType.ROOT_FIFTH, BassPatternType.ROOT, BassPatternType.SUSTAINED))
        self.assertAlmostEqual(queued.scheduled_time, queued.musical_time, places=4)

    def test_low_confidence_does_not_anticipate_or_fill_weak_beats(self):
        bass = BassPlayer(sample_rate=1000, pattern=BassPatternType.AUTO)
        safe_downbeat = self.predictive_context(200, level="LOW")
        safe_downbeat.beat = 4
        safe_downbeat.beat_position = .5
        event = bass.on_musical_context(safe_downbeat)
        queued = bass.synthesizer.scheduled_events[0]
        self.assertEqual(event.beat, 1)
        self.assertAlmostEqual(queued.scheduled_time, queued.musical_time, places=4)

        weak_beat = self.predictive_context(200, level="LOW")
        self.assertIsNone(BassPlayer(sample_rate=1000).on_musical_context(weak_beat))


if __name__ == "__main__":
    unittest.main()
