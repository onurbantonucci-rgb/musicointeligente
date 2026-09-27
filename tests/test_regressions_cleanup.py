"""Regressões encontradas durante uma limpeza de código morto/imports (v0.8).

Cobre bugs reais descobertos ao investigar avisos do pyflakes de variáveis locais
"atribuídas mas nunca usadas" no PositionEstimator — não eram apenas ruído, escondiam
dois defeitos genuínos na previsão estrutural de acorde (EstimatedPosition.predicted_chord).
"""

import unittest

from app.input.chart_parser import ChartParser
from app.music.chart_alignment import ChartAlignment
from app.music.musical_clock import MusicalClock
from app.music.position_estimator import PositionEstimator
from app.music.prediction_engine import PredictionEngine, MusicPrediction


class TestPredictedChordWiring(unittest.TestCase):
    """A previsão do PredictionEngine deve realmente alcançar EstimatedPosition."""

    def _estimator(self):
        chart = ChartParser.parse("[Intro]\nC  G  Am  F\n[Verso]\nC  G  Am  F\n")
        alignment = ChartAlignment(chart)
        clock = MusicalClock(bpm=120.0, meter="4/4")
        engine = PredictionEngine()
        return PositionEstimator(alignment, clock, prediction_engine=engine), clock, engine

    def test_prediction_engine_override_reaches_estimated_position(self):
        # BUG 1: a atribuição de fallback ("effective_chord") era descartada e o retorno
        # sempre usava nominal_pos.next_chord cru, ignorando o motor de previsão.
        # BUG 2: o código lia "pred.predicted_next_chord" (singular), um atributo que
        # nunca existiu em MusicPrediction (o campo real é "predicted_chords", plural),
        # então getattr sempre voltava None e o override nunca disparava.
        estimator, clock, engine = self._estimator()
        engine._latest_prediction = MusicPrediction(predicted_chords=["Dm"])

        clock.update(0.5)
        estimator.update(timestamp=0.5, detected_chord="C", detected_confidence=0.9)

        self.assertEqual(estimator.current_estimated_position.predicted_chord, "Dm")

    def test_placeholder_prediction_falls_back_to_chart_next_chord(self):
        # Sem previsão útil (lista vazia ou "--"), cai no próximo acorde literal da cifra.
        estimator, clock, engine = self._estimator()
        engine._latest_prediction = MusicPrediction(predicted_chords=[])

        clock.update(0.5)
        pos = estimator.update(timestamp=0.5, detected_chord="C", detected_confidence=0.9)

        self.assertEqual(estimator.current_estimated_position.predicted_chord, pos.next_chord)


if __name__ == "__main__":
    unittest.main()
