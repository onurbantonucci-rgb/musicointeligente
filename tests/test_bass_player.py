"""Bateria de testes formais do Baixista Virtual (BassPlayer v0.2).

Cobre os 12 testes obrigatórios:
  1. Tom C -> toca C
  2. Tom G -> toca G
  3. Tom Am -> toca A
  4. Tom F -> toca F
  5. Padrão ROOT + FIFTH (C->C-G-C-G, G->G-D-G-D, Am->A-E-A-E)
  6. Sincronização temporal a 120 BPM (intervalo de 500 ms)
  7. Mudança de andamento a 100 BPM (intervalo de 600 ms)
  8. Mudança de acorde (C -> G transição imediata)
  9. Duração do acorde (2 tempos vs 4 tempos)
  10. Fallback em baixa confiança (toca apenas fundamental)
  11. Teste de síntese de áudio (sem clipping, faixa -1.0 a +1.0)
  12. Teste integrado com áudio real (violao_teste_120bpm.wav)
"""

import os
import unittest
import numpy as np

from app.music.musical_context import MusicalContext
from app.instruments.bass_model import BassPatternType, BassDecision, BassNoteEvent
from app.instruments.bass_decision import BassDecisionEngine
from app.instruments.bass_pattern import BassPatternGenerator
from app.instruments.bass_performance import BassPerformanceEngine
from app.instruments.bass_synthesizer import BassSynthesizer
from app.instruments.bass_player import BassPlayer
from app.analysis.audio_analyzer import AudioAnalyzer
from app.audio.audio_loader import FileAudioSource
from app.utils.audio_generator import generate_acoustic_guitar_sample


class TestBassPlayer(unittest.TestCase):
    """Bateria de testes unitários e de integração do Baixista Virtual."""

    def setUp(self):
        self.decision_engine = BassDecisionEngine()
        self.pattern_gen = BassPatternGenerator()
        self.perf_engine = BassPerformanceEngine()
        self.synth = BassSynthesizer(sample_rate=44100)
        self.bass_player = BassPlayer(sample_rate=44100)

    # -------------------------------------------------------------
    # 1. Tom C -> toca C
    # -------------------------------------------------------------
    def test_1_chord_c_plays_c(self):
        """TESTE 1 — Acorde C deve resultar na nota fundamental C no baixo."""
        ctx = MusicalContext(timestamp=0.0)
        ctx.chord = "C"
        ctx.key = "C Major"
        ctx.bpm = 120.0
        ctx.chord_confidence = 0.90

        decision = self.decision_engine.decide(ctx, pattern_override=BassPatternType.ROOT)
        self.assertTrue(decision.root_note_name.startswith("C"), f"Esperado C, obtido {decision.root_note_name}")
        # C no registro do baixo é C1 (MIDI 24/fora) ou C2 (MIDI 36)
        self.assertEqual(decision.root_midi_note, 36)

    # -------------------------------------------------------------
    # 2. Tom G -> toca G
    # -------------------------------------------------------------
    def test_2_chord_g_plays_g(self):
        """TESTE 2 — Acorde G deve resultar na nota fundamental G no baixo."""
        ctx = MusicalContext(timestamp=0.0)
        ctx.chord = "G"
        ctx.key = "G Major"
        ctx.bpm = 120.0
        ctx.chord_confidence = 0.90

        decision = self.decision_engine.decide(ctx, pattern_override=BassPatternType.ROOT)
        self.assertTrue(decision.root_note_name.startswith("G"), f"Esperado G, obtido {decision.root_note_name}")
        # G no registro do baixo é G1 (MIDI 31) ou G2 (MIDI 43)
        self.assertIn(decision.root_midi_note, [31, 43])

    # -------------------------------------------------------------
    # 3. Tom Am -> toca A
    # -------------------------------------------------------------
    def test_3_chord_am_plays_a(self):
        """TESTE 3 — Acorde Am deve resultar na nota fundamental A no baixo."""
        ctx = MusicalContext(timestamp=0.0)
        ctx.chord = "Am"
        ctx.key = "C Major"
        ctx.bpm = 120.0
        ctx.chord_confidence = 0.90

        decision = self.decision_engine.decide(ctx, pattern_override=BassPatternType.ROOT)
        self.assertTrue(decision.root_note_name.startswith("A"), f"Esperado A, obtido {decision.root_note_name}")
        # A no registro do baixo é A1 (MIDI 33) ou A2 (MIDI 45)
        self.assertIn(decision.root_midi_note, [33, 45])

    # -------------------------------------------------------------
    # 4. Tom F -> toca F
    # -------------------------------------------------------------
    def test_4_chord_f_plays_f(self):
        """TESTE 4 — Acorde F deve resultar na nota fundamental F no baixo."""
        ctx = MusicalContext(timestamp=0.0)
        ctx.chord = "F"
        ctx.key = "C Major"
        ctx.bpm = 120.0
        ctx.chord_confidence = 0.90

        decision = self.decision_engine.decide(ctx, pattern_override=BassPatternType.ROOT)
        self.assertTrue(decision.root_note_name.startswith("F"), f"Esperado F, obtido {decision.root_note_name}")
        # F no registro do baixo é F1 (MIDI 29) ou F2 (MIDI 41)
        self.assertIn(decision.root_midi_note, [29, 41])

    # -------------------------------------------------------------
    # 5. Padrão ROOT + FIFTH: C->C-G-C-G, G->G-D-G-D, Am->A-E-A-E
    # -------------------------------------------------------------
    def test_5_pattern_root_fifth(self):
        """TESTE 5 — Padrão ROOT + FIFTH deve gerar alternância correta de fundamental e quinta."""
        test_cases = [
            ("C", "C", "G"),
            ("G", "G", "D"),
            ("Am", "A", "E"),
        ]

        for chord, expected_root, expected_fifth in test_cases:
            ctx = MusicalContext(timestamp=0.0)
            ctx.chord = chord
            ctx.chord_confidence = 0.90
            ctx.bpm = 120.0

            decision = self.decision_engine.decide(ctx, pattern_override=BassPatternType.ROOT_FIFTH)
            steps = self.pattern_gen.generate_pattern(decision, beats=4, chord_duration_beats=4.0)

            self.assertEqual(len(steps), 4, f"Padrão deve conter 4 passos para compasso 4/4 no acorde {chord}")

            # Beat 1: Fundamental
            self.assertEqual(steps[0][0], 1)
            self.assertTrue(steps[0][1].startswith(expected_root), f"Beat 1 esperado {expected_root}, obtido {steps[0][1]}")

            # Beat 2: Quinta
            self.assertEqual(steps[1][0], 2)
            self.assertTrue(steps[1][1].startswith(expected_fifth), f"Beat 2 esperado {expected_fifth}, obtido {steps[1][1]}")

            # Beat 3: Fundamental
            self.assertEqual(steps[2][0], 3)
            self.assertTrue(steps[2][1].startswith(expected_root), f"Beat 3 esperado {expected_root}, obtido {steps[2][1]}")

            # Beat 4: Quinta
            self.assertEqual(steps[3][0], 4)
            self.assertTrue(steps[3][1].startswith(expected_fifth), f"Beat 4 esperado {expected_fifth}, obtido {steps[3][1]}")

    # -------------------------------------------------------------
    # 6. Sincronização temporal: 120 BPM = 500 ms entre notas
    # -------------------------------------------------------------
    def test_6_temporal_sync_120_bpm(self):
        """TESTE 6 — A 120 BPM, o intervalo entre tempos musicais deve ser de exatamente 500 ms."""
        ctx = MusicalContext(timestamp=0.0)
        ctx.chord = "C"
        ctx.bpm = 120.0
        ctx.chord_confidence = 0.90

        decision = self.decision_engine.decide(ctx, pattern_override=BassPatternType.ROOT)
        steps = self.pattern_gen.generate_pattern(decision, beats=4, chord_duration_beats=4.0)

        events = self.perf_engine.create_events_for_bar(
            decision=decision,
            pattern_steps=steps,
            bar=1,
            bar_start_time=0.0,
            beat_duration=0.5,  # 60 / 120
            total_beats=4
        )

        self.assertEqual(len(events), 4)
        for i in range(len(events) - 1):
            delta = events[i + 1].start_time - events[i].start_time
            self.assertAlmostEqual(delta, 0.500, places=3,
                                   msg=f"Intervalo entre notas a 120 BPM deve ser 0.500s, obtido {delta:.4f}s")

    # -------------------------------------------------------------
    # 7. Mudança de andamento: 100 BPM = 600 ms entre notas
    # -------------------------------------------------------------
    def test_7_temporal_sync_100_bpm(self):
        """TESTE 7 — A 100 BPM, o intervalo entre tempos musicais deve ser de exatamente 600 ms."""
        ctx = MusicalContext(timestamp=0.0)
        ctx.chord = "C"
        ctx.bpm = 100.0
        ctx.chord_confidence = 0.90

        decision = self.decision_engine.decide(ctx, pattern_override=BassPatternType.ROOT)
        steps = self.pattern_gen.generate_pattern(decision, beats=4, chord_duration_beats=4.0)

        events = self.perf_engine.create_events_for_bar(
            decision=decision,
            pattern_steps=steps,
            bar=1,
            bar_start_time=0.0,
            beat_duration=0.6,  # 60 / 100
            total_beats=4
        )

        self.assertEqual(len(events), 4)
        for i in range(len(events) - 1):
            delta = events[i + 1].start_time - events[i].start_time
            self.assertAlmostEqual(delta, 0.600, places=3,
                                   msg=f"Intervalo entre notas a 100 BPM deve ser 0.600s, obtido {delta:.4f}s")

    # -------------------------------------------------------------
    # 8. Mudança de acorde: C para G muda imediatamente para G
    # -------------------------------------------------------------
    def test_8_chord_change_immediate(self):
        """TESTE 8 — Transição de acorde (C -> G) altera a nota do baixo imediatamente no próximo tempo/compasso."""
        self.bass_player.reset()
        self.bass_player.pattern = BassPatternType.ROOT

        # Compasso 1, Beat 1: Acorde C
        ctx1 = MusicalContext(timestamp=0.0)
        ctx1.chord = "C"
        ctx1.bpm = 120.0
        ctx1.bar = 1
        ctx1.beat = 1
        ctx1.chord_confidence = 0.90
        ev1 = self.bass_player.on_musical_context(ctx1)
        self.assertIsNotNone(ev1)
        self.assertTrue(ev1.note.startswith("C"), f"Esperado C, obtido {ev1.note}")

        # Compasso 1, Beat 2: Mudança harmônica imediata para G
        ctx2 = MusicalContext(timestamp=0.5)
        ctx2.chord = "G"
        ctx2.bpm = 120.0
        ctx2.bar = 1
        ctx2.beat = 2
        ctx2.chord_confidence = 0.90
        ev2 = self.bass_player.on_musical_context(ctx2)
        self.assertIsNotNone(ev2)
        self.assertTrue(ev2.note.startswith("G"), f"Esperado G após transição, obtido {ev2.note}")

    # -------------------------------------------------------------
    # 9. Duração do acorde: 2 tempos vs 4 tempos
    # -------------------------------------------------------------
    def test_9_chord_duration_adaptation(self):
        """TESTE 9 — Acorde com 2 tempos gera padrão curto; com 4 tempos gera padrão completo."""
        ctx = MusicalContext(timestamp=0.0)
        ctx.chord = "C"
        ctx.bpm = 120.0
        ctx.chord_confidence = 0.90

        decision = self.decision_engine.decide(ctx, pattern_override=BassPatternType.AUTO)

        # 2 tempos de duração
        steps_2_beats = self.pattern_gen.generate_pattern(decision, beats=4, chord_duration_beats=2.0)
        self.assertEqual(len(steps_2_beats), 2, f"Esperado 2 passos para duração de 2 tempos, obtido {len(steps_2_beats)}")

        # 4 tempos de duração
        steps_4_beats = self.pattern_gen.generate_pattern(decision, beats=4, chord_duration_beats=4.0)
        self.assertGreaterEqual(len(steps_4_beats), 4, f"Esperado 4 passos para duração completa de 4 tempos, obtido {len(steps_4_beats)}")

    # -------------------------------------------------------------
    # 10. Fallback: baixa confiança -> toca apenas fundamental
    # -------------------------------------------------------------
    def test_10_low_confidence_fallback(self):
        """TESTE 10 — Se a confiança for baixa (< 0.40), o sistema entra em fallback tocando apenas a fundamental."""
        ctx = MusicalContext(timestamp=0.0)
        ctx.chord = "C"
        ctx.bpm = 120.0
        ctx.chord_confidence = 0.25  # Menor que MIN_BASS_CHORD_CONFIDENCE (0.40)

        decision = self.decision_engine.decide(ctx, pattern_override=BassPatternType.AUTO)
        self.assertEqual(decision.pattern_type, BassPatternType.ROOT,
                         "Confiança baixa deve forçar o padrão ROOT (fundamental pura)")
        self.assertTrue("Fallback" in decision.reason)

        steps = self.pattern_gen.generate_pattern(decision, beats=4, chord_duration_beats=4.0)
        for _, note_name, _, reason in steps:
            self.assertTrue(note_name.startswith("C"))
            self.assertTrue("Fundamental" in reason)

    # -------------------------------------------------------------
    # 11. Teste de síntese: geração de áudio sem clipping
    # -------------------------------------------------------------
    def test_11_synthesis_no_clipping(self):
        """TESTE 11 — Síntese do contrabaixo deve gerar sinal dentro da faixa de -1.0 a +1.0 sem clipping ou NaN."""
        # 1. Teste de nota isolada
        audio = self.synth.synthesize_note_to_array(midi_note=36, duration=1.0, velocity=100, sr=44100)
        self.assertGreater(len(audio), 0)
        self.assertFalse(np.isnan(audio).any(), "Sinal contém NaN")
        self.assertFalse(np.isinf(audio).any(), "Sinal contém Inf")
        self.assertLessEqual(np.max(np.abs(audio)), 1.0, "Sinal ultrapassou +1.0 (clipping)")

        # 2. Teste de renderização contínua de blocos (render_chunk)
        self.synth.clear()
        self.synth.trigger_note(midi_note=36, velocity=100, duration=0.5)
        self.synth.trigger_note(midi_note=43, velocity=100, duration=0.5)

        chunk = self.synth.render_chunk(frames=2048, sample_rate=44100)
        self.assertEqual(chunk.shape, (2048, 2), "Saída estéreo esperada com forma (2048, 2)")
        self.assertLessEqual(np.max(np.abs(chunk)), 1.0, "Mixagem de vozes causou clipping")

        # 3. Teste de volume zero (mudo)
        self.synth.volume = 0.0
        silent_chunk = self.synth.render_chunk(frames=1024, sample_rate=44100)
        self.assertTrue(np.all(silent_chunk == 0.0), "Volume 0.0 deve produzir silêncio estrito")

    def test_11b_attack_has_no_deliberate_fifth_harmonic_tone(self):
        """O ataque de D não pode carregar um seno forte em 5*f (F#)."""
        midi_d = 38
        sr = 44100
        audio = BassSynthesizer(sample_rate=sr).synthesize_note_to_array(
            midi_d, duration=.08, sr=sr)
        attack = audio[:int(.012 * sr)]
        t = np.arange(len(attack), dtype=np.float32) / sr
        fundamental = abs(np.dot(attack, np.sin(2 * np.pi * 73.416 * t)))
        fifth = abs(np.dot(attack, np.sin(2 * np.pi * 5 * 73.416 * t)))
        self.assertLess(fifth, fundamental * .45)

    # -------------------------------------------------------------
    # 12. Teste integrado com áudio real: violao_teste_120bpm.wav
    # -------------------------------------------------------------
    def test_12_integrated_acoustic_guitar(self):
        """TESTE 12 — Valida que o baixista acompanha os acordes detectados em violao_teste_120bpm.wav."""
        wav_path = os.path.abspath("violao_teste_120bpm.wav")
        if not os.path.exists(wav_path):
            generate_acoustic_guitar_sample(wav_path, bpm=120.0)

        source = FileAudioSource(wav_path)
        analyzer = AudioAnalyzer()
        sr = source.get_sample_rate()

        # Pré-análise de andamento
        bpm = analyzer.pre_analyze_track(source.mono_data, sr)
        self.assertGreater(bpm, 0.0)

        # Simula execução por 10 segundos
        played_bass_notes = []
        for t in np.arange(0.1, 10.0, 0.05):
            frame_idx = int(t * sr)
            chunk = source.get_chunk_at(frame_idx, 4096)
            ctx = analyzer.analyze_chunk(chunk, sr, float(t))

            # Verifica se o BassPlayer disparou eventos
            cur_ev = analyzer.bass_player.current_event
            if cur_ev is not None and cur_ev not in played_bass_notes:
                played_bass_notes.append(cur_ev)

        # O baixista deve ter executado notas correspondentes à progressão do violão (C, G, Am, F)
        self.assertGreater(len(played_bass_notes), 0, "O baixista deve ter tocado eventos durante a faixa")
        unique_note_letters = {ev.note[0] for ev in played_bass_notes}
        self.assertTrue(any(letter in unique_note_letters for letter in ["C", "G", "A", "F"]),
                        f"O baixista deve tocar as raízes da harmonia do violão, tocou: {unique_note_letters}")

        # Verifica método get_prediction
        pred_note, pred_timing = analyzer.bass_player.get_prediction(current_bar=2, current_beat=1)
        self.assertNotEqual(pred_note, "--")
        self.assertIn("Compasso", pred_timing)


if __name__ == "__main__":
    unittest.main()
