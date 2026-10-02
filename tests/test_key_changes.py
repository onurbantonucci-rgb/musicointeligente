"""Mudanças de tom na cifra ("Tom: X" no meio da música) e a linha do tempo de tons."""
import os
import tempfile
import tkinter as tk
import unittest

import numpy as np

from app.analysis.chart_harmony_aligner import BeatChroma, align_chart_to_beats, chord_template
from app.input.chart_parser import ChartParser
from app.music.chart_alignment import ChartAlignment
from app.music.key_timeline import KeyChangeEvent, KeyTimeline
from app.song.song import Song
from app.song.song_session import SongSession

CHART = """Tom: G
[Verso]
G          D
Santo é o Senhor
Em         C
Pra sempre reinará

[Refrão]
G       D
linha 1
Em      C
linha 2
Tom: A
A       E
linha 3
F#m     D
linha 4
"""

MANY = """Tom: G
[Verso]
G  D
Tom: A
A  E
Tom: Bb
Bb  F
Tom: C
C  G
"""


class TestKeyTimelineParsing(unittest.TestCase):
    def test_initial_key_without_modulation(self):            # TEST 15
        chart = ChartParser.parse("Tom: G\n[Verso]\nG  D\nlinha\n")
        self.assertEqual(chart.key, "G Major")
        self.assertEqual(chart.key_changes, [])

    def test_single_modulation(self):                          # TEST 16
        chart = ChartParser.parse(CHART)
        self.assertEqual(chart.key, "G Major")
        self.assertEqual([(e.from_key, e.to_key, e.semitone_delta) for e in chart.key_changes],
                         [("G Major", "A Major", 2)])
        self.assertEqual([r.key for r in chart.key_timeline.regions], ["G Major", "A Major"])

    def test_multiple_modulations_keep_order(self):            # TEST 17
        chart = ChartParser.parse(MANY)
        self.assertEqual([e.to_key for e in chart.key_changes], ["A Major", "Bb Major", "C Major"])
        self.assertEqual([e.from_key for e in chart.key_changes], ["G Major", "A Major", "Bb Major"])
        self.assertEqual([e.id for e in chart.key_changes], ["key_01", "key_02", "key_03"])
        self.assertEqual(ChartAlignment(chart).event_keys(),
                         ["G Major"] * 2 + ["A Major"] * 2 + ["Bb Major"] * 2 + ["C Major"] * 2)

    def test_change_inside_the_chorus_does_not_end_the_section(self):   # TEST 18
        chart = ChartParser.parse(CHART)
        self.assertEqual([s.name for s in chart.sections], ["Verso", "Refrão"])
        alignment = ChartAlignment(chart)
        types = alignment.event_section_types()
        keys = alignment.event_keys()
        chorus = [k for k, t in zip(keys, types) if t == "CHORUS"]
        self.assertEqual(chorus, ["G Major"] * 4 + ["A Major"] * 4)

    def test_new_key_starts_at_the_next_playable_chord(self):  # TEST 19
        chart = ChartParser.parse(CHART)
        alignment = ChartAlignment(chart)
        chords = alignment.event_chords()
        first_a = chords.index("A")
        self.assertEqual(alignment.key_at_event(first_a), "A Major")
        self.assertEqual(alignment.key_at_event(first_a - 1), "G Major")     # o C de antes
        self.assertEqual(chart.key_changes[0].anchor_bar, alignment.event_bar(first_a))
        self.assertEqual(alignment.next_key_change(first_a - 1), (first_a, "A Major"))

    def test_minor_keys_and_explicit_anchor(self):
        chart = ChartParser.parse("Tom: Em\n[Verso]\nEm  C\nTom: F# menor @ compasso 2 tempo 3\nG  D\n")
        self.assertEqual(chart.key, "E Minor")
        event = chart.key_changes[0]
        self.assertEqual((event.to_key, event.anchor_bar, event.anchor_beat, event.explicit_anchor),
                         ("F# Minor", 2, 3.0, True))
        self.assertEqual(chart.key_timeline.key_at(2, 1.0), "E Minor")
        self.assertEqual(chart.key_timeline.key_at(2, 3.0), "F# Minor")

    def test_key_change_survives_serialization(self):
        chart = ChartParser.parse(MANY)
        from app.music.chord_chart import ChordChart
        again = ChordChart.from_dict(chart.to_dict())
        self.assertEqual([e.to_key for e in again.key_changes], ["A Major", "Bb Major", "C Major"])

    def test_global_transposition_moves_every_modulation(self):   # TEST 24
        chart = ChartParser.parse(MANY)
        chart.transpose(2)
        self.assertEqual(chart.key, "A Major")
        self.assertEqual([(e.from_key, e.to_key) for e in chart.key_changes],
                         [("A Major", "B Major"), ("B Major", "C Major"), ("C Major", "D Major")])
        tom_lines = [l for l in chart.raw_text.split("\n") if l.startswith("Tom:")]
        self.assertEqual(tom_lines, ["Tom: A", "Tom: B", "Tom: C", "Tom: D"])
        self.assertEqual([e.to_key for e in ChartParser.parse(chart.raw_text).key_changes],
                         ["B Major", "C Major", "D Major"])

    def test_timeline_object(self):
        timeline = KeyTimeline("G Major", [KeyChangeEvent("k1", 3, anchor_bar=9, to_key="A Major")])
        self.assertEqual(timeline.key_at(8), "G Major")
        self.assertEqual(timeline.key_at(9), "A Major")
        self.assertEqual(timeline.next_change_after(5).to_key, "A Major")
        self.assertIsNone(timeline.next_change_after(9))
        self.assertEqual(timeline.transposed(1, use_flats=True).key_at(9), "Bb Major")


class TestSessionKnowsTheNextKey(unittest.TestCase):
    def session(self, chart=CHART):
        return SongSession(Song(title="Mod", chart_text=chart, bpm=68))

    def test_current_and_next_key_are_published_before_the_change(self):   # TEST 20
        session = self.session()
        chords = session.alignment.event_chords()
        before = chords.index("A") - 1
        session.position_estimator.relocate_to_event(before, "teste")
        session._after_external_relocation()
        ctx = session.context
        self.assertEqual(ctx.key, "G Major")
        self.assertEqual(ctx.next_expected_key, "A Major")
        self.assertTrue(ctx.key_change_pending)
        self.assertEqual(ctx.bars_until_key_change, 1)
        self.assertEqual(ctx.beats_until_key_change, 4.0)
        self.assertEqual(ctx.key_change_chord, "A")
        session.position_estimator.relocate_to_event(before + 1, "teste")
        session._after_external_relocation()
        self.assertEqual(session.context.key, "A Major")
        self.assertEqual(session.context.key_region_index, 1)
        self.assertFalse(session.context.key_change_pending)

    def test_chart_tracking_fields_unchanged_by_key_changes(self):    # TEST 27
        with_change = self.session()
        without = self.session(CHART.replace("Tom: A\n", ""))
        for session in (with_change, without):
            session.position_estimator.relocate_to_event(5, "teste")
            session._after_external_relocation()
        a, b = with_change.chart_position, without.chart_position
        for field in ("line_index", "event_index", "section_occurrence", "current_chord", "next_chord",
                      "section_name"):
            self.assertEqual(getattr(a, field), getattr(b, field), field)

    def test_capo_keeps_the_written_and_sounding_keys_coherent(self):    # TEST 25
        chart_text = "Tom: A\nCapotraste: 2ª casa\n" + CHART.split("\n", 1)[1].replace("Tom: A", "Tom: B")
        chart = ChartParser.parse(chart_text)
        alignment = ChartAlignment(chart)
        chords = alignment.event_chords()
        # Áudio tocando as FORMAS escritas (sem o capo soando): o tom do alinhamento
        # vira o da forma, 2 semitons abaixo do "Tom:" soante — também na modulação.
        beats = [c for c in chords for _ in range(4)]
        chroma = np.array([chord_template(c) for c in beats])
        bc = BeatChroma(beat_times=np.arange(len(beats) + 1) * 0.8, chroma=chroma,
                        energy=np.ones(len(beats)), duration=len(beats) * 0.8)
        timeline = align_chart_to_beats(bc, chords, 4, chart.capo_semitones,
                                        alignment.section_start_events(), key=chart.key,
                                        event_keys=alignment.event_keys())
        self.assertEqual(timeline.transposition, 0)
        self.assertEqual(timeline.key, "G Major")
        self.assertEqual(sorted(set(timeline.event_keys)), ["A Major", "G Major"])


class TestKeyChangeEditor(unittest.TestCase):
    def test_insert_key_change_goes_before_the_chord_of_the_cursor_line(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = tk.Tk()
        root.withdraw()
        try:
            from app.ui.main_window import MainWindow
            app = MainWindow(root, project_file_path=os.path.join(tmp.name, "project.json"))
            view = app.text_chart_view
            view.delete("1.0", "end")
            view.insert("1.0", "Tom: G\n[Refrão]\nG       D\nlinha 1\nA       E\nlinha 3\n")
            view.mark_set("insert", "6.2")            # cursor na letra "linha 3"
            line = app.insert_key_change("A")
            lines = view.get("1.0", "end-1c").split("\n")
            self.assertEqual(line, 5)
            self.assertEqual(lines[4], "Tom: A")
            self.assertEqual(lines[5].split(), ["A", "E"])
            self.assertTrue(app._has_unsaved_chart_edits)
            self.assertEqual(len(app._key_change_labels), 1)
            self.assertIn("G → A", app._key_change_labels[0].cget("text"))
            # A marcação visual não entra no texto da cifra.
            self.assertNotIn("MODULAÇÃO", view.get("1.0", "end-1c"))
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
