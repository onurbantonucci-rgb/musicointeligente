"""Passagens só de acordes (reintro/instrumental) viram blocos próprios da cifra."""
import json
import os
import tempfile
import unittest

from app.input.chart_parser import ChartParser
from app.music.sectionizer import split_chord_only_passages
from app.project.project_manager import INSTRUMENTAL_SPLIT_FLAG, ProjectManager

CHART = """Tom: G

[Intro] G  D  Em  C
        G  D  Em  C

[Primeira Parte]
G                D
Tu cantavas sobre mim
Em            C
Tão bom pra mim

[Refrão]
G               D
Oh impressionante infinito
Em             C
E ousado amor de Deus

( G  D  Em  C )
( G  D  Em  C )

[Segunda Parte]
G             D
Inimigo eu fui
Em          C
Tão bom pra mim

Am   F
"""


def section_names(text):
    return [(s.name, s.section_type) for s in ChartParser.parse(text).sections]


class TestSplitChordOnlyPassages(unittest.TestCase):
    def test_intro_coming_back_after_the_chorus_becomes_a_reintro_block(self):
        text = split_chord_only_passages(CHART)
        self.assertEqual(section_names(text), [
            ("Intro", "INTRO"), ("Primeira Parte", "VERSE"), ("Refrão", "CHORUS"),
            ("Reintro", "INSTRUMENTAL"), ("Segunda Parte", "VERSE"),
            ("Instrumental", "INSTRUMENTAL")])
        chart = ChartParser.parse(text)
        reintro = chart.sections[3]
        self.assertEqual([c.symbol.original_symbol for c in reintro.chords],
                         ["G", "D", "Em", "C"] * 2)
        self.assertEqual(reintro.lyrics, [])
        chorus = chart.sections[2]
        self.assertEqual([c.symbol.original_symbol for c in chorus.chords], ["G", "D", "Em", "C"])

    def test_the_song_keeps_every_chord_in_order(self):
        def chords(text):
            return [c.symbol.original_symbol for s in ChartParser.parse(text).sections for c in s.chords]
        self.assertEqual(chords(split_chord_only_passages(CHART)), chords(CHART))

    def test_a_different_progression_is_an_instrumental(self):
        text = split_chord_only_passages(CHART)
        self.assertIn("[Instrumental]\nAm   F", text)

    def test_chords_right_under_the_last_lyric_stay_in_the_section(self):
        text = "[Verso]\nG        D\nTu cantavas sobre mim\n   C   D\n"
        self.assertEqual(split_chord_only_passages(text), text)

    def test_chords_between_sung_lines_are_not_split(self):
        text = "[Verso]\nG        D\nTu cantavas sobre mim\n\n( G  D )\n\nG     D\nTão bom pra mim\n"
        self.assertEqual(split_chord_only_passages(text), text)

    def test_running_twice_changes_nothing(self):
        once = split_chord_only_passages(CHART)
        self.assertEqual(split_chord_only_passages(once), once)

    def test_reintro_header_is_not_an_intro(self):
        self.assertEqual(ChartParser.parse_section_header("[Reintro]")[1], "INSTRUMENTAL")


class TestProjectSplitsPassages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "projeto.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_imported_chart_gets_the_blocks(self):
        manager = ProjectManager(self.path)
        song = manager.import_song(title="Ousado", chart_text=CHART)
        self.assertIn("[Reintro]", song.chart_text)
        names = [s["name"] for s in song.chart_data["sections"]]
        self.assertIn("Reintro", names)

    def test_songs_saved_before_get_split_once_on_load(self):
        manager = ProjectManager(self.path)
        song = manager.import_song(title="Ousado", chart_text="x")
        song.chart_text = CHART                      # como era salvo antes da separação
        song.metadata.pop(INSTRUMENTAL_SPLIT_FLAG)
        manager.save_project(self.path)

        reloaded = ProjectManager(self.path)
        song = reloaded.project.get_active_setlist().songs[0]
        self.assertIn("[Reintro]", song.chart_text)
        self.assertIn("Reintro", [s["name"] for s in song.chart_data["sections"]])

        # O usuário junta o trecho de volta no Refrão: não separamos de novo.
        reloaded.update_song_chart(song.id, CHART)
        reloaded.save_project(self.path)
        again = ProjectManager(self.path).project.get_active_setlist().songs[0]
        self.assertNotIn("[Reintro]", again.chart_text)
        with open(self.path, encoding="utf-8") as f:
            self.assertTrue(json.load(f)["setlists"][0]["songs"][0]["metadata"][INSTRUMENTAL_SPLIT_FLAG])


if __name__ == "__main__":
    unittest.main()
