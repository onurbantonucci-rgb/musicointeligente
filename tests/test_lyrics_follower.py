"""Letra cantada como segunda referência de posição na cifra (ao vivo)."""
import unittest

from app.analysis.lyrics_follower import LyricsEvidence, LyricsIndex, LyricsMatcher, normalize_word
from app.input.chart_parser import ChartParser
from app.music.musical_context import MusicalContext
from app.song.song import Song
from app.song.song_session import SongSession

CHART = """Tom: G
[Verso]
G                D
Tu cantavas sobre mim
Em            C
Tão bom pra mim
[Refrão]
G               D
Oh impressionante infinito
Em             C
E ousado amor de Deus
"""


class TestLyricsIndex(unittest.TestCase):
    def setUp(self):
        self.index = LyricsIndex.from_chart(ChartParser.parse(CHART))

    def test_lines_keep_leading_words_and_whole_words(self):
        texts = [line.text for line in self.index.lines]
        self.assertEqual(texts, ["tu cantavas sobre mim", "tao bom pra mim",
                                 "oh impressionante infinito", "e ousado amor de deus"])

    def test_each_word_knows_its_bar(self):
        first = self.index.lines[0].words
        self.assertEqual([(w.word, w.bar) for w in first],
                         [("tu", 1), ("cantavas", 1), ("sobre", 2), ("mim", 2)])
        chorus = self.index.lines[2]
        self.assertEqual(chorus.section_name, "Refrão")
        self.assertTrue(chorus.first_in_section)

    def test_normalization_ignores_accents_and_punctuation(self):
        self.assertEqual(normalize_word("Tão,"), "tao")


class TestLyricsMatcher(unittest.TestCase):
    def setUp(self):
        self.index = LyricsIndex.from_chart(ChartParser.parse(CHART))

    def hear(self, words, start=10.0):
        matcher = LyricsMatcher(self.index)
        for i, word in enumerate(words):
            matcher.hear(word, start + 0.3 * i)
        return matcher.best()

    def test_finds_the_sung_line_and_bar(self):
        evidence = self.hear(["ousado", "amor", "de", "deus"])
        self.assertEqual(evidence.line.text, "e ousado amor de deus")
        self.assertEqual(evidence.bar, 8)          # refrão = compassos 5..8; "Deus" sob o C

    def test_tolerates_a_misrecognized_word(self):
        evidence = self.hear(["oh", "impressionante", "xyz", "infinito"])
        self.assertEqual(evidence.line.section_name, "Refrão")

    def test_too_few_words_is_not_evidence(self):
        self.assertIsNone(self.hear(["mim", "pra"]))


class TestSessionUsesLyrics(unittest.TestCase):
    def started_session(self):
        session = SongSession(Song(title="Letra", chart_text=CHART, bpm=120))
        for t in (0.0, 0.2, 0.4):
            session.update_audio_tick(t, detected_chord="G", detected_confidence=.9,
                                      source_context=MusicalContext(audio_activity=.1,
                                                                    smoothed_detected_chord="G",
                                                                    chord_start_time=0.0))
        self.assertTrue(session.music_started)
        return session

    def evidence(self, session, text, time):
        index = LyricsIndex.from_chart(session.chart)
        matcher = LyricsMatcher(index)
        for i, word in enumerate(text.split()):
            matcher.hear(word, time - 0.3 * (len(text.split()) - 1 - i))
        return matcher.best()

    def test_lyrics_ahead_of_the_cursor_move_the_chart_forward(self):
        session = self.started_session()
        self.assertEqual(session.current_bar, 1)
        evidence = self.evidence(session, "e ousado amor de deus", 0.5)
        self.assertEqual(session.apply_lyrics_evidence(evidence, 0.5), "corrigida")
        # Sem o acorde de "Deus" (C) soando, fica no compasso anterior (Em): o baixo
        # não troca antes de ouvir a troca.
        self.assertEqual(session.current_bar, 7)
        self.assertEqual(session.current_section, "Refrão")
        self.assertIn("Letra corrigiu", session.lyrics_status)

    def test_the_chord_heard_now_picks_the_exact_bar(self):
        session = self.started_session()
        session.update_audio_tick(0.45, detected_chord="C", detected_confidence=.9,
                                  source_context=MusicalContext(audio_activity=.1,
                                                                smoothed_detected_chord="C",
                                                                chord_start_time=0.42))
        evidence = self.evidence(session, "e ousado amor de deus", 0.5)
        session.apply_lyrics_evidence(evidence, 0.5)
        self.assertEqual(session.current_bar, 8)

    def test_lyrics_matching_the_cursor_only_confirm(self):
        session = self.started_session()
        evidence = self.evidence(session, "tu cantavas sobre", 0.5)
        self.assertEqual(session.apply_lyrics_evidence(evidence, 0.5), "confirmada")
        self.assertEqual(session.current_bar, 1)

    def test_lyrics_never_start_the_song(self):
        session = SongSession(Song(title="Letra", chart_text=CHART, bpm=120))
        evidence = self.evidence(session, "e ousado amor de deus", 0.5)
        self.assertEqual(session.apply_lyrics_evidence(evidence, 0.5), "ignorada")
        self.assertFalse(session.music_started)

    def test_lyrics_do_not_pull_the_chart_back_into_a_section_middle(self):
        session = self.started_session()
        forward = self.evidence(session, "e ousado amor de deus", 0.5)
        session.apply_lyrics_evidence(forward, 0.5)
        behind = self.evidence(session, "tao bom pra mim", 0.6)     # 2ª linha do verso (não é início)
        self.assertEqual(session.apply_lyrics_evidence(behind, 0.6), "ignorada")
        self.assertEqual(session.current_bar, 7)

    def test_section_start_line_allows_a_repeat(self):
        session = self.started_session()
        forward = self.evidence(session, "e ousado amor de deus", 0.5)
        session.apply_lyrics_evidence(forward, 0.5)
        repeat = LyricsEvidence(time=0.6, bar=1, line=LyricsIndex.from_chart(session.chart).lines[0],
                                matched=4, confidence=0.9)
        self.assertEqual(session.apply_lyrics_evidence(repeat, 0.6), "repetição")
        self.assertEqual(session.current_section, "Verso")


SAME_CHORDS = """Tom: G
[Verso 1]
G                D
Tu cantavas sobre mim
Em            C
Tão bom pra mim
[Verso 2]
G                D
Inimigo eu fui
Em            C
Mas Teu amor lutou
"""


class TestLyricsSeparatesSameChords(unittest.TestCase):
    def evidence(self, session, text, time):
        matcher = LyricsMatcher(LyricsIndex.from_chart(session.chart))
        for i, word in enumerate(text.split()):
            matcher.hear(word, time - 0.3 * (len(text.split()) - 1 - i))
        return matcher.best()

    def test_same_progression_different_lyrics(self):                  # TEST 28
        session = SongSession(Song(title="Letra", chart_text=SAME_CHORDS, bpm=120))
        for t in (0.0, 0.2, 0.4):
            session.update_audio_tick(t, detected_chord="G", detected_confidence=.9,
                                      source_context=MusicalContext(audio_activity=.1,
                                                                    smoothed_detected_chord="G",
                                                                    chord_start_time=0.0))
        self.assertEqual(session.current_section, "Verso 1")
        evidence = self.evidence(session, "mas teu amor lutou", 0.5)
        self.assertEqual(evidence.line.section_name, "Verso 2")
        session.apply_lyrics_evidence(evidence, 0.5)
        self.assertEqual(session.current_section, "Verso 2")

    def test_long_jump_needs_the_first_line_of_another_section(self):
        chart = "Tom: G\n[Verso]\n" + "G  D  Em  C\nla la la\n" * 6 + \
                "[Ponte]\nG                D\nTraz luz para as sombras\nEm       C\nEscala montanhas\n"
        session = SongSession(Song(title="Ponte", chart_text=chart, bpm=120))
        for t in (0.0, 0.2, 0.4):
            session.update_audio_tick(t, detected_chord="G", detected_confidence=.9,
                                      source_context=MusicalContext(audio_activity=.1,
                                                                    smoothed_detected_chord="G",
                                                                    chord_start_time=0.0))
        evidence = self.evidence(session, "traz luz para as sombras", 0.5)
        self.assertTrue(evidence.line.first_in_section)
        self.assertEqual(session.apply_lyrics_evidence(evidence, 0.5), "corrigida")
        self.assertEqual(session.current_section, "Ponte")


class TestLyricView(unittest.TestCase):
    def test_shows_the_line_being_sung_and_the_piece_over_the_current_chord(self):
        session = SongSession(Song(title="Letra", chart_text=CHART, bpm=120))
        lines = CHART.split("\n")
        expected = {1: ("Tu cantavas sobre mim", "Tu cantavas"), 2: ("Tu cantavas sobre mim", "sobre mim"),
                    6: ("Oh impressionante infinito", "impressionante infinito"),
                    8: ("E ousado amor de Deus", "de Deus")}
        for bar, (line, piece) in expected.items():
            session.position_estimator.relocate_to_bar(bar, "teste")
            session._after_external_relocation()
            view = session.lyric_view()
            self.assertEqual((view.state, view.line_text, view.segment), ("cantando", line, piece))
            self.assertEqual(lines[view.line_number - 1][view.segment_start:
                                                         view.segment_start + len(piece)], piece)
        self.assertEqual(session.lyric_view().next_line, "")
        session.position_estimator.relocate_to_bar(1, "teste")
        session._after_external_relocation()
        self.assertEqual(session.lyric_view().next_line, "Tão bom pra mim")


class TestLyricsIsOptional(unittest.TestCase):
    def test_lyrics_following_is_off_by_default_and_loads_nothing(self):
        import numpy as np
        from app.analysis.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        self.assertFalse(analyzer.lyrics_following)
        session = SongSession(Song(title="Letra", chart_text=CHART, bpm=120))
        analyzer.analyze_chunk(np.zeros(4096, dtype=np.float32), 44100, 0.0, session=session)
        self.assertFalse(analyzer.lyrics_loading)
        self.assertFalse(analyzer.lyrics_available)

    def test_lyrics_following_can_be_turned_on(self):
        from app.analysis.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        analyzer.set_lyrics_following(True)
        self.assertTrue(analyzer.lyrics_following)


if __name__ == "__main__":
    unittest.main()
