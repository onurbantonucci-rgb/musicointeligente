"""Regressões de persistência, transporte e integração musical."""
import os
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from app.audio.audio_loader import FileAudioSource
from app.audio.audio_player import AudioPlayer, PlaybackState
from app.instruments.bass_synthesizer import BassSynthesizer
from app.project.project_manager import ProjectManager
from app.song.song import Song
from app.song.song_session import SongSession
from app.ui.main_window import MainWindow


class TestPersistenceRegressions(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "project.json"
        self.manager = ProjectManager(str(self.path))

    def test_replace_failure_preserves_existing_file_and_cleans_temp(self):
        self.manager.save_project()
        original = self.path.read_bytes()
        self.manager.project.name = "New name"
        with patch("app.project.project_manager.os.replace", side_effect=OSError("locked")):
            with self.assertRaises(OSError):
                self.manager.save_project()
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_serialization_failure_preserves_existing_file_and_cleans_temp(self):
        self.manager.save_project()
        original = self.path.read_bytes()
        with patch("app.project.project_manager.json.dump", side_effect=TypeError("bad data")):
            with self.assertRaises(TypeError):
                self.manager.save_project()
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_saved_repertoire_and_chart_are_restored(self):
        song = self.manager.import_song(title="Saved song", chart_text="[Intro]\nC  G")
        self.manager.open_song(song)
        self.manager.update_song_chart(song.id, "[Intro]\nD  A")
        restored = ProjectManager(str(self.path))
        self.assertEqual(restored.active_session.song.title, "Saved song")
        self.assertEqual(restored.active_session.current_chord, "D")

    def test_autosave_errors_are_propagated_for_edit_and_transpose(self):
        song = self.manager.import_song(chart_text="Tom: C\n[Intro]\nC  G")
        with patch.object(self.manager, "save_project", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.manager.update_song_chart(song.id, "[Intro]\nD  A")
            with self.assertRaises(OSError):
                self.manager.transpose_song(song, "G Major")


class TestAudioRegressions(unittest.TestCase):
    def test_bass_is_audible_in_mono_and_stereo_without_clipping(self):
        for channels in (1, 2):
            with self.subTest(channels=channels):
                player = AudioPlayer()
                source = Mock()
                source.get_position.return_value = 0.0
                source.get_duration.return_value = 10.0
                source.get_sample_rate.return_value = 44100
                source.read_playback_chunk.return_value = np.zeros((2048, channels), dtype=np.float32)
                player._source = source
                player._state = PlaybackState.PLAYING
                synth = BassSynthesizer()
                synth.trigger_note(40)
                player.on_mix_audio = synth.render_chunk
                output = np.zeros((2048, channels), dtype=np.float32)
                player._audio_callback(output, 2048, None, None)
                self.assertGreater(float(np.max(np.abs(output))), 0.0)
                self.assertLessEqual(float(np.max(np.abs(output))), 1.0)

    def test_analysis_does_not_skip_audio_or_include_instrument_mix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "input.wav")
            sf.write(path, np.full((1000, 2), 0.1, dtype=np.float32), 1000, subtype="FLOAT")
            source = FileAudioSource(path)
            self.addCleanup(source.close)
            player = AudioPlayer()
            player._source = source
            player._state = PlaybackState.PLAYING
            player.on_audio_chunk = Mock()
            player.on_mix_audio = lambda frames, sr, pos: np.full((frames, 2), 0.2, dtype=np.float32)
            output = np.zeros((100, 2), dtype=np.float32)
            player._audio_callback(output, 100, None, None)
            self.assertAlmostEqual(source.get_position(), 0.1)
            np.testing.assert_allclose(player.on_audio_chunk.call_args.args[0], 0.1)
            np.testing.assert_allclose(output, 0.3)

    def test_unload_releases_source_and_stops_transport(self):
        player = AudioPlayer()
        source = Mock()
        player._source = source
        player._state = PlaybackState.PLAYING
        player.unload_source()
        source.close.assert_called_once()
        self.assertIsNone(player.source)
        self.assertEqual(player.state, PlaybackState.STOPPED)


class TestSessionRegressions(unittest.TestCase):
    def test_detected_tempo_updates_clock_and_context(self):
        session = SongSession(Song(bpm=120.0))
        session.update_audio_tick(4.0, detected_bpm=60.0)
        self.assertEqual(session.clock.bpm, 60.0)
        self.assertEqual(session.context.bpm, 60.0)
        self.assertEqual(session.current_bar, 2)

    def test_manual_tempo_override_is_preserved(self):
        song = Song(bpm=120.0)
        song.performance_settings.bpm_override = 90.0
        session = SongSession(song)
        session.update_audio_tick(4.0, detected_bpm=60.0)
        self.assertEqual(session.clock.bpm, 90.0)

    def test_next_section_name_and_navigation_are_separate(self):
        session = SongSession(Song(chart_text="[Intro]\nC  G\n[Refrão]\nAm  F"))
        self.assertIsInstance(session.next_section, str)
        self.assertNotEqual(session.next_section, "--")
        expected = session.next_section
        position = session.next_section_jump()
        self.assertEqual(position.section_name, expected)

    def test_chart_edit_updates_tempo_and_meter(self):
        session = SongSession(Song(chart_text="Tom: C\nBPM: 120\n[Intro]\nC  G"))
        session.update_chart_text("Tom: D\nBPM: 90\nCompasso: 3/4\n[Intro]\nD  A")
        self.assertEqual(session.clock.bpm, 90.0)
        self.assertEqual(session.clock.meter, "3/4")


class TestInterfaceRegressions(unittest.TestCase):
    def window(self):
        window = MainWindow.__new__(MainWindow)
        window.player = Mock()
        window.analyzer = Mock()
        window.project_manager = Mock()
        window.waveform_view = Mock()
        window.seek_var = Mock()
        window.seek_slider = Mock()
        for name in ("lbl_file_name", "lbl_meta_details", "lbl_time_cur", "lbl_time_total", "lbl_status_msg", "lbl_val_bpm", "lbl_sub_meter"):
            setattr(window, name, Mock())
        window._refresh_setlist_listbox = Mock()
        window._update_transport_state = Mock()
        window._load_audio_file = Mock()
        window._audio_generation = 1
        window._tempo_results = queue.SimpleQueue()
        window._current_source = object()
        return window

    def test_chart_only_switch_unloads_previous_audio(self):
        window = self.window()
        window._switch_to_song(Song(audio_path=""))
        window.player.unload_source.assert_called_once()
        self.assertIsNone(window._current_source)
        window.analyzer.reset_musical_history.assert_called_once()
        window._load_audio_file.assert_not_called()

    def test_stale_tempo_results_are_ignored(self):
        window = self.window()
        old_result = Mock()
        window._tempo_results.put((0, old_result, None))
        window._apply_pending_tempo_results()
        window.analyzer.apply_tempo_analysis.assert_not_called()

    def test_current_tempo_result_is_applied(self):
        window = self.window()
        result = Mock()
        window.analyzer.apply_tempo_analysis.return_value = 90.0
        window._tempo_results.put((1, result, None))
        window._apply_pending_tempo_results()
        window.analyzer.apply_tempo_analysis.assert_called_once_with(result)
        window.analyzer.set_beat_chroma.assert_called_once_with(None)

    def test_failed_save_keeps_editor_marked_unsaved(self):
        window = self.window()
        window.text_chart_view = Mock()
        window.text_chart_view.get.return_value = "[Intro]\nC  G"
        window._set_chart_status = Mock()
        window.root = Mock()
        window.project_manager.update_song_chart.side_effect = OSError("disk full")
        with patch("app.ui.main_window.messagebox.showerror") as error:
            window._on_save_chart_direct()
        self.assertTrue(window._has_unsaved_chart_edits)
        error.assert_called_once()
        window.lbl_status_msg.config.assert_not_called()
