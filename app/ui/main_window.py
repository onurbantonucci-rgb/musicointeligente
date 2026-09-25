"""Janela principal do Virtual Band AI (v0.1-Completo).

Dashboard interativo local desenvolvido com Tkinter e TTK com estética profissional escura:
- Controles de reprodução (Play, Pause, Stop, Seek, Waveform)
- Exibição em tempo real do MusicalContext (Nota, Acorde, Tonalidade, BPM, Compasso, Beat)
- Métricas temporais: Tempo no acorde, Acorde anterior, Tempo no tom, Tom anterior
- Relógio Musical sincronizado (MusicalClock) com LEDs rítmicos pulsantes
- Medição de Latência Quadripartida (Processamento, Janela, Estabilização, Estimada)
- Tabela e Timeline do Histórico de Acordes (ChordHistory) e Tonalidade (KeyHistory)
- Cromagrama com as 12 notas temperadas
"""

import os
import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional
import numpy as np

from app.audio.audio_loader import FileAudioSource
from app.audio.audio_player import AudioPlayer, PlaybackState
from app.analysis.audio_analyzer import AudioAnalyzer
from app.analysis.tempo_detector import OnsetTempoDetector
from app.music.musical_context import MusicalContext
from app.ui.waveform_view import WaveformView
from app.ui.chroma_view import ChromaView
from app.ui.edit_history import ChartEditState, EditHistory
from app.utils.audio_generator import generate_test_song, generate_acoustic_guitar_sample
from app.utils.timing import LatencyTracker
from app.instruments import BassPatternType, BassNoteValue, BassHarmonySource
from app.music.theory import midi_to_hz
from app.project.project_manager import ProjectManager, DEFAULT_PROJECT_PATH
from app.song.song import Song
from app.instruments.registry import VirtualPlayerRegistry
from app.input.chart_parser import ChartParser
from app.music.sectionizer import SECTION_COLORS
from app.music.transposition import parse_key_root_mode
from app.input.chart_sources import (
    TextChartSource,
    DocxChartSource,
    ImageChartSource,
    PdfChartSource,
    RawChartDocument
)



class MainWindow:
    """Janela principal da aplicação com tema escuro e dashboard de contexto musical."""

    BASS_PATTERNS = [
        "FUNDAMENTAIS",
        "AUTO",
        "ROOT + FIFTH",
        "ROOT + FIFTH + OCTAVE",
        "SUSTAINED"
    ]

    PATTERN_MAP = {
        "FUNDAMENTAIS": BassPatternType.FUNDAMENTALS,
        "AUTO": BassPatternType.AUTO,
        "ROOT": BassPatternType.ROOT,
        "ROOT + FIFTH": BassPatternType.ROOT_FIFTH,
        "ROOT + FIFTH + OCTAVE": BassPatternType.ROOT_FIFTH_OCTAVE,
        "SUSTAINED": BassPatternType.SUSTAINED,
    }

    BASS_NOTE_VALUES = {
        "Semibreve — a cada 4 tempos": BassNoteValue.WHOLE,
        "Mínima — a cada 2 tempos": BassNoteValue.HALF,
        "Semínima — a cada tempo": BassNoteValue.QUARTER,
        "Colcheia — 2x por tempo": BassNoteValue.EIGHTH,
        "Semicolcheia — 4x por tempo": BassNoteValue.SIXTEENTH,
    }

    BASS_HARMONY_SOURCES = {
        "Seguir sempre a cifra": BassHarmonySource.CHART,
        "Cifra + confirmação do áudio": BassHarmonySource.FOLLOW,
    }

    ANALYSIS_CHANNELS = (
        "Automático",
        "Mistura estéreo",
        "Canal esquerdo",
        "Canal direito",
    )

    BPM_SOURCES = {
        "BPM da música (áudio)": "AUDIO",
        "BPM informado na cifra": "CHART",
    }
    KEY_SOURCES = {
        "Tom informado na cifra": "CHART",
        "Tom detectado no áudio": "AUDIO",
    }

    def __init__(self, root: tk.Tk, project_file_path: Optional[str] = None):
        self.root = root
        self.root.title("Virtual Band AI — Musician Play-Along & Banda Virtual (v0.4)")
        self.root.geometry("1180x820")
        self.root.minsize(1000, 640)
        self.root.configure(bg="#11141a")

        # Motor de Áudio e Analisador com Contexto e Memória Musical
        self.player = AudioPlayer()
        self.analyzer = AudioAnalyzer()
        self.analyzer.bass_player.pattern = BassPatternType.FUNDAMENTALS
        self.analyzer.bass_player.note_value = BassNoteValue.HALF
        self.analyzer.bass_player.harmony_source = BassHarmonySource.CHART
        self._tempo_results = queue.SimpleQueue()
        self._audio_generation = 0
        self._high_res_active = False  # modo de alta resolução harmônica (ligado no play-along)
        self.context = self.analyzer.context
        self.latency_tracker = LatencyTracker()

        # Conectar mixagem de áudio com o sintetizador de contrabaixo local (Zero Latency)
        self.player.on_mix_audio = self.analyzer.bass_player.synthesizer.render_chunk

        # Gerenciador de Projetos, Setlist e Músico Play-Along (v0.4)
        self.project_manager = ProjectManager(project_file_path or DEFAULT_PROJECT_PATH)
        self.virtual_players = VirtualPlayerRegistry(bass_player=self.analyzer.bass_player)
        self._follow_mode_enabled = True
        self._has_unsaved_chart_edits = False
        self._chart_edit_history = EditHistory(limit=100)

        # Flags e rastreadores de interface
        self._is_user_dragging_slider = False
        self._current_source: FileAudioSource = None
        self._recommended_analysis_channel: Optional[int] = None
        self._last_rendered_events_count = 0
        self._last_rendered_bass_events_count = 0
        self._update_counter = 0

        # Configurar Estilos TTK
        self._setup_theme()
        self._init_default_project_content()

        # Construir Componentes Visuais
        self._build_header()
        self._build_file_section()
        self._build_transport_section()
        self._build_main_workspace()
        self._build_statusbar()

        # Vincular Atalhos de Teclado
        self._bind_shortcuts()

        # Trazer janela para o primeiro plano no Windows
        self.root.lift()
        self.root.focus_force()

        # Iniciar ciclo de atualização da interface (30 FPS thread-safe)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(33, self._ui_update_loop)

    def _setup_theme(self) -> None:
        """Configura paleta de cores escura e tipografia limpa para o TTK."""
        self.style = ttk.Style()
        self.style.theme_use("clam")

        bg_dark = "#11141a"
        card_bg = "#181d26"
        card_border = "#262f3e"
        text_primary = "#f0f4f8"
        text_secondary = "#8c9ba5"
        accent_blue = "#00b4d8"
        accent_green = "#00e676"

        self.style.configure(".", background=bg_dark, foreground=text_primary, font=("Segoe UI", 9))
        self.style.configure("TFrame", background=bg_dark)
        self.style.configure("Card.TFrame", background=card_bg, relief="solid", borderwidth=1)

        # Abas (Notebook)
        self.style.configure("TNotebook", background=bg_dark, borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#1e2430", foreground="#8c9ba5",
                             font=("Segoe UI", 9, "bold"), padding=[14, 5])
        self.style.map("TNotebook.Tab",
                       background=[("selected", "#181d26"), ("active", "#252d3d")],
                       foreground=[("selected", "#00d2ff"), ("active", "#ffffff")])

        # Botões
        self.style.configure("Primary.TButton", background="#0077b6", foreground="#ffffff",
                             font=("Segoe UI", 9, "bold"), padding=6, borderwidth=0)
        self.style.map("Primary.TButton",
                       background=[("active", "#0096c7"), ("disabled", "#2a3442")],
                       foreground=[("disabled", "#6c7a89")])

        self.style.configure("Success.TButton", background="#0d9488", foreground="#ffffff",
                             font=("Segoe UI", 9, "bold"), padding=(12, 6), borderwidth=0)
        self.style.map("Success.TButton",
                       background=[("active", "#14b8a6"), ("disabled", "#134e4a")],
                       foreground=[("disabled", "#6c7a89")])

        self.style.configure("Transport.TButton", background="#212836", foreground="#ffffff",
                             font=("Segoe UI", 9, "bold"), padding=(10, 6), borderwidth=0)
        self.style.map("Transport.TButton",
                       background=[("active", "#2c3649"), ("pressed", "#161b24"), ("disabled", "#191e28")],
                       foreground=[("disabled", "#5a6677")])

        self.style.configure("Guitar.TButton", background="#8b5a2b", foreground="#ffffff",
                             font=("Segoe UI", 9, "bold"), padding=(10, 6), borderwidth=0)
        self.style.map("Guitar.TButton",
                       background=[("active", "#a06832"), ("disabled", "#3a2818")],
                       foreground=[("disabled", "#6c7a89")])

        self.style.configure("Play.TButton", background="#107c41", foreground="#ffffff",
                             font=("Segoe UI", 10, "bold"), padding=(16, 6), borderwidth=0)
        self.style.map("Play.TButton",
                       background=[("active", "#139c52"), ("disabled", "#193324")],
                       foreground=[("disabled", "#5a6677")])

        # Treeview (Tabela do Histórico de Acordes)
        self.style.configure(
            "Treeview",
            background="#141820",
            foreground="#f0f4f8",
            fieldbackground="#141820",
            rowheight=22,
            font=("Segoe UI", 8),
            borderwidth=0
        )
        self.style.configure(
            "Treeview.Heading",
            background="#1e2430",
            foreground="#00d2ff",
            font=("Segoe UI", 8, "bold"),
            borderwidth=1
        )
        self.style.map("Treeview", background=[("selected", "#0077b6")], foreground=[("selected", "#ffffff")])

        # Slider de busca (Scale)
        self.style.configure("Seek.Horizontal.TScale", background=bg_dark, troughcolor="#202735")


    def _build_header(self) -> None:
        """Cabeçalho superior da aplicação."""
        header_frame = tk.Frame(self.root, bg="#181d26", height=44, padx=16, pady=6)
        header_frame.pack(fill="x", side="top")

        title_lbl = tk.Label(header_frame, text="VIRTUAL BAND AI", bg="#181d26",
                             fg="#00d2ff", font=("Segoe UI", 13, "bold"))
        title_lbl.pack(side="left")

        sub_lbl = tk.Label(header_frame, text=" |  Músico Play-Along & Banda Virtual Autônoma",
                           bg="#181d26", fg="#8c9ba5", font=("Segoe UI", 9))
        sub_lbl.pack(side="left", padx=4)

        status_tag = tk.Label(header_frame, text="v0.4: Músico Play-Along Ativo", bg="#0f3424",
                              fg="#00e676", font=("Segoe UI", 8, "bold"), padx=8, pady=2)
        status_tag.pack(side="right")

    def _build_file_section(self) -> None:
        """Seção de seleção de arquivo, geradores e algoritmo de pitch."""
        file_frame = tk.Frame(self.root, bg="#181d26", bd=1, relief="solid", padx=14, pady=8)
        file_frame.pack(fill="x", padx=14, pady=(8, 4))

        top_row = tk.Frame(file_frame, bg="#181d26")
        top_row.pack(fill="x")

        # Botão Importar Cifra / Nova Música (DESTAQUE PRINCIPAL)
        self.btn_import_chart = ttk.Button(
            top_row,
            text="🎼 IMPORTAR CIFRA (TXT / DOCX / PDF / IMG / COLAR)",
            style="Success.TButton",
            command=self._on_new_song_dialog
        )
        self.btn_import_chart.pack(side="left", padx=(0, 8))

        # Botão Abrir Arquivo de Áudio
        self.btn_open = ttk.Button(top_row, text="📂 ABRIR ÁUDIO (WAV / MP3)",
                                   style="Primary.TButton", command=self._on_open_file)
        self.btn_open.pack(side="left", padx=(0, 8))

        # Botão Gerar Sintético
        self.btn_gen_test = ttk.Button(top_row, text="🎵 Sintético (C-G-Am-F)",
                                       style="Transport.TButton", command=self._on_generate_test_audio)
        self.btn_gen_test.pack(side="left", padx=(0, 8))

        # Botão Gerar Violão Acústico Realista
        self.btn_gen_guitar = ttk.Button(top_row, text="🎸 Violão Acústico",
                                         style="Guitar.TButton", command=self._on_generate_guitar_audio)
        self.btn_gen_guitar.pack(side="left")

        # Seletor de Algoritmo de Pitch
        algo_frame = tk.Frame(top_row, bg="#181d26")
        algo_frame.pack(side="right")

        tk.Label(algo_frame, text="Algoritmo Pitch:", bg="#181d26", fg="#8c9ba5",
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 6))

        self.combo_algo = ttk.Combobox(
            algo_frame,
            values=self.analyzer.get_available_algorithms(),
            state="readonly",
            width=15,
            font=("Segoe UI", 8)
        )
        self.combo_algo.set("Autocorrelação")
        self.combo_algo.pack(side="left")
        self.combo_algo.bind("<<ComboboxSelected>>", self._on_algo_changed)

        hearing_row = tk.Frame(file_frame, bg="#181d26")
        hearing_row.pack(fill="x", pady=(6, 0))
        tk.Label(hearing_row, text="Referência analisada:", bg="#181d26", fg="#8c9ba5",
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 6))
        self.combo_analysis_channel = ttk.Combobox(
            hearing_row, values=self.ANALYSIS_CHANNELS, state="readonly",
            width=18, font=("Segoe UI", 8))
        self.combo_analysis_channel.set("Automático")
        self.combo_analysis_channel.pack(side="left")
        self.combo_analysis_channel.bind("<<ComboboxSelected>>",
                                         self._on_analysis_channel_changed)

        # Linha de Metadados
        meta_row = tk.Frame(file_frame, bg="#181d26")
        meta_row.pack(fill="x", pady=(6, 0))

        self.lbl_file_name = tk.Label(meta_row, text="Nenhum arquivo carregado", bg="#181d26",
                                      fg="#ffffff", font=("Segoe UI", 9, "bold"))
        self.lbl_file_name.pack(side="left")

        self.lbl_meta_details = tk.Label(meta_row, text="Duração: --:--  |  Sample Rate: -- Hz  |  Canais: --",
                                         bg="#181d26", fg="#8c9ba5", font=("Segoe UI", 9))
        self.lbl_meta_details.pack(side="right")

    def _build_transport_section(self) -> None:
        """Controles de reprodução, pause, stop e slider de busca."""
        trans_frame = tk.Frame(self.root, bg="#141820", padx=14, pady=4)
        trans_frame.pack(fill="x", padx=14, pady=2)

        # Barra de Progresso / Seek Slider
        seek_row = tk.Frame(trans_frame, bg="#141820")
        seek_row.pack(fill="x", pady=(0, 4))

        self.lbl_time_cur = tk.Label(seek_row, text="00:00", bg="#141820", fg="#00d2ff",
                                     font=("Consolas", 10, "bold"), width=6)
        self.lbl_time_cur.pack(side="left")

        self.seek_var = tk.DoubleVar(value=0.0)
        self.seek_slider = ttk.Scale(seek_row, from_=0.0, to=100.0, orient="horizontal",
                                     variable=self.seek_var, style="Seek.Horizontal.TScale",
                                     command=self._on_slider_move)
        self.seek_slider.pack(side="left", fill="x", expand=True, padx=8)

        self.seek_slider.bind("<ButtonPress-1>", self._on_slider_press)
        self.seek_slider.bind("<ButtonRelease-1>", self._on_slider_release)

        self.lbl_time_total = tk.Label(seek_row, text="00:00", bg="#141820", fg="#8c9ba5",
                                       font=("Consolas", 10), width=6)
        self.lbl_time_total.pack(side="right")

        # Botões de Transporte
        btn_row = tk.Frame(trans_frame, bg="#141820")
        btn_row.pack(fill="x")

        self.btn_play = ttk.Button(btn_row, text="▶ PLAY", style="Play.TButton", command=self._on_play)
        self.btn_play.pack(side="left", padx=(0, 6))

        self.btn_pause = ttk.Button(btn_row, text="⏸ PAUSE", style="Transport.TButton", command=self._on_pause)
        self.btn_pause.pack(side="left", padx=6)

        self.btn_stop = ttk.Button(btn_row, text="■ STOP", style="Transport.TButton", command=self._on_stop)
        self.btn_stop.pack(side="left", padx=6)

        self.btn_rewind = ttk.Button(btn_row, text="⏮ INÍCIO", style="Transport.TButton", command=self._on_rewind)
        self.btn_rewind.pack(side="left", padx=6)

        self.lbl_state = tk.Label(btn_row, text="PARADO", bg="#141820", fg="#5a6677",
                                  font=("Segoe UI", 9, "bold"), padx=10)
        self.lbl_state.pack(side="right")

    def _build_waveform_section(self, parent: Optional[tk.Frame] = None) -> None:
        """Área gráfica da Waveform com Playhead dinâmico e Timeline Estrutural (v0.3)."""
        target = parent if parent is not None else self.root
        wf_container = tk.Frame(target, bg="#181d26", bd=1, relief="solid", padx=8, pady=4)
        wf_container.pack(fill="x", padx=10, pady=(6, 2))

        header_row = tk.Frame(wf_container, bg="#181d26")
        header_row.pack(fill="x", pady=(0, 2))

        tk.Label(header_row, text="FORMA DE ONDA & TIMELINE ESTRUTURAL (Clique para navegar)",
                 bg="#181d26", fg="#8c9ba5", font=("Segoe UI", 8, "bold")).pack(side="left")

        self.waveform_view = WaveformView(wf_container, height=65, bg_color="#12151c",
                                          wave_color="#00a8cc", playhead_color="#ff5252")
        self.waveform_view.pack(fill="x", expand=True)
        self.waveform_view.on_seek_requested = self._on_waveform_seek

        # Timeline Estrutural Dinâmica (v0.3)
        self.canvas_timeline = tk.Canvas(wf_container, height=22, bg="#12151c", highlightthickness=0)
        self.canvas_timeline.pack(fill="x", expand=True, pady=(2, 0))
        self.canvas_timeline.bind("<Button-1>", self._on_timeline_click)

    def _on_timeline_click(self, event: tk.Event) -> None:
        """Permite navegar diretamente clicando sobre qualquer ponto da timeline estrutural."""
        if self._current_source is None:
            return
        duration = self._current_source.get_duration()
        if duration <= 0:
            return
        w = self.canvas_timeline.winfo_width()
        if w <= 0:
            return
        target_time = max(0.0, min(duration, (event.x / w) * duration))
        self._on_waveform_seek(target_time)

    def _draw_structure_timeline(self, current_pos: float, total_duration: float) -> None:
        """Desenha a representação visual da timeline estrutural colorida."""
        self.canvas_timeline.delete("all")
        w = self.canvas_timeline.winfo_width()
        h = self.canvas_timeline.winfo_height()
        if w <= 10 or h <= 5:
            return

        # Fundo
        self.canvas_timeline.create_rectangle(0, 0, w, h, fill="#12151c", outline="")

        if total_duration <= 0:
            return

        sections = self.analyzer.structure_analyzer.structure.sections
        section_colors = {
            "INTRO": "#4a5568",
            "VERSE": "#2b6cb0",
            "PRE_CHORUS": "#d69e2e",
            "CHORUS": "#9b2c2c",
            "BRIDGE": "#805ad5",
            "SOLO": "#dd6b20",
            "INSTRUMENTAL": "#319795",
            "BREAK": "#4a5568",
            "OUTRO": "#2d3748",
            "UNKNOWN": "#1f2937"
        }

        # Desenhar blocos de seções
        for sec in sections:
            x1 = max(0.0, (sec.start_time / total_duration) * w)
            x2 = min(float(w), (sec.end_time / total_duration) * w)
            if x2 - x1 < 1.0:
                x2 = x1 + 1.0
            color = section_colors.get(sec.section_type, "#2b6cb0")
            self.canvas_timeline.create_rectangle(x1, 0, x2, h, fill=color, outline="#11141a")
            if (x2 - x1) > 28:
                txt = f"{sec.section_type[:5]} ({sec.label})"
                self.canvas_timeline.create_text((x1 + x2) / 2, h / 2, text=txt, fill="#ffffff", font=("Segoe UI", 6, "bold"))

        # Desenhar cursor de reprodução (Playhead)
        px = max(0.0, min(float(w), (current_pos / total_duration) * w))
        self.canvas_timeline.create_line(px, 0, px, h, fill="#00e676", width=2)


    def _build_analysis_dashboard(self, parent: Optional[tk.Frame] = None) -> None:
        """Painel de Métricas Musicais (Fases 11, 12 e 13)."""
        target = parent if parent is not None else self.root
        source_bar = tk.Frame(target, bg="#11141a")
        source_bar.pack(fill="x", padx=12, pady=(6, 0))
        tk.Label(source_bar, text="REFERÊNCIAS DA MÚSICA", bg="#11141a", fg="#00d2ff",
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 10))
        tk.Label(source_bar, text="BPM:", bg="#11141a", fg="#8c9ba5",
                 font=("Segoe UI", 8)).pack(side="left")
        self.combo_bpm_source = ttk.Combobox(
            source_bar, values=tuple(self.BPM_SOURCES), state="readonly", width=23,
            font=("Segoe UI", 8))
        self.combo_bpm_source.set("BPM da música (áudio)")
        self.combo_bpm_source.pack(side="left", padx=(4, 16))
        self.combo_bpm_source.bind("<<ComboboxSelected>>", self._on_bpm_source_changed)
        tk.Label(source_bar, text="Tom:", bg="#11141a", fg="#8c9ba5",
                 font=("Segoe UI", 8)).pack(side="left")
        self.combo_key_source = ttk.Combobox(
            source_bar, values=tuple(self.KEY_SOURCES), state="readonly", width=23,
            font=("Segoe UI", 8))
        self.combo_key_source.set("Tom informado na cifra")
        self.combo_key_source.pack(side="left", padx=4)
        self.combo_key_source.bind("<<ComboboxSelected>>", self._on_key_source_changed)
        dash_frame = tk.Frame(target, bg="#11141a")
        dash_frame.pack(fill="both", expand=True, padx=10, pady=4)

        dash_frame.columnconfigure((0, 1, 2, 3), weight=1, uniform="col")
        dash_frame.rowconfigure(0, weight=2)
        dash_frame.rowconfigure(1, weight=3)

        # -------------------------------------------------------------
        # 1. NOTA ATUAL
        # -------------------------------------------------------------
        card_pitch = tk.Frame(dash_frame, bg="#181d26", bd=1, relief="solid", padx=8, pady=6)
        card_pitch.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)
        tk.Label(card_pitch, text="PITCH BRUTO (MONOFÔNICO)", bg="#181d26", fg="#00d2ff", font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.lbl_val_note = tk.Label(card_pitch, text="--", bg="#181d26", fg="#5a6677", font=("Segoe UI", 24, "bold"))
        self.lbl_val_note.pack(pady=1)
        self.lbl_sub_note = tk.Label(card_pitch, text="0.0 Hz  |  Conf: 0%", bg="#181d26", fg="#8c9ba5", font=("Segoe UI", 8))
        self.lbl_sub_note.pack()
        self.lbl_cents = tk.Label(card_pitch, text="Afinação: --", bg="#181d26", fg="#5a6677", font=("Segoe UI", 8))
        self.lbl_cents.pack()

        # -------------------------------------------------------------
        # 2. ACORDE ATUAL & TEMPO NO ACORDE (Fase 11)
        # -------------------------------------------------------------
        card_chord = tk.Frame(dash_frame, bg="#181d26", bd=1, relief="solid", padx=8, pady=6)
        card_chord.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)
        tk.Label(card_chord, text="ACORDE ATUAL & TEMPO", bg="#181d26", fg="#00d2ff", font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.lbl_val_chord = tk.Label(card_chord, text="--", bg="#181d26", fg="#5a6677", font=("Segoe UI", 24, "bold"))
        self.lbl_val_chord.pack(pady=1)
        self.lbl_chord_duration = tk.Label(card_chord, text="Tempo no acorde: 0.00 s", bg="#181d26", fg="#00e676", font=("Segoe UI", 8, "bold"))
        self.lbl_chord_duration.pack()
        self.lbl_prev_chord = tk.Label(card_chord, text="Anterior: --  |  Conf: 0%", bg="#181d26", fg="#8c9ba5", font=("Segoe UI", 8))
        self.lbl_prev_chord.pack()
        self.lbl_chord_inv = tk.Label(card_chord, text="Fundamental (Raiz)", bg="#181d26", fg="#5a6677", font=("Segoe UI", 7))
        self.lbl_chord_inv.pack()
        self.lbl_chord_raw = tk.Label(card_chord, text="RAW: -- | Conf: 0%", bg="#181d26", fg="#8c9ba5", font=("Segoe UI", 7))
        self.lbl_chord_raw.pack(anchor="w")
        self.lbl_chord_candidate = tk.Label(card_chord, text="Candidato: --", bg="#181d26", fg="#8c9ba5", font=("Segoe UI", 7))
        self.lbl_chord_candidate.pack(anchor="w")
        self.lbl_chord_prior = tk.Label(card_chord, text="Cifra: -- | Prior: OFF", bg="#181d26", fg="#8c9ba5", font=("Segoe UI", 7))
        self.lbl_chord_prior.pack(anchor="w")
        self.lbl_harmonic_event = tk.Label(card_chord, text="Evento harmônico: -- | 0.0 beats",
                                            bg="#181d26", fg="#8c9ba5", font=("Segoe UI", 7))
        self.lbl_harmonic_event.pack(anchor="w")

        # -------------------------------------------------------------
        # 3. TONALIDADE ESTIMADA (KEY) (Fase 11)
        # -------------------------------------------------------------
        card_key = tk.Frame(dash_frame, bg="#181d26", bd=1, relief="solid", padx=8, pady=6)
        card_key.grid(row=0, column=2, sticky="nsew", padx=3, pady=3)
        self.lbl_key_source = tk.Label(card_key, text="TONALIDADE ESTIMADA", bg="#181d26",
                                       fg="#00d2ff", font=("Segoe UI", 8, "bold"))
        self.lbl_key_source.pack(anchor="w")
        self.lbl_val_key = tk.Label(card_key, text="--", bg="#181d26", fg="#5a6677", font=("Segoe UI", 20, "bold"))
        self.lbl_val_key.pack(pady=1)
        self.lbl_key_duration = tk.Label(card_key, text="Tempo no tom: 0.0 s", bg="#181d26", fg="#00e676", font=("Segoe UI", 8, "bold"))
        self.lbl_key_duration.pack()
        self.lbl_prev_key = tk.Label(card_key, text="Anterior: --  |  Conf: 0%", bg="#181d26", fg="#8c9ba5", font=("Segoe UI", 8))
        self.lbl_prev_key.pack()
        self.lbl_key_status = tk.Label(card_key, text="Adaptativo (Krumhansl-Schmuckler)", bg="#181d26", fg="#5a6677", font=("Segoe UI", 7))
        self.lbl_key_status.pack()

        # -------------------------------------------------------------
        # 4. RELÓGIO MUSICAL & BEAT (Fase 11)
        # -------------------------------------------------------------
        card_tempo = tk.Frame(dash_frame, bg="#181d26", bd=1, relief="solid", padx=8, pady=6)
        card_tempo.grid(row=0, column=3, sticky="nsew", padx=3, pady=3)
        tk.Label(card_tempo, text="RELÓGIO MUSICAL & BEAT", bg="#181d26", fg="#00d2ff", font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.lbl_val_bpm = tk.Label(card_tempo, text="-- BPM", bg="#181d26", fg="#5a6677", font=("Segoe UI", 20, "bold"))
        self.lbl_val_bpm.pack(pady=1)
        self.lbl_sub_meter = tk.Label(card_tempo, text="Compasso: 1  |  4/4", bg="#181d26", fg="#ffffff", font=("Segoe UI", 8, "bold"))
        self.lbl_sub_meter.pack()
        self.lbl_sub_beat = tk.Label(card_tempo, text="Tempo: -- / 4", bg="#181d26", fg="#8c9ba5", font=("Segoe UI", 8))
        self.lbl_sub_beat.pack()
        self.lbl_beat_leds = tk.Label(card_tempo, text="○   ○   ○   ○", bg="#181d26", fg="#5a6677",
                                      font=("Segoe UI", 11, "bold"))
        self.lbl_beat_leds.pack(pady=1)

        # -------------------------------------------------------------
        # 5. LATÊNCIA QUADRIPARTIDA & CROMAGRAMA (Colunas 0 e 1)
        # -------------------------------------------------------------
        left_bottom = tk.Frame(dash_frame, bg="#181d26", bd=1, relief="solid", padx=8, pady=6)
        left_bottom.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=3, pady=3)

        tk.Label(left_bottom, text="LATÊNCIA QUADRIPARTIDA (TEMPO REAL)",
                 bg="#181d26", fg="#00d2ff", font=("Segoe UI", 8, "bold")).pack(anchor="w")

        lat_grid = tk.Frame(left_bottom, bg="#141820", padx=6, pady=4)
        lat_grid.pack(fill="x", pady=2)

        self.lbl_lat_proc = tk.Label(lat_grid, text="Cálculo DSP: 0.0 ms", bg="#141820",
                                     fg="#00e676", font=("Segoe UI", 8, "bold"))
        self.lbl_lat_proc.pack(anchor="w")

        self.lbl_lat_win = tk.Label(lat_grid, text="Janela de Análise: 92.8 ms (4096 samples)",
                                    bg="#141820", fg="#8c9ba5", font=("Segoe UI", 8))
        self.lbl_lat_win.pack(anchor="w")

        self.lbl_lat_stab = tk.Label(lat_grid, text="Estabilização Temporal: 200.0 ms",
                                     bg="#141820", fg="#8c9ba5", font=("Segoe UI", 8))
        self.lbl_lat_stab.pack(anchor="w")

        self.lbl_lat_est = tk.Label(lat_grid, text="Latência Musical Estimada: ~247 ms",
                                    bg="#141820", fg="#ffd166", font=("Segoe UI", 8, "bold"))
        self.lbl_lat_est.pack(anchor="w")

        # Cromagrama
        tk.Label(left_bottom, text="CROMAGRAMA (12 NOTAS TEMPERADAS)",
                 bg="#181d26", fg="#00d2ff", font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(4, 0))

        self.chroma_view = ChromaView(left_bottom, height=45, bg_color="#141820", active_color="#00e676")
        self.chroma_view.pack(fill="both", expand=True, pady=2)
        self.lbl_active_notes = tk.Label(left_bottom, text="Notas ativas: --", bg="#181d26",
                                          fg="#00d2ff", font=("Consolas", 8), anchor="w")
        self.lbl_active_notes.pack(fill="x")

        # Histórico resumido
        hist_box = tk.Frame(left_bottom, bg="#141820", padx=6, pady=3)
        hist_box.pack(fill="x")
        self.lbl_hist_progression = tk.Label(hist_box, text="Aguardando reprodução...",
                                            bg="#141820", fg="#00d2ff", font=("Segoe UI", 8, "bold"))
        self.lbl_hist_progression.pack(anchor="w")

        self.lbl_hist_key = tk.Label(hist_box, text="Tonalidade: Aguardando...",
                                     bg="#141820", fg="#8c9ba5", font=("Segoe UI", 7))
        self.lbl_hist_key.pack(anchor="w")

        # -------------------------------------------------------------
        # 6. TABELA DO HISTÓRICO DE ACORDES (CHORD HISTORY - Fases 12 e 13)
        # -------------------------------------------------------------
        right_bottom = tk.Frame(dash_frame, bg="#181d26", bd=1, relief="solid", padx=8, pady=6)
        right_bottom.grid(row=1, column=2, columnspan=2, sticky="nsew", padx=3, pady=3)

        tbl_header = tk.Frame(right_bottom, bg="#181d26")
        tbl_header.pack(fill="x", pady=(0, 2))
        tk.Label(tbl_header, text="CHORD HISTORY (MUDANÇAS REAIS CONFIRMADAS)",
                 bg="#181d26", fg="#00d2ff", font=("Segoe UI", 8, "bold")).pack(side="left")

        # Treeview com Scrollbar
        tbl_container = tk.Frame(right_bottom, bg="#141820")
        tbl_container.pack(fill="both", expand=True)

        columns = ("time", "chord", "duration", "confidence")
        self.tree_history = ttk.Treeview(tbl_container, columns=columns, show="headings",
                                         height=4, selectmode="none")

        self.tree_history.heading("time", text="TIME")
        self.tree_history.heading("chord", text="CHORD")
        self.tree_history.heading("duration", text="DURATION")
        self.tree_history.heading("confidence", text="CONFIDENCE")

        self.tree_history.column("time", width=85, anchor="center")
        self.tree_history.column("chord", width=75, anchor="center")
        self.tree_history.column("duration", width=85, anchor="center")
        self.tree_history.column("confidence", width=85, anchor="center")

        scrollbar = ttk.Scrollbar(tbl_container, orient="vertical", command=self.tree_history.yview)
        self.tree_history.configure(yscrollcommand=scrollbar.set)

        self.tree_history.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_main_workspace(self) -> None:
        """Container com as abas principais de trabalho da aplicação (v0.4)."""
        self.main_notebook = ttk.Notebook(self.root)
        self.main_notebook.pack(fill="both", expand=True, padx=14, pady=(4, 2))

        # Compatibilidade com referências existentes a bottom_notebook
        self.bottom_notebook = self.main_notebook

        # Aba 1: Músico Play-Along & Cifra (v0.4) — ABA PRINCIPAL PADRÃO
        self.playalong_tab = tk.Frame(self.main_notebook, bg="#181d26")
        self.main_notebook.add(self.playalong_tab, text="  🎼 Músico Play-Along & Cifra  ")
        self._build_playalong_content(self.playalong_tab)

        # Aba 2: Laboratório de Áudio & DSP (v0.1)
        self.audio_lab_tab = tk.Frame(self.main_notebook, bg="#11141a")
        self.main_notebook.add(self.audio_lab_tab, text="  🔬 Laboratório de Áudio & DSP  ")
        self._build_waveform_section(self.audio_lab_tab)
        self._build_analysis_dashboard(self.audio_lab_tab)

        # Aba 3: Baixista Virtual (v0.2)
        self.bass_tab = tk.Frame(self.main_notebook, bg="#181d26")
        self.main_notebook.add(self.bass_tab, text="  🎸 Baixista Virtual (BassPlayer)  ")
        self._build_bass_player_content(self.bass_tab)

        # Aba 4: Estrutura Musical & Predição (v0.3)
        self.structure_tab = tk.Frame(self.main_notebook, bg="#181d26")
        self.main_notebook.add(self.structure_tab, text="  🏛️ Estrutura & Previsão  ")
        self._build_structure_prediction_content(self.structure_tab)

        # Seleciona Play-Along como aba inicial ativa
        self.main_notebook.select(self.playalong_tab)

    # Método de compatibilidade
    def _build_bottom_tabs_section(self) -> None:
        pass

    def _build_bass_player_content(self, parent: tk.Frame) -> None:
        """Painel de Controle e Visualização do Baixista Virtual (v0.2)."""
        self.bass_container = tk.Frame(parent, bg="#181d26", bd=0, padx=8, pady=4)
        self.bass_container.pack(fill="both", expand=True)

        # -------------------------------------------------------------
        # 1. Barra Superior de Controles do Baixo
        # -------------------------------------------------------------
        ctrl_bar = tk.Frame(self.bass_container, bg="#181d26")
        ctrl_bar.pack(fill="x", pady=(0, 4))

        # Título
        lbl_bass_title = tk.Label(
            ctrl_bar,
            text="🎸 BAIXISTA VIRTUAL (BASSPLAYER)",
            bg="#181d26",
            fg="#00e676",
            font=("Segoe UI", 9, "bold")
        )
        lbl_bass_title.pack(side="left", padx=(0, 10))

        # Botão Toggle Ativo / Silenciado
        self.btn_bass_toggle = tk.Button(
            ctrl_bar,
            text="🎸 BAIXO: ATIVADO",
            bg="#0f3424",
            fg="#00e676",
            activebackground="#144d34",
            activeforeground="#00e676",
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._toggle_bass
        )
        self.btn_bass_toggle.pack(side="left", padx=(0, 14))

        # Seletor de Padrão
        tk.Label(ctrl_bar, text="Padrão:", bg="#181d26", fg="#8c9ba5",
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 4))

        self.combo_bass_pattern = ttk.Combobox(
            ctrl_bar,
            values=self.BASS_PATTERNS,
            state="readonly",
            width=22,
            font=("Segoe UI", 8)
        )
        self.combo_bass_pattern.set("FUNDAMENTAIS")
        self.combo_bass_pattern.pack(side="left", padx=(0, 14))
        self.combo_bass_pattern.bind("<<ComboboxSelected>>", self._on_bass_pattern_changed)

        # Slider de Volume do Baixo
        tk.Label(ctrl_bar, text="Volume Baixo:", bg="#181d26", fg="#8c9ba5",
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 4))

        self.bass_vol_var = tk.DoubleVar(value=85.0)
        self.scale_bass_vol = ttk.Scale(
            ctrl_bar,
            from_=0.0,
            to=100.0,
            orient="horizontal",
            variable=self.bass_vol_var,
            command=self._on_bass_volume_changed,
            length=100
        )
        self.scale_bass_vol.pack(side="left", padx=(0, 6))

        self.lbl_bass_vol_val = tk.Label(ctrl_bar, text="85%", bg="#181d26", fg="#00d2ff",
                                         font=("Consolas", 8, "bold"), width=4)
        self.lbl_bass_vol_val.pack(side="left", padx=(0, 10))

        # Botão Modo Debug / Histórico de Decisões
        self.btn_bass_debug_toggle = tk.Button(
            ctrl_bar,
            text="📋 Log de Decisões (Debug) ▼",
            bg="#212836",
            fg="#f0f4f8",
            activebackground="#2c3649",
            activeforeground="#ffffff",
            font=("Segoe UI", 8),
            relief="flat",
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._toggle_bass_debug
        )
        self.btn_bass_debug_toggle.pack(side="right")

        rhythm_bar = tk.Frame(self.bass_container, bg="#181d26")
        rhythm_bar.pack(fill="x", pady=(0, 5))
        tk.Label(rhythm_bar, text="Fonte das notas:", bg="#181d26", fg="#8c9ba5",
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 6))
        self.combo_bass_harmony_source = ttk.Combobox(
            rhythm_bar, values=list(self.BASS_HARMONY_SOURCES), state="readonly",
            width=27, font=("Segoe UI", 8))
        self.combo_bass_harmony_source.set("Seguir sempre a cifra")
        self.combo_bass_harmony_source.pack(side="left", padx=(0, 12))
        self.combo_bass_harmony_source.bind("<<ComboboxSelected>>", self._on_bass_harmony_source_changed)
        tk.Label(rhythm_bar, text="Repetição da tônica:", bg="#181d26", fg="#8c9ba5",
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 6))
        self.combo_bass_note_value = ttk.Combobox(
            rhythm_bar, values=list(self.BASS_NOTE_VALUES), state="readonly",
            width=29, font=("Segoe UI", 8))
        self.combo_bass_note_value.set("Mínima — a cada 2 tempos")
        self.combo_bass_note_value.pack(side="left", padx=(0, 12))
        self.combo_bass_note_value.bind("<<ComboboxSelected>>", self._on_bass_note_value_changed)
        tk.Label(rhythm_bar, text="O BPM e a batida do músico definem os instantes.",
                 bg="#181d26", fg="#8c9ba5", font=("Segoe UI", 8)).pack(side="left")

        # -------------------------------------------------------------
        # 2. Quatro Cartões de Métricas do Baixista
        # -------------------------------------------------------------
        cards_frame = tk.Frame(self.bass_container, bg="#181d26")
        cards_frame.pack(fill="x", pady=2)
        cards_frame.columnconfigure((0, 1, 2, 3), weight=1, uniform="bass_cards")

        # Card 1: Nota Atual Tocada pelo Baixo
        c1 = tk.Frame(cards_frame, bg="#141820", bd=1, relief="solid", padx=8, pady=4)
        c1.grid(row=0, column=0, sticky="nsew", padx=2)
        tk.Label(c1, text="NOTA SOANDO DO BAIXO", bg="#141820", fg="#00e676", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_bass_note = tk.Label(c1, text="--", bg="#141820", fg="#5a6677", font=("Segoe UI", 18, "bold"))
        self.lbl_bass_note.pack()
        self.lbl_bass_freq = tk.Label(c1, text="MIDI --  |  -- Hz", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 7))
        self.lbl_bass_freq.pack()

        # Card 2: Motivo da Escolha (Fundamental, 5ª, 8ª, Inversão)
        c2 = tk.Frame(cards_frame, bg="#141820", bd=1, relief="solid", padx=8, pady=4)
        c2.grid(row=0, column=1, sticky="nsew", padx=2)
        tk.Label(c2, text="ORIGEM / TEMPO", bg="#141820", fg="#00d2ff", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_bass_reason = tk.Label(c2, text="Aguardando compasso...", bg="#141820", fg="#5a6677", font=("Segoe UI", 10, "bold"))
        self.lbl_bass_reason.pack(pady=2)
        self.lbl_bass_voice_leading = tk.Label(c2, text="Condução harmônica inteligente", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 7))
        self.lbl_bass_voice_leading.pack()
        self.lbl_bass_chart_source = tk.Label(c2, text="Cifra agora: --  |  Fonte: --",
                                               bg="#141820", fg="#8c9ba5",
                                               font=("Segoe UI", 7))
        self.lbl_bass_chart_source.pack()

        # Card 3: Padrão em Execução
        c3 = tk.Frame(cards_frame, bg="#141820", bd=1, relief="solid", padx=8, pady=4)
        c3.grid(row=0, column=2, sticky="nsew", padx=2)
        tk.Label(c3, text="PADRÃO EM EXECUÇÃO", bg="#141820", fg="#00d2ff", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_bass_pattern_active = tk.Label(c3, text="FUNDAMENTAIS", bg="#141820", fg="#ffffff", font=("Segoe UI", 11, "bold"))
        self.lbl_bass_pattern_active.pack(pady=2)
        self.lbl_bass_sub_pattern = tk.Label(c3, text="Adaptativo às mudanças", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 7))
        self.lbl_bass_sub_pattern.pack()

        # Card 4: Próxima Nota Prevista
        c4 = tk.Frame(cards_frame, bg="#141820", bd=1, relief="solid", padx=8, pady=4)
        c4.grid(row=0, column=3, sticky="nsew", padx=2)
        tk.Label(c4, text="PRÓXIMA NOTA PREVISTA", bg="#141820", fg="#ffd166", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_bass_next_note = tk.Label(c4, text="--", bg="#141820", fg="#5a6677", font=("Segoe UI", 11, "bold"))
        self.lbl_bass_next_note.pack(pady=2)
        self.lbl_bass_next_timing = tk.Label(c4, text="Aguardando harmonia", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 7))
        self.lbl_bass_next_timing.pack()

        # -------------------------------------------------------------
        # 3. Painel de Debug (Tabela de Decisões do Baixista)
        # -------------------------------------------------------------
        self.frame_bass_debug = tk.Frame(self.bass_container, bg="#141820", bd=1, relief="solid")
        self.frame_bass_debug.pack(fill="x", pady=(4, 0))
        self._bass_debug_visible = True

        debug_cols = ("bar", "beat", "chord", "note", "midi", "pattern", "reason")
        self.tree_bass_debug = ttk.Treeview(self.frame_bass_debug, columns=debug_cols, show="headings",
                                            height=3, selectmode="none")

        self.tree_bass_debug.heading("bar", text="COMPASSO")
        self.tree_bass_debug.heading("beat", text="TEMPO")
        self.tree_bass_debug.heading("chord", text="ACORDE")
        self.tree_bass_debug.heading("note", text="NOTA BAIXO")
        self.tree_bass_debug.heading("midi", text="MIDI / HZ")
        self.tree_bass_debug.heading("pattern", text="PADRÃO")
        self.tree_bass_debug.heading("reason", text="MOTIVO HARMÔNICO")

        self.tree_bass_debug.column("bar", width=75, anchor="center")
        self.tree_bass_debug.column("beat", width=65, anchor="center")
        self.tree_bass_debug.column("chord", width=75, anchor="center")
        self.tree_bass_debug.column("note", width=85, anchor="center")
        self.tree_bass_debug.column("midi", width=105, anchor="center")
        self.tree_bass_debug.column("pattern", width=120, anchor="center")
        self.tree_bass_debug.column("reason", width=260, anchor="w")

        debug_scroll = ttk.Scrollbar(self.frame_bass_debug, orient="vertical", command=self.tree_bass_debug.yview)
        self.tree_bass_debug.configure(yscrollcommand=debug_scroll.set)

        self.tree_bass_debug.pack(side="left", fill="both", expand=True)
        debug_scroll.pack(side="right", fill="y")

    def _build_structure_prediction_content(self, parent: tk.Frame) -> None:
        """Conteúdo da aba Estrutura & Previsão (v0.3)."""
        container = tk.Frame(parent, bg="#181d26", padx=8, pady=4)
        container.pack(fill="both", expand=True)

        # 1. Barra superior: Título e Botão de Exportar
        top_bar = tk.Frame(container, bg="#181d26")
        top_bar.pack(fill="x", pady=(0, 4))

        tk.Label(
            top_bar,
            text="🏛️ ESTRUTURA MUSICAL, MEMÓRIA DE PADRÕES & PREDIÇÃO ANTECIPADA (v0.3)",
            bg="#181d26",
            fg="#00d2ff",
            font=("Segoe UI", 9, "bold")
        ).pack(side="left")

        self.btn_export_report = tk.Button(
            top_bar,
            text="💾 Exportar Relatório JSON (Report)",
            bg="#0077b6",
            fg="#ffffff",
            activebackground="#0096c7",
            activeforeground="#ffffff",
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            padx=10,
            pady=2,
            cursor="hand2",
            command=self._on_export_structure_report
        )
        self.btn_export_report.pack(side="right")

        # 2. Quatro Cartões de Resumo
        cards_frame = tk.Frame(container, bg="#181d26")
        cards_frame.pack(fill="x", pady=2)
        cards_frame.columnconfigure((0, 1, 2, 3), weight=1, uniform="struct_cards")

        # Card 1: Seção Atual & Padrão
        sc1 = tk.Frame(cards_frame, bg="#141820", bd=1, relief="solid", padx=6, pady=3)
        sc1.grid(row=0, column=0, sticky="nsew", padx=2)
        tk.Label(sc1, text="SEÇÃO ATUAL & PADRÃO", bg="#141820", fg="#00d2ff", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_struct_cur_section = tk.Label(sc1, text="--", bg="#141820", fg="#5a6677", font=("Segoe UI", 16, "bold"))
        self.lbl_struct_cur_section.pack(pady=1)
        self.lbl_struct_progress = tk.Label(sc1, text="Progresso: --%  |  Comp. --", bg="#141820", fg="#00e676", font=("Segoe UI", 8))
        self.lbl_struct_progress.pack()

        # Card 2: Próxima Seção Prevista
        sc2 = tk.Frame(cards_frame, bg="#141820", bd=1, relief="solid", padx=6, pady=3)
        sc2.grid(row=0, column=1, sticky="nsew", padx=2)
        tk.Label(sc2, text="PRÓXIMA SEÇÃO (PREVISTA)", bg="#141820", fg="#ffd166", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_pred_next_section = tk.Label(sc2, text="--", bg="#141820", fg="#5a6677", font=("Segoe UI", 16, "bold"))
        self.lbl_pred_next_section.pack(pady=1)
        self.lbl_pred_timing = tk.Label(sc2, text="Em -- compassos", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 8))
        self.lbl_pred_timing.pack()

        # Card 3: Look-ahead Harmônico (Próximos Acordes)
        sc3 = tk.Frame(cards_frame, bg="#141820", bd=1, relief="solid", padx=6, pady=3)
        sc3.grid(row=0, column=2, sticky="nsew", padx=2)
        tk.Label(sc3, text="LOOK-AHEAD HARMÔNICO (1-4C)", bg="#141820", fg="#00e676", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_pred_lookahead = tk.Label(sc3, text="--", bg="#141820", fg="#5a6677", font=("Segoe UI", 13, "bold"))
        self.lbl_pred_lookahead.pack(pady=2)
        self.lbl_pred_reason = tk.Label(sc3, text="Aguardando harmonia", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 7))
        self.lbl_pred_reason.pack()

        # Card 4: Forma Global da Música
        sc4 = tk.Frame(cards_frame, bg="#141820", bd=1, relief="solid", padx=6, pady=3)
        sc4.grid(row=0, column=3, sticky="nsew", padx=2)
        tk.Label(sc4, text="FORMA GLOBAL DA MÚSICA", bg="#141820", fg="#c77dff", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_struct_form = tk.Label(sc4, text="--", bg="#141820", fg="#5a6677", font=("Segoe UI", 14, "bold"))
        self.lbl_struct_form.pack(pady=1)
        self.lbl_struct_summary = tk.Label(sc4, text="0 seções  |  0 padrões", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 7))
        self.lbl_struct_summary.pack()

        # 3. Tabelas Lado a Lado: Padrões Conhecidos & Matriz de Transições
        tables_frame = tk.Frame(container, bg="#181d26")
        tables_frame.pack(fill="both", expand=True, pady=(3, 0))
        tables_frame.columnconfigure((0, 1), weight=1, uniform="struct_tables")

        # Tabela 1 (Esquerda): Padrões Conhecidos
        left_tbl = tk.Frame(tables_frame, bg="#141820", bd=1, relief="solid", padx=4, pady=3)
        left_tbl.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        tk.Label(left_tbl, text="PADRÕES CONHECIDOS (PATTERN MEMORY)", bg="#141820", fg="#00d2ff",
                 font=("Segoe UI", 7, "bold")).pack(anchor="w", pady=(0, 2))

        pat_cols = ("id", "signature", "bars", "occurrences", "stability")
        self.tree_patterns = ttk.Treeview(left_tbl, columns=pat_cols, show="headings", height=3, selectmode="none")
        self.tree_patterns.heading("id", text="ID")
        self.tree_patterns.heading("signature", text="ASSINATURA DE ACORDES")
        self.tree_patterns.heading("bars", text="COMP.")
        self.tree_patterns.heading("occurrences", text="OCORR.")
        self.tree_patterns.heading("stability", text="ESTAB.")
        self.tree_patterns.column("id", width=45, anchor="center")
        self.tree_patterns.column("signature", width=200, anchor="w")
        self.tree_patterns.column("bars", width=45, anchor="center")
        self.tree_patterns.column("occurrences", width=50, anchor="center")
        self.tree_patterns.column("stability", width=55, anchor="center")

        pat_scroll = ttk.Scrollbar(left_tbl, orient="vertical", command=self.tree_patterns.yview)
        self.tree_patterns.configure(yscrollcommand=pat_scroll.set)
        self.tree_patterns.pack(side="left", fill="both", expand=True)
        pat_scroll.pack(side="right", fill="y")

        # Tabela 2 (Direita): Transições entre Seções
        right_tbl = tk.Frame(tables_frame, bg="#141820", bd=1, relief="solid", padx=4, pady=3)
        right_tbl.grid(row=0, column=1, sticky="nsew", padx=(2, 0))
        tk.Label(right_tbl, text="MATRIZ DE TRANSIÇÃO ENTRE SEÇÕES P(B|A)", bg="#141820", fg="#ffd166",
                 font=("Segoe UI", 7, "bold")).pack(anchor="w", pady=(0, 2))

        trans_cols = ("from_sec", "to_sec", "transition", "prob", "count")
        self.tree_transitions = ttk.Treeview(right_tbl, columns=trans_cols, show="headings", height=3, selectmode="none")
        self.tree_transitions.heading("from_sec", text="DE")
        self.tree_transitions.heading("to_sec", text="PARA")
        self.tree_transitions.heading("transition", text="PADRÃO (ORIGEM → DESTINO)")
        self.tree_transitions.heading("prob", text="PROBAB.")
        self.tree_transitions.heading("count", text="QTD")
        self.tree_transitions.column("from_sec", width=75, anchor="center")
        self.tree_transitions.column("to_sec", width=75, anchor="center")
        self.tree_transitions.column("transition", width=160, anchor="center")
        self.tree_transitions.column("prob", width=65, anchor="center")
        self.tree_transitions.column("count", width=45, anchor="center")

        trans_scroll = ttk.Scrollbar(right_tbl, orient="vertical", command=self.tree_transitions.yview)
        self.tree_transitions.configure(yscrollcommand=trans_scroll.set)
        self.tree_transitions.pack(side="left", fill="both", expand=True)
        trans_scroll.pack(side="right", fill="y")

    def _on_export_structure_report(self) -> None:
        """Salva o relatório estrutural JSON completo da música."""
        try:
            filepath = filedialog.asksaveasfilename(
                title="Exportar Relatório de Estrutura Musical (JSON)",
                defaultextension=".json",
                initialfile="music_structure_report.json",
                filetypes=[("Arquivos JSON (*.json)", "*.json"), ("Todos os Arquivos (*.*)", "*.*")]
            )
            if filepath:
                self.analyzer.export_structure_report(filepath)
                messagebox.showinfo(
                    "Exportação Concluída",
                    f"Relatório de estrutura exportado com sucesso para:\n{filepath}"
                )
                self.lbl_status_msg.config(text=f"Relatório exportado: {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Erro ao Exportar Relatório", f"Falha ao salvar relatório JSON:\n{e}")

    def _refresh_structure_tables(self) -> None:
        """Atualiza tabelas de padrões conhecidos e matriz de transições."""
        # 1. Padrões
        patterns = self.analyzer.pattern_memory.get_known_patterns()
        self.tree_patterns.delete(*self.tree_patterns.get_children())
        for pat in patterns:
            sig_txt = " - ".join(pat.chord_signatures[:6])
            if len(pat.chord_signatures) > 6:
                sig_txt += "..."
            stab_txt = f"{int(pat.stability_score * 100)}%"
            self.tree_patterns.insert(
                "", "end",
                values=(pat.id, sig_txt, pat.duration_bars, pat.occurrence_count, stab_txt)
            )

        # 2. Transições
        transitions = self.analyzer.pattern_memory.get_transitions()
        self.tree_transitions.delete(*self.tree_transitions.get_children())
        for tr in transitions:
            prob_txt = f"{tr.probability * 100:.0f}%"
            self.tree_transitions.insert(
                "", "end",
                values=(tr.from_section, tr.to_section, f"{tr.from_pattern_id} → {tr.to_pattern_id}", prob_txt, tr.count)
            )


    def _build_statusbar(self) -> None:
        """Barra de status inferior."""
        status_bar = tk.Frame(self.root, bg="#0d1015", height=24, padx=12)
        status_bar.pack(fill="x", side="bottom")

        self.lbl_status_msg = tk.Label(status_bar, text="Pronto. Carregue um áudio para analisar o Contexto Musical.",
                                       bg="#0d1015", fg="#8c9ba5", font=("Segoe UI", 8))
        self.lbl_status_msg.pack(side="left")

        self.lbl_status_driver = tk.Label(status_bar, text="PortAudio / SoundDevice (Local 100%)",
                                          bg="#0d1015", fg="#5a6677", font=("Segoe UI", 8))
        self.lbl_status_driver.pack(side="right")

    def _bind_shortcuts(self) -> None:
        """Atalhos de teclado para maior conveniência (com proteção de foco para campos de texto)."""
        def _is_text_focused():
            try:
                focused = self.root.focus_get()
                return isinstance(focused, (tk.Text, tk.Entry, ttk.Entry))
            except Exception:
                return False

        self.root.bind("<space>", lambda e: None if _is_text_focused() else self._toggle_play_pause())
        self.root.bind("<Escape>", lambda e: None if _is_text_focused() else self._on_stop())
        self.root.bind("<Home>", lambda e: None if _is_text_focused() else self._on_rewind())
        self.root.bind("<Left>", lambda e: None if _is_text_focused() else self._seek_relative(-3.0))
        self.root.bind("<Right>", lambda e: None if _is_text_focused() else self._seek_relative(+3.0))
        self.root.bind("<Control-s>", lambda e: self._on_save_chart_direct())
        self.root.bind("<Control-S>", lambda e: self._on_save_chart_direct())
        self.root.bind("<Control-z>", self._on_undo_shortcut)
        self.root.bind("<Control-Shift-z>", self._on_redo_shortcut)
        self.root.bind("<Control-Shift-Z>", self._on_redo_shortcut)

    def _on_open_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Selecionar Arquivo de Áudio",
            filetypes=[
                ("Arquivos de Áudio (*.wav;*.mp3)", "*.wav;*.mp3"),
                ("Áudio WAV (*.wav)", "*.wav"),
                ("Áudio MP3 (*.mp3)", "*.mp3"),
                ("Todos os Arquivos (*.*)", "*.*")
            ]
        )
        if file_path:
            self._load_audio_file(file_path)

    def _on_generate_test_audio(self) -> None:
        try:
            self.lbl_status_msg.config(text="Gerando áudio sintético padronizado C-G-Am-F (120 BPM)...")
            self.root.update_idletasks()
            test_file = generate_test_song("test_song_120bpm.wav", bpm=120.0)
            self._load_audio_file(test_file)
            self.lbl_status_msg.config(text=f"Áudio de teste carregado: {os.path.basename(test_file)}")
        except Exception as e:
            messagebox.showerror("Erro ao Gerar Áudio", f"Falha ao sintetizar áudio de teste: {e}")

    def _on_generate_guitar_audio(self) -> None:
        try:
            self.lbl_status_msg.config(text="Gerando áudio realista de Violão Acústico C-G-Am-F (120 BPM)...")
            self.root.update_idletasks()
            test_file = generate_acoustic_guitar_sample("violao_teste_120bpm.wav", bpm=120.0)
            self._load_audio_file(test_file)
            self.lbl_status_msg.config(text=f"Violão acústico carregado: {os.path.basename(test_file)}")
        except Exception as e:
            messagebox.showerror("Erro ao Gerar Áudio", f"Falha ao sintetizar violão: {e}")

    def _load_audio_file(self, file_path: str) -> None:
        """Carrega o arquivo e inicializa a análise assíncrona de andamento."""
        try:
            self.lbl_status_msg.config(text=f"Carregando {os.path.basename(file_path)}...")
            self.root.update_idletasks()

            source = FileAudioSource(file_path)
            self._recommended_analysis_channel = self.analyzer.recommend_analysis_channel(
                source.multichannel_data, source.get_sample_rate())
            self.combo_analysis_channel.set("Automático")
            source.set_analysis_channel(self._recommended_analysis_channel)
            self._audio_generation += 1
            self.analyzer.reset_musical_history()
            self._current_source = source
            self.player.load_source(source)

            # Limpar tabelas de histórico ao carregar nova faixa
            for item in self.tree_history.get_children():
                self.tree_history.delete(item)
            self._last_rendered_events_count = 0

            for item in self.tree_bass_debug.get_children():
                self.tree_bass_debug.delete(item)
            self._last_rendered_bass_events_count = 0

            for item in self.tree_patterns.get_children():
                self.tree_patterns.delete(item)
            for item in self.tree_transitions.get_children():
                self.tree_transitions.delete(item)
            self._draw_structure_timeline(0.0, 0.0)

            # Atualizar Metadados
            duration = source.get_duration()
            sr = source.get_sample_rate()
            channels_txt = "Estéreo (2 canais)" if source.channels == 2 else f"{source.channels} canal(is)"
            heard_txt = self._analysis_channel_description(source)

            self.lbl_file_name.config(text=source.name)
            self.lbl_meta_details.config(
                text=(f"Duração: {self._format_time(duration)}  |  Sample Rate: {sr} Hz  |  "
                      f"Canais: {channels_txt}  |  Ouvindo: {heard_txt}")
            )
            self.lbl_time_total.config(text=self._format_time(duration))
            self.seek_slider.config(to=duration)
            self.seek_var.set(0.0)

            # Extrair e desenhar Waveform
            mins, maxs = source.get_waveform_envelope(num_points=900)
            self.waveform_view.set_waveform_data(mins, maxs, duration)

            # Atualizar formato de análise
            self.analyzer.update_audio_format(
                sample_rate=sr, chunk_size=4096,
                output_latency_ms=self.player.estimated_output_latency_ms)
            self.latency_tracker.update_config(sample_rate=sr, chunk_size=4096)

            # Iniciar pré-análise de andamento em segundo plano sem travar a interface
            self.lbl_status_msg.config(text=f"Arquivo '{source.name}' carregado. Analisando andamento...")
            threading.Thread(target=self._run_background_tempo_analysis, args=(source.mono_data.copy(), sr, self._audio_generation), daemon=True).start()

            self._update_transport_state(PlaybackState.STOPPED)

        except Exception as e:
            messagebox.showerror("Erro ao Carregar Áudio", f"Não foi possível abrir o arquivo:\n{e}")
            self.lbl_status_msg.config(text="Erro ao carregar arquivo de áudio.")

    def _run_background_tempo_analysis(self, mono_data: np.ndarray, sr: int, generation: int) -> None:
        try:
            detector = OnsetTempoDetector()
            detector.analyze_audio(mono_data, sr)
            self._tempo_results.put((generation, detector))
        except Exception:
            logging.getLogger(__name__).exception("Falha na análise de andamento")

    def _apply_pending_tempo_results(self) -> None:
        while True:
            try:
                generation, detector = self._tempo_results.get_nowait()
            except queue.Empty:
                break
            if generation != self._audio_generation or self._current_source is None:
                continue
            bpm = self.analyzer.apply_tempo_analysis(detector)
            self.lbl_val_bpm.config(text=f"{bpm:.0f} BPM", fg="#ffffff")
            self.lbl_sub_meter.config(text=f"Compasso: 1  |  {self.analyzer.clock.meter}")
            self.lbl_status_msg.config(text=f"Andamento detectado: {bpm:.0f} BPM. Pronto para tocar.")

    def _on_play(self) -> None:
        if self._current_source is None:
            messagebox.showinfo("Aviso", "Por favor, abra um arquivo de áudio primeiro.")
            return
        session = self.project_manager.active_session
        if session is not None:
            starting_from_zero = not session.is_paused and self.player.get_position() <= 0.01
            if session.is_paused:
                session.resume()
            else:
                session.start()
            if starting_from_zero:
                sr = self._current_source.get_sample_rate()
                opening = self._current_source.get_chunk_at(0, min(4096, sr))
                if opening.size and float(np.sqrt(np.mean(np.square(opening.astype(np.float64))))) >= .005:
                    self.analyzer.bass_player.prepare_chart_start(session.context)
        self.player.play()
        self._update_transport_state(PlaybackState.PLAYING)

    def _on_pause(self) -> None:
        self.player.pause()
        if self.project_manager.active_session is not None:
            self.project_manager.active_session.pause()
        self._update_transport_state(PlaybackState.PAUSED)

    def _on_stop(self) -> None:
        self.player.stop()
        self.analyzer.reset_musical_history()
        if self.project_manager.active_session is not None:
            self.project_manager.active_session.stop()
        self.virtual_players.reset_all()
        self.waveform_view.set_playhead_position(0.0)
        self.seek_var.set(0.0)
        self.lbl_time_cur.config(text="00:00")
        self.lbl_struct_cur_section.config(text="--", fg="#5a6677")
        self.lbl_struct_progress.config(text="Progresso: --%  |  Comp. --", fg="#5a6677")
        self.lbl_pred_next_section.config(text="--", fg="#5a6677")
        self.lbl_pred_timing.config(text="Em -- compassos", fg="#5a6677")
        self.lbl_pred_lookahead.config(text="--", fg="#5a6677")
        self.lbl_pred_reason.config(text="Aguardando harmonia", fg="#5a6677")
        self.lbl_struct_form.config(text="--", fg="#5a6677")
        self.lbl_struct_summary.config(text="0 seções  |  0 padrões", fg="#5a6677")
        total_dur = self._current_source.get_duration() if self._current_source else 0.0
        self._draw_structure_timeline(0.0, total_dur)
        self._update_transport_state(PlaybackState.STOPPED)


    def _on_rewind(self) -> None:
        if self._current_source is not None:
            self.player.seek(0.0)
            self.waveform_view.set_playhead_position(0.0)
            self.seek_var.set(0.0)
            self.lbl_time_cur.config(text="00:00")
            if self.project_manager.active_session is not None:
                self.project_manager.active_session.restart()

    def _toggle_play_pause(self) -> None:
        if self.player.state == PlaybackState.PLAYING:
            self._on_pause()
        else:
            self._on_play()

    def _seek_relative(self, delta_seconds: float) -> None:
        if self._current_source is not None:
            current = self.player.get_position()
            duration = self.player.get_duration()
            target = max(0.0, min(current + delta_seconds, duration))
            self.player.seek(target)
            self.waveform_view.set_playhead_position(target)
            self.seek_var.set(target)

    def _on_waveform_seek(self, target_seconds: float) -> None:
        if self._current_source is not None:
            self.player.seek(target_seconds)
            self.seek_var.set(target_seconds)
            self.lbl_time_cur.config(text=self._format_time(target_seconds))

    def _on_slider_press(self, event) -> None:
        self._is_user_dragging_slider = True

    def _on_slider_move(self, val) -> None:
        try:
            sec = float(val)
            self.lbl_time_cur.config(text=self._format_time(sec))
        except ValueError:
            pass

    def _on_slider_release(self, event) -> None:
        self._is_user_dragging_slider = False
        target = self.seek_var.get()
        if self._current_source is not None:
            self.player.seek(target)
            self.waveform_view.set_playhead_position(target)

    def _ui_update_loop(self) -> None:
        """Loop principal de atualização visual (30 FPS) da interface."""
        try:
            self._apply_pending_tempo_results()
            self._update_counter += 1
            if self._current_source is not None:
                pos = self.player.get_position()
                state = self.player.state
                sr = self._current_source.get_sample_rate()

                if not self._is_user_dragging_slider:
                    self.seek_var.set(pos)
                    self.lbl_time_cur.config(text=self._format_time(pos))
                    self.waveform_view.set_playhead_position(pos)

                if state == PlaybackState.PLAYING or self._is_user_dragging_slider:
                    frame_idx = int(pos * sr)
                    chunk = self._current_source.get_analysis_chunk_at(frame_idx, 4096)
                    ctx = self.analyzer.analyze_chunk(chunk, sr, pos,
                                                      session=self.project_manager.active_session,
                                                      players=self.virtual_players)

                    # 1. Nota Atual
                    if ctx.note != "--":
                        self.lbl_val_note.config(text=ctx.note, fg="#00e676")
                        dev_sign = f"+{ctx.cents_deviation:.1f}" if ctx.cents_deviation > 0 else f"{ctx.cents_deviation:.1f}"
                        self.lbl_sub_note.config(
                            text=f"{ctx.frequency:.2f} Hz  |  Conf: {int(ctx.note_confidence * 100)}%",
                            fg="#ffffff"
                        )
                        cents_color = "#00e676" if abs(ctx.cents_deviation) <= 10 else "#00d2ff" if abs(ctx.cents_deviation) <= 25 else "#ffb703"
                        self.lbl_cents.config(text=f"Afinação: {dev_sign} cents", fg=cents_color)
                    else:
                        self.lbl_val_note.config(text="--", fg="#5a6677")
                        self.lbl_sub_note.config(text="Sem sinal tonal", fg="#5a6677")
                        self.lbl_cents.config(text="Afinação: --", fg="#5a6677")

                    # 2. Ouvido harmônico: RAW, estabilizado, candidato e prior
                    stable_chord = getattr(ctx, "smoothed_detected_chord", "--")
                    stable_conf = getattr(ctx, "stable_chord_confidence", 0.0)
                    stable_duration = getattr(ctx, "stable_chord_duration", 0.0)
                    if stable_chord != "--":
                        self.lbl_val_chord.config(text=stable_chord, fg="#00e676")
                        self.lbl_chord_duration.config(text=f"Tempo no acorde: {stable_duration:.2f} s", fg="#00e676")
                        support_ms = getattr(ctx, "current_chord_support_age_ms", 0.0)
                        prev_txt = f"Anterior: {ctx.previous_chord}  |  Stable conf: {int(stable_conf * 100)}%  |  Suporte: {support_ms:.0f} ms"
                        self.lbl_prev_chord.config(text=prev_txt, fg="#ffffff")
                        stable_root = getattr(ctx, "stable_chord_root", "--")
                        stable_quality = getattr(ctx, "stable_chord_quality", "--")
                        self.lbl_chord_inv.config(
                            text=f"Raiz estável: {stable_root}  |  Qualidade: {stable_quality}",
                            fg="#8c9ba5")
                    else:
                        self.lbl_val_chord.config(text="--", fg="#5a6677")
                        self.lbl_chord_duration.config(text="Tempo no acorde: 0.00 s", fg="#5a6677")
                        self.lbl_prev_chord.config(text=f"Anterior: {ctx.previous_chord}  |  Conf: 0%", fg="#5a6677")
                        self.lbl_chord_inv.config(text="Aguardando harmonia...", fg="#5a6677")
                    raw_chord = getattr(ctx, "raw_detected_chord", "--")
                    raw_conf = getattr(ctx, "raw_chord_confidence", 0.0)
                    self.lbl_chord_raw.config(
                        text=f"RAW: {raw_chord}  |  Conf: {int(raw_conf * 100)}%",
                        fg="#ffffff" if raw_chord != "--" else "#5a6677")
                    candidate = getattr(ctx, "chord_candidate", "--")
                    candidate_root = getattr(ctx, "chord_candidate_root", "--")
                    candidate_conf = getattr(ctx, "chord_candidate_confidence", 0.0)
                    candidate_age = getattr(ctx, "chord_candidate_age_ms", 0.0)
                    candidate_frames = getattr(ctx, "chord_candidate_frames", 0)
                    self.lbl_chord_candidate.config(
                        text=(f"Candidato: {candidate} (raiz {candidate_root})  |  Conf: {int(candidate_conf * 100)}%  |  "
                              f"{candidate_age:.0f} ms / {candidate_frames} frames"),
                        fg="#ffd166" if candidate != "--" else "#5a6677")
                    expected = getattr(ctx, "expected_chart_chord", "--")
                    prior_on = getattr(ctx, "chart_prior_enabled", False)
                    tracking = getattr(ctx, "tracking_state", "TRACKING")
                    self.lbl_chord_prior.config(
                        text=f"Cifra: {expected}  |  Prior: {'ON' if prior_on else 'OFF'}  |  {tracking}",
                        fg="#00e676" if prior_on else "#8c9ba5")
                    self.lbl_harmonic_event.config(
                        text=(f"Evento: {ctx.harmonic_event_chord}  |  Raiz: "
                              f"{ctx.harmonic_event_root}  |  "
                              f"{ctx.current_chord_elapsed_beats:.1f} beats"),
                        fg="#00e676" if ctx.harmonic_event_chord != "--" else "#8c9ba5")

                    # 3. Tonalidade Estimada & Tempo no Tom
                    chart_key_active = bool(self.project_manager.active_session and
                                            self.project_manager.active_session.alignment.event_count and
                                            getattr(self.project_manager.active_session.song.performance_settings,
                                                    "key_source", "CHART").upper() == "CHART")
                    self.lbl_key_source.config(text=("TOM DA CIFRA / SELETOR" if chart_key_active
                                                     else "TONALIDADE ESTIMADA"))
                    if ctx.key != "--":
                        self.lbl_val_key.config(text=ctx.key, fg="#00e676")
                        self.lbl_key_duration.config(text=f"Tempo no tom: {ctx.key_duration:.1f} s", fg="#00e676")
                        prev_key_txt = ("Tom informado para esta música | Conf: 100%" if chart_key_active else
                                        f"Anterior: {ctx.previous_key}  |  Conf: {int(ctx.key_confidence * 100)}%")
                        self.lbl_prev_key.config(text=prev_key_txt, fg="#ffffff")
                        if chart_key_active:
                            self.lbl_key_status.config(text="Cifra / seletor", fg="#00e676")
                        elif ctx.key_candidate != "--" and ctx.key_candidate != ctx.key:
                            cand_txt = f"Candidata: {ctx.key_candidate} ({int(ctx.candidate_confidence * 100)}%)"
                            self.lbl_key_status.config(text=cand_txt, fg="#ffb703")
                        else:
                            self.lbl_key_status.config(
                                text=f"Tonalidade Estável ({int(ctx.key_confidence * 100)}%)",
                                fg="#00e676"
                            )
                    else:
                        self.lbl_val_key.config(text="--", fg="#5a6677")
                        self.lbl_key_duration.config(text="Tempo no tom: 0.0 s", fg="#5a6677")
                        self.lbl_prev_key.config(text="Anterior: --  |  Conf: 0%", fg="#5a6677")
                        self.lbl_key_status.config(text="Aguardando harmonia...", fg="#5a6677")


                    # 4. Relógio Musical & Beat
                    if ctx.bpm > 0:
                        self.lbl_val_bpm.config(text=f"{ctx.bpm:.0f} BPM", fg="#ffffff")
                        self.lbl_sub_meter.config(text=f"Compasso: {ctx.bar}  |  {ctx.meter}", fg="#ffffff")
                        self.lbl_sub_beat.config(text=f"Tempo {ctx.beat} / {ctx.meter.split('/')[0]}", fg="#ffffff")

                        leds = ["○", "○", "○", "○"]
                        b_idx = max(0, min(3, ctx.beat - 1))
                        leds[b_idx] = "●"
                        led_color = "#ffd166" if ctx.beat == 1 else "#00e676" if ctx.is_beat else "#00b4d8"
                        self.lbl_beat_leds.config(text="   ".join(leds), fg=led_color)

                    # 5. Latência Quadripartida
                    self.lbl_lat_proc.config(text=(f"Captura: {ctx.capture_latency:.1f} ms  | "
                                                    f"Cálculo DSP: {ctx.processing_latency:.1f} ms"))
                    self.lbl_lat_win.config(text=(f"Análise: {ctx.analysis_latency:.1f} ms "
                                                   f"(janela {ctx.analysis_window:.1f} ms)"))
                    self.lbl_lat_stab.config(text=(f"Estabilização: {ctx.stabilization_delay:.1f} ms  | "
                                                    f"Scheduler: {ctx.scheduling_latency:.1f} ms  | "
                                                    f"Saída: {ctx.output_latency:.1f} ms"))
                    self.lbl_lat_est.config(text=f"Total estimado: ~{ctx.total_estimated_latency:.1f} ms")

                    # 6. Cromagrama
                    self.chroma_view.update_chroma(ctx.chroma_vector)
                    active = sorted(ctx.active_notes.items(), key=lambda item: -item[1])
                    self.lbl_active_notes.config(
                        text="Notas ativas: " + ("  ".join(
                            f"{note} {score:.2f}" for note, score in active[:6]) if active else "--"))

                    # 7. Progressão Recente & Histórico de Tom
                    self.lbl_hist_progression.config(text=self.analyzer.chord_history.get_summary_text())
                    self.lbl_hist_key.config(text=f"Tonalidades: {self.analyzer.key_history.get_summary_text()}")

                    # 8. Atualizar Tabela de Histórico de Acordes (a cada 6 frames ou se novo evento foi adicionado)
                    if self._update_counter % 6 == 0:
                        self._refresh_chord_history_table()

                    # 9. Baixista Virtual (v0.2)
                    bass = self.analyzer.bass_player
                    cur_ev = bass.current_event
                    cur_dec = bass.current_decision
                    playing = bass.synthesizer.playing_event

                    if bass.enabled:
                        pat_name = ("FUNDAMENTAIS" if bass.pattern == BassPatternType.FUNDAMENTALS else
                                    cur_dec.pattern_type.value.upper() if cur_dec else bass.pattern.value.upper())
                        self.lbl_bass_pattern_active.config(text=pat_name, fg="#00d2ff")
                        self.lbl_bass_sub_pattern.config(text=self.combo_bass_note_value.get())
                        chart_now = getattr(ctx, "chart_published_chord", "--")
                        source_name = "Cifra" if bass.harmony_source == BassHarmonySource.CHART else "Cifra + áudio"
                        self.lbl_bass_chart_source.config(
                            text=f"Cifra agora: {chart_now}  |  Fonte: {source_name}",
                            fg="#00e676" if bass.harmony_source == BassHarmonySource.CHART else "#8c9ba5")
                        if playing is not None:
                            self.lbl_bass_note.config(text=playing.note, fg="#00e676")
                            freq = midi_to_hz(playing.midi_note)
                            self.lbl_bass_freq.config(
                                text=f"MIDI {playing.midi_note}  |  {freq:.1f} Hz  |  Vel: {playing.velocity}")
                            delay = bass.synthesizer.last_render_delay_ms
                            self.lbl_bass_reason.config(
                                text=f"Soando agora · {playing.source} · atraso {delay:.0f} ms",
                                fg="#ffffff")
                        else:
                            self.lbl_bass_note.config(text="--", fg="#5a6677")
                            self.lbl_bass_freq.config(text="Aguardando próximo ataque")
                            self.lbl_bass_reason.config(text="Baixo em silêncio", fg="#8c9ba5")
                        prepared = bass.synthesizer.scheduled_events
                        if prepared:
                            self.lbl_bass_next_note.config(text=prepared[0].note, fg="#ffd166")
                            self.lbl_bass_next_timing.config(
                                text=f"Agendada para {prepared[0].scheduled_time:.2f} s", fg="#8c9ba5")
                        else:
                            self.lbl_bass_next_note.config(text="--", fg="#5a6677")
                            self.lbl_bass_next_timing.config(text="Aguardando próximo ataque", fg="#8c9ba5")
                    elif not bass.enabled:
                        self.lbl_bass_note.config(text="MUTED", fg="#ff5252")
                        self.lbl_bass_freq.config(text="Sintetizador desativado")
                        self.lbl_bass_reason.config(text="Baixista silenciado pelo usuário", fg="#8c9ba5")
                        self.lbl_bass_next_note.config(text="--", fg="#5a6677")
                        self.lbl_bass_next_timing.config(text="Silenciado", fg="#5a6677")

                    if self._update_counter % 6 == 0:
                        self._refresh_bass_debug_table()

                    # 10. Estrutura Musical & Predição Antecipada (v0.3)
                    cur_sec = ctx.current_section
                    if cur_sec != "UNKNOWN" or ctx.current_pattern != "--":
                        sec_disp = f"{cur_sec} ({ctx.current_pattern})" if ctx.current_pattern != "--" else cur_sec
                        self.lbl_struct_cur_section.config(text=sec_disp, fg="#00e676")
                        prog_pct = int(ctx.section_progress * 100)
                        self.lbl_struct_progress.config(
                            text=f"Progresso: {prog_pct}%  |  Comp. {ctx.bar}  |  Conf: {int(ctx.structure_confidence * 100)}%",
                            fg="#ffffff"
                        )
                    else:
                        self.lbl_struct_cur_section.config(text="--", fg="#5a6677")
                        self.lbl_struct_progress.config(text="Progresso: --%  |  Comp. --", fg="#5a6677")

                    if ctx.predicted_next_section not in ("UNKNOWN", "--"):
                        self.lbl_pred_next_section.config(text=ctx.predicted_next_section, fg="#ffd166")
                        self.lbl_pred_timing.config(
                            text=f"Em {ctx.bars_until_change} compasso(s) ({ctx.beats_until_change} tempos)  |  Conf: {int(ctx.prediction_confidence * 100)}%",
                            fg="#ffffff"
                        )
                    else:
                        self.lbl_pred_next_section.config(text="--", fg="#5a6677")
                        self.lbl_pred_timing.config(text="Sem transição prevista", fg="#5a6677")

                    if ctx.predicted_next_chords:
                        look_txt = " → ".join(ctx.predicted_next_chords[:4])
                        self.lbl_pred_lookahead.config(text=look_txt, fg="#00d2ff")
                        self.lbl_pred_reason.config(text=ctx.prediction_reason, fg="#8c9ba5")
                    else:
                        self.lbl_pred_lookahead.config(text="--", fg="#5a6677")
                        self.lbl_pred_reason.config(text="Aguardando harmonia", fg="#5a6677")

                    struct = self.analyzer.structure_analyzer.structure
                    if struct.abstract_sequence:
                        self.lbl_struct_form.config(text=" - ".join(struct.abstract_sequence), fg="#c77dff")
                        self.lbl_struct_summary.config(
                            text=f"{len(struct.sections)} seções  |  {len(struct.patterns)} padrões",
                            fg="#ffffff"
                        )
                    else:
                        self.lbl_struct_form.config(text="--", fg="#5a6677")
                        self.lbl_struct_summary.config(text="0 seções  |  0 padrões", fg="#5a6677")

                    # Atualizar Timeline Estrutural Dinâmica
                    total_dur = self._current_source.get_duration() if self._current_source else 0.0
                    self._draw_structure_timeline(pos, total_dur)

                    # 11. Músico Play-Along & Fusão Cifra-Áudio (v0.4)
                    if self.project_manager.active_session:
                        session = self.project_manager.active_session
                        self._update_playalong_hud_from_session(session)

                        # Realimenta a cifra como PRIOR da detecção do próximo frame
                        # ("ouvir esperando o que a cifra prevê"). Só enviesa casos ambíguos.
                        try:
                            chart_key = session.chart.key if session.chart else None
                        except Exception:
                            chart_key = None
                        self.analyzer.set_harmonic_expectation(session.expected_chord, chart_key)
                        # O cromagrama harmônico já é padrão (acordes precisos). Com cifra
                        # ativa, ligamos também as tétrades/suspensos — só na transição.
                        if not self._high_res_active:
                            self.analyzer.set_detect_extensions(True)
                            self._high_res_active = True
                    else:
                        # Modo livre (sem cifra ativa): audição cega, sem prior.
                        self.analyzer.set_harmonic_expectation(None, None)
                        if self._high_res_active:
                            self.analyzer.set_detect_extensions(False)
                            self._high_res_active = False

                    if self._update_counter % 12 == 0:
                        self._refresh_structure_tables()

                elif state == PlaybackState.STOPPED:
                    self.lbl_val_note.config(text="--", fg="#5a6677")
                    self.lbl_sub_note.config(text="0.0 Hz  |  Conf: 0%", fg="#5a6677")
                    self.lbl_cents.config(text="Afinação: --", fg="#5a6677")
                    self.lbl_val_chord.config(text="--", fg="#5a6677")
                    self.lbl_chord_duration.config(text="Tempo no acorde: 0.00 s", fg="#5a6677")
                    self.lbl_beat_leds.config(text="○   ○   ○   ○", fg="#5a6677")
                    self.lbl_sub_beat.config(text="Tempo: -- / 4", fg="#5a6677")
                    self.lbl_bass_note.config(text="--", fg="#5a6677")
                    self.lbl_bass_freq.config(text="MIDI --  |  -- Hz", fg="#5a6677")
                    self.lbl_bass_reason.config(text="Aguardando reprodução...", fg="#5a6677")
                    self.lbl_bass_next_note.config(text="--", fg="#5a6677")
                    self.lbl_bass_next_timing.config(text="Aguardando harmonia", fg="#5a6677")
                    self.lbl_struct_cur_section.config(text="--", fg="#5a6677")
                    self.lbl_struct_progress.config(text="Progresso: --%  |  Comp. --", fg="#5a6677")
                    self.lbl_pred_next_section.config(text="--", fg="#5a6677")
                    self.lbl_pred_timing.config(text="Em -- compassos", fg="#5a6677")
                    self.lbl_pred_lookahead.config(text="--", fg="#5a6677")
                    self.lbl_pred_reason.config(text="Aguardando harmonia", fg="#5a6677")
                    self.lbl_struct_form.config(text="--", fg="#5a6677")
                    self.lbl_struct_summary.config(text="0 seções  |  0 padrões", fg="#5a6677")
                    self.chroma_view.clear()
                    total_dur = self._current_source.get_duration() if self._current_source else 0.0
                    self._draw_structure_timeline(0.0, total_dur)

                if state == PlaybackState.STOPPED and self.lbl_state.cget("text") == "REPRODUZINDO":
                    self._update_transport_state(PlaybackState.STOPPED)

        except Exception:
            logging.getLogger(__name__).exception("Falha na atualização da interface")
        finally:
            self.root.after(33, self._ui_update_loop)

    def _refresh_chord_history_table(self) -> None:
        """Atualiza a tabela gráfica com os acordes confirmados pelo ChordHistory.
        
        Suporta durações completas (30s, 3 min, 10 min, etc.) inserindo novos eventos
        incrementalmente sem travar a interface ou perder eventos passados.
        """
        events = self.analyzer.chord_history.get_events()
        children = self.tree_history.get_children()
        rendered_count = len(children)
        total_count = len(events)

        # Se o histórico foi reiniciado (ex: Seek, Stop ou novo áudio carregado)
        if total_count < rendered_count:
            for item in children:
                self.tree_history.delete(item)
            rendered_count = 0

        # Se houver novos eventos após o último renderizado, adiciona incrementalmente
        if total_count > rendered_count:
            for i in range(rendered_count, total_count):
                ev = events[i]
                d = ev.to_dict()
                self.tree_history.insert("", "end", values=(
                    d["time_str"],
                    d["chord"],
                    d["duration_str"],
                    d["confidence_str"]
                ))

            # Auto-scroll para acompanhar a execução em tempo real
            updated_children = self.tree_history.get_children()
            if updated_children:
                self.tree_history.see(updated_children[-1])

            self._last_rendered_events_count = total_count

    def _on_algo_changed(self, event=None) -> None:
        algo = self.combo_algo.get()
        self.analyzer.set_pitch_detector(algo)
        self.lbl_status_msg.config(text=f"Algoritmo de detecção de pitch alterado para: {algo}")

    def _analysis_channel_description(self, source: FileAudioSource) -> str:
        if source.channels <= 1:
            return "mono"
        automatic = self.combo_analysis_channel.get() == "Automático"
        suffix = " (automático)" if automatic else ""
        if source.analysis_channel == 0:
            return f"canal esquerdo{suffix}"
        if source.analysis_channel == 1:
            return f"canal direito{suffix}"
        return f"mistura estéreo{suffix}"

    def _on_analysis_channel_changed(self, event=None) -> None:
        source = self._current_source
        if source is None:
            return
        choice = self.combo_analysis_channel.get()
        channel = (self._recommended_analysis_channel if choice == "Automático" else
                   0 if choice == "Canal esquerdo" else
                   1 if choice == "Canal direito" else None)
        if source.channels <= 1:
            channel = None
        source.set_analysis_channel(channel)
        heard_txt = self._analysis_channel_description(source)
        channels_txt = "Estéreo (2 canais)" if source.channels == 2 else f"{source.channels} canal(is)"
        self.lbl_meta_details.config(
            text=(f"Duração: {self._format_time(source.get_duration())}  |  "
                  f"Sample Rate: {source.get_sample_rate()} Hz  |  "
                  f"Canais: {channels_txt}  |  Ouvindo: {heard_txt}"))
        self.lbl_status_msg.config(
            text=f"Referência de análise alterada para {heard_txt}. O ouvido se ajustará nos próximos instantes.")

    def _update_transport_state(self, state: PlaybackState) -> None:
        if state == PlaybackState.PLAYING:
            self.lbl_state.config(text="REPRODUZINDO", fg="#00e676")
            self.btn_play.config(text="▶ TOCANDO", state="disabled")
            self.btn_pause.config(state="normal")
            self.btn_stop.config(state="normal")
        elif state == PlaybackState.PAUSED:
            self.lbl_state.config(text="PAUSADO", fg="#ffb703")
            self.btn_play.config(text="▶ RESUMIR", state="normal")
            self.btn_pause.config(state="disabled")
            self.btn_stop.config(state="normal")
        else:
            self.lbl_state.config(text="PARADO", fg="#5a6677")
            self.btn_play.config(text="▶ PLAY", state="normal")
            self.btn_pause.config(state="disabled")
            self.btn_stop.config(state="disabled")

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 0 or np.isnan(seconds):
            return "00:00"
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def _toggle_bass(self) -> None:
        new_state = not self.analyzer.bass_player.enabled
        self.analyzer.bass_player.enabled = new_state
        if new_state:
            self.btn_bass_toggle.config(
                text="🎸 BAIXO: ATIVADO",
                bg="#0f3424",
                fg="#00e676",
                activebackground="#144d34",
                activeforeground="#00e676"
            )
            self.lbl_status_msg.config(text="Baixista Virtual ATIVADO e acompanhando a harmonia.")
        else:
            self.btn_bass_toggle.config(
                text="🔇 BAIXO: SILENCIADO",
                bg="#341818",
                fg="#ff5252",
                activebackground="#4a2222",
                activeforeground="#ff5252"
            )
            self.lbl_status_msg.config(text="Baixista Virtual SILENCIADO.")

    def _on_bass_pattern_changed(self, event=None) -> None:
        val = self.combo_bass_pattern.get()
        p = self.PATTERN_MAP.get(val, BassPatternType.AUTO)
        self.analyzer.bass_player.pattern = p
        self.lbl_status_msg.config(text=f"Padrão rítmico do baixo alterado para: {val}")

    def _on_bass_note_value_changed(self, event=None) -> None:
        label = self.combo_bass_note_value.get()
        self.analyzer.bass_player.note_value = self.BASS_NOTE_VALUES[label]
        self.lbl_status_msg.config(text=f"Repetição da tônica: {label}")

    def _on_bass_harmony_source_changed(self, event=None) -> None:
        label = self.combo_bass_harmony_source.get()
        self.analyzer.bass_player.harmony_source = self.BASS_HARMONY_SOURCES[label]
        self.lbl_status_msg.config(text=f"Fonte das notas do baixo: {label}")

    def _on_bass_volume_changed(self, val) -> None:
        try:
            vol_val = float(val)
            self.analyzer.bass_player.volume = vol_val / 100.0
            self.lbl_bass_vol_val.config(text=f"{int(vol_val)}%")
        except ValueError:
            pass

    def _toggle_bass_debug(self) -> None:
        if self._bass_debug_visible:
            self.frame_bass_debug.pack_forget()
            self._bass_debug_visible = False
            self.btn_bass_debug_toggle.config(text="📋 Log de Decisões (Debug) ▶")
        else:
            self.frame_bass_debug.pack(fill="x", pady=(4, 0))
            self._bass_debug_visible = True
            self.btn_bass_debug_toggle.config(text="📋 Log de Decisões (Debug) ▼")

    def _refresh_bass_debug_table(self) -> None:
        """Atualiza a tabela gráfica com os eventos do Baixista Virtual."""
        events = self.analyzer.bass_player.get_recent_events()
        children = self.tree_bass_debug.get_children()
        rendered_count = len(children)
        total_count = len(events)

        if total_count < rendered_count:
            for item in children:
                self.tree_bass_debug.delete(item)
            rendered_count = 0

        if total_count > rendered_count:
            for i in range(rendered_count, total_count):
                ev = events[i]
                freq = midi_to_hz(ev.midi_note) if ev.midi_note > 0 else 0.0
                chord_txt = self.analyzer.context.chord if self.analyzer.context.chord != "--" else "--"
                self.tree_bass_debug.insert("", "end", values=(
                    f"#{ev.bar}",
                    f"{ev.beat} / 4",
                    chord_txt,
                    ev.note,
                    f"{ev.midi_note} ({freq:.1f}Hz)",
                    self.analyzer.bass_player.pattern.value.upper(),
                    ev.reason
                ))

            updated_children = self.tree_bass_debug.get_children()
            if updated_children:
                self.tree_bass_debug.see(updated_children[-1])

            self._last_rendered_bass_events_count = total_count

    # ============================================================
    # Músico Play-Along, Setlist & Cancioneiro de Cifras (v0.4)
    # ============================================================
    def _init_default_project_content(self) -> None:
        """Inicializa repertório padrão com músicas de demonstração caso esteja vazio."""
        setlist = self.project_manager.project.get_active_setlist()
        if not setlist.songs:
            # Música 1: Violão Acústico C-G-Am-F
            chart_1 = (
                "Tom: C\nBPM: 120\n\n"
                "[Intro]\n"
                "C        G\n"
                "Hoje o sol vai brilhar\n"
                "Am       F\n"
                "Nossa canção vai ecoar\n\n"
                "[Verso]\n"
                "C        G\n"
                "Caminhando pela praia\n"
                "Am       F\n"
                "O violão nos acompanha\n\n"
                "[Refrão]\n"
                "C        G        Am       F x2\n"
                "Vem cantar com a gente neste dia tão feliz"
            )
            self.project_manager.import_song(
                title="Violão Acústico (C-G-Am-F)",
                artist="Virtual Band Studio",
                audio_path="violao_teste_120bpm.wav",
                chart_text=chart_1,
                key="C Major",
                bpm=120.0,
                duration=32.0,
                auto_add_to_setlist=True
            )

            # Música 2: Balada em Ré Menor
            chart_2 = (
                "Tom: Dm\nBPM: 120\n\n"
                "[Verso]\n"
                "Dm       Bb       C        A\n"
                "Noite calma na cidade esperando a novidade\n"
                "Dm       Bb       C        A\n"
                "Os acordes vão surgindo com total sinceridade\n\n"
                "[Refrão]\n"
                "Dm       Bb       C        A x2\n"
                "A banda virtual acompanha o músico tocar"
            )
            self.project_manager.import_song(
                title="Balada em Ré Menor",
                artist="Virtual Band Studio",
                audio_path="test_song_120bpm.wav",
                chart_text=chart_2,
                key="D Minor",
                bpm=120.0,
                duration=32.0,
                auto_add_to_setlist=True
            )

        first_song = setlist.get_active_song()
        if first_song:
            self.project_manager.open_song(first_song)

    def _build_playalong_content(self, parent: tk.Frame) -> None:
        """Painel completo do Músico Play-Along, Setlist e Cifra (v0.4)."""
        container = tk.Frame(parent, bg="#181d26", padx=8, pady=4)
        container.pack(fill="both", expand=True)

        # -------------------------------------------------------------
        # 1. Painel Esquerdo: Setlist, Repertório & Prontidão da Banda
        # -------------------------------------------------------------
        left_pane = tk.Frame(container, bg="#141820", width=310, bd=1, relief="solid", padx=8, pady=6)
        left_pane.pack(side="left", fill="y", padx=(0, 8))
        left_pane.pack_propagate(False)

        # Título do Projeto e Setlist
        self.lbl_setlist_proj_title = tk.Label(
            left_pane,
            text=f"📁 {self.project_manager.project.name.upper()}",
            bg="#141820",
            fg="#00d2ff",
            font=("Segoe UI", 9, "bold"),
            anchor="w"
        )
        self.lbl_setlist_proj_title.pack(fill="x", pady=(0, 2))

        self.lbl_active_setlist_name = tk.Label(
            left_pane,
            text=f"SETLIST: {self.project_manager.project.get_active_setlist().name}",
            bg="#141820",
            fg="#ffd166",
            font=("Segoe UI", 8, "bold"),
            anchor="w"
        )
        self.lbl_active_setlist_name.pack(fill="x", pady=(0, 4))

        # Listbox do Repertório
        list_frame = tk.Frame(left_pane, bg="#141820")
        list_frame.pack(fill="both", expand=True, pady=(0, 4))

        self.listbox_setlist = tk.Listbox(
            list_frame,
            selectmode="extended",
            exportselection=False,
            bg="#0d1015",
            fg="#ffffff",
            selectbackground="#0077b6",
            selectforeground="#ffffff",
            font=("Segoe UI", 9),
            bd=0,
            highlightthickness=1,
            highlightbackground="#262f3e",
            activestyle="none"
        )
        sb_setlist = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox_setlist.yview)
        self.listbox_setlist.configure(yscrollcommand=sb_setlist.set)
        self.listbox_setlist.pack(side="left", fill="both", expand=True)
        sb_setlist.pack(side="right", fill="y")
        self.listbox_setlist.bind("<<ListboxSelect>>", self._on_setlist_select)
        self.listbox_setlist.bind("<Delete>", self._on_delete_selected_songs)

        # Botões de Navegação no Setlist
        nav_box = tk.Frame(left_pane, bg="#141820")
        nav_box.pack(fill="x", pady=(2, 4))

        btn_first = tk.Button(nav_box, text="⏮", bg="#202735", fg="#ffffff", relief="flat",
                              font=("Segoe UI", 8), command=self._on_first_song)
        btn_first.pack(side="left", fill="x", expand=True, padx=1)

        btn_prev = tk.Button(nav_box, text="◀ Ant.", bg="#202735", fg="#ffffff", relief="flat",
                             font=("Segoe UI", 8), command=self._on_prev_song)
        btn_prev.pack(side="left", fill="x", expand=True, padx=1)

        btn_next = tk.Button(nav_box, text="Próx. ▶", bg="#202735", fg="#ffffff", relief="flat",
                             font=("Segoe UI", 8), command=self._on_next_song)
        btn_next.pack(side="left", fill="x", expand=True, padx=1)

        btn_last = tk.Button(nav_box, text="⏭", bg="#202735", fg="#ffffff", relief="flat",
                             font=("Segoe UI", 8), command=self._on_last_song)
        btn_last.pack(side="left", fill="x", expand=True, padx=1)

        # Botões de Ação do Projeto
        act_box = tk.Frame(left_pane, bg="#141820")
        act_box.pack(fill="x", pady=(2, 6))

        btn_new_song = tk.Button(act_box, text="➕ Importar Cifra", bg="#0f3424", fg="#00e676", relief="flat",
                                 font=("Segoe UI", 8, "bold"), command=self._on_new_song_dialog)
        btn_new_song.pack(side="left", fill="x", expand=True, padx=(0, 2))

        btn_save_proj = tk.Button(act_box, text="💾 Salvar", bg="#1a2d42", fg="#00d2ff", relief="flat",
                                  font=("Segoe UI", 8, "bold"), command=self._on_save_project)
        btn_save_proj.pack(side="left", fill="x", expand=True, padx=(2, 0))

        btn_remove_song = tk.Button(
            act_box, text="🗑 Remover", bg="#3a1c1c", fg="#ff9b94", relief="flat",
            font=("Segoe UI", 8, "bold"), command=self._on_delete_selected_songs)
        btn_remove_song.pack(side="left", fill="x", expand=True, padx=(4, 0))

        # Card de Prontidão da Banda Virtual
        band_card = tk.Frame(left_pane, bg="#181d26", bd=1, relief="solid", padx=6, pady=4)
        band_card.pack(fill="x", side="bottom")

        tk.Label(band_card, text="🎸 BANDA VIRTUAL (PRONTIDÃO)", bg="#181d26", fg="#8c9ba5",
                 font=("Segoe UI", 7, "bold"), anchor="w").pack(fill="x", pady=(0, 2))

        players_grid = tk.Frame(band_card, bg="#181d26")
        players_grid.pack(fill="x")

        self.lbl_status_bass = tk.Label(players_grid, text="🎸 Baixo: READY", bg="#181d26", fg="#00e676",
                                        font=("Segoe UI", 8, "bold"))
        self.lbl_status_bass.grid(row=0, column=0, sticky="w", padx=2, pady=1)

        self.lbl_status_drums = tk.Label(players_grid, text="🥁 Bateria: READY", bg="#181d26", fg="#00e676",
                                         font=("Segoe UI", 8, "bold"))
        self.lbl_status_drums.grid(row=0, column=1, sticky="w", padx=2, pady=1)

        self.lbl_status_keys = tk.Label(players_grid, text="🎹 Teclado: READY", bg="#181d26", fg="#00e676",
                                        font=("Segoe UI", 8, "bold"))
        self.lbl_status_keys.grid(row=1, column=0, sticky="w", padx=2, pady=1)

        self.lbl_status_guitar = tk.Label(players_grid, text="🎸 Guitarra: READY", bg="#181d26", fg="#00e676",
                                          font=("Segoe UI", 8, "bold"))
        self.lbl_status_guitar.grid(row=1, column=1, sticky="w", padx=2, pady=1)

        # -------------------------------------------------------------
        # 2. Painel Direito: Dashboard ao Vivo, Cancioneiro & Controles
        # -------------------------------------------------------------
        right_pane = tk.Frame(container, bg="#181d26")
        right_pane.pack(side="left", fill="both", expand=True)

        # Linha Superior do HUD do Músico
        hud_top = tk.Frame(right_pane, bg="#141820", bd=1, relief="solid", padx=8, pady=4)
        hud_top.pack(fill="x", pady=(0, 4))

        self.lbl_playalong_song_title = tk.Label(
            hud_top,
            text="MÚSICA ATIVA",
            bg="#141820",
            fg="#ffffff",
            font=("Segoe UI", 11, "bold")
        )
        self.lbl_playalong_song_title.pack(side="left", padx=(0, 10))

        btn_hud_import = tk.Button(
            hud_top,
            text="➕ Importar / Trocar Cifra",
            bg="#0f3424",
            fg="#00e676",
            activebackground="#144d34",
            activeforeground="#00e676",
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._on_new_song_dialog
        )
        btn_hud_import.pack(side="left", padx=(0, 10))

        self.btn_follow_mode = tk.Button(
            hud_top,
            text="FOLLOW MODE: ATIVADO",
            bg="#0f3424",
            fg="#00e676",
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            padx=8,
            pady=2,
            command=self._toggle_follow_mode
        )
        self.btn_follow_mode.pack(side="right", padx=4)

        self.lbl_mode_tag = tk.Label(
            hud_top,
            text="MODO: PLAY-ALONG (ÁUDIO)",
            bg="#1f2937",
            fg="#00d2ff",
            font=("Segoe UI", 8, "bold"),
            padx=6,
            pady=2
        )
        self.lbl_mode_tag.pack(side="right", padx=4)

        # Métrica Cards do Músico (Acorde Gigante, Seção, Próximo Acorde, Compasso)
        hud_metrics = tk.Frame(right_pane, bg="#181d26")
        hud_metrics.pack(fill="x", pady=(0, 4))

        # Card 1: Seção Atual
        c_sec = tk.Frame(hud_metrics, bg="#141820", bd=1, relief="solid", padx=6, pady=4)
        c_sec.pack(side="left", fill="both", expand=True, padx=(0, 4))
        tk.Label(c_sec, text="SEÇÃO ATUAL", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_playalong_section = tk.Label(c_sec, text="--", bg="#141820", fg="#ffd166", font=("Segoe UI", 12, "bold"))
        self.lbl_playalong_section.pack(anchor="w")

        # Card 2: Acorde Atual (Gigante!)
        c_chord = tk.Frame(hud_metrics, bg="#141820", bd=1, relief="solid", padx=8, pady=2)
        c_chord.pack(side="left", fill="both", expand=True, padx=2)
        tk.Label(c_chord, text="ACORDE ATUAL", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_playalong_chord = tk.Label(c_chord, text="--", bg="#141820", fg="#00e676", font=("Segoe UI", 18, "bold"))
        self.lbl_playalong_chord.pack(anchor="w")

        # Card 3: Próximo Acorde
        c_next = tk.Frame(hud_metrics, bg="#141820", bd=1, relief="solid", padx=6, pady=4)
        c_next.pack(side="left", fill="both", expand=True, padx=2)
        tk.Label(c_next, text="PRÓXIMO ACORDE", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_playalong_next_chord = tk.Label(c_next, text="--", bg="#141820", fg="#00d2ff", font=("Segoe UI", 14, "bold"))
        self.lbl_playalong_next_chord.pack(anchor="w")

        # Card 4: Compasso / Beat / BPM
        c_bar = tk.Frame(hud_metrics, bg="#141820", bd=1, relief="solid", padx=6, pady=4)
        c_bar.pack(side="left", fill="both", expand=True, padx=2)
        tk.Label(c_bar, text="COMPASSO & RITMO", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_playalong_bar_beat = tk.Label(c_bar, text="Comp. 1  |  Tempo 1 / 4", bg="#141820", fg="#ffffff", font=("Segoe UI", 9, "bold"))
        self.lbl_playalong_bar_beat.pack(anchor="w")
        self.lbl_playalong_bpm = tk.Label(c_bar, text="120.0 BPM", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 8))
        self.lbl_playalong_bpm.pack(anchor="w")

        # Card 5: Status da Fusão Cifra + Áudio
        c_fus = tk.Frame(hud_metrics, bg="#141820", bd=1, relief="solid", padx=6, pady=4)
        c_fus.pack(side="left", fill="both", expand=True, padx=(4, 0))
        tk.Label(c_fus, text="FUSÃO CIFRA-ÁUDIO", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_playalong_fusion = tk.Label(c_fus, text="Cifra Soberana (Ok)", bg="#141820", fg="#00e676", font=("Segoe UI", 9, "bold"))
        self.lbl_playalong_fusion.pack(anchor="w")
        self.lbl_playalong_fusion_sub = tk.Label(c_fus, text="Áudio confirmando", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 8))
        self.lbl_playalong_fusion_sub.pack(anchor="w")

        # Barra de Ações Rápidas de Edição Direta da Cifra
        chart_edit_bar = tk.Frame(right_pane, bg="#181d26", padx=8, pady=4, bd=1, relief="solid")
        chart_edit_bar.pack(fill="x", pady=(0, 3))

        tk.Label(
            chart_edit_bar,
            text="✍️ CIFRA & LETRA (Edição Direta Ativa):",
            bg="#181d26",
            fg="#00d2ff",
            font=("Segoe UI", 9, "bold")
        ).pack(side="left", padx=(0, 8))

        self.btn_save_chart = tk.Button(
            chart_edit_bar,
            text="💾 Salvar Cifra (Ctrl+S)",
            bg="#00e676",
            fg="#0d1015",
            activebackground="#00c853",
            activeforeground="#0d1015",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
            pady=2,
            cursor="hand2",
            command=self._on_save_chart_direct
        )
        self.btn_save_chart.pack(side="left", padx=3)

        btn_discard_chart = tk.Button(
            chart_edit_bar,
            text="↺ Descartar",
            bg="#202735",
            fg="#ffffff",
            activebackground="#2a3344",
            activeforeground="#ffffff",
            font=("Segoe UI", 8),
            relief="flat",
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._on_discard_chart_edits
        )
        btn_discard_chart.pack(side="left", padx=3)

        for _txt, _cmd in (
            ("↶ Desfazer", self._undo_structural_edit),
            ("↷ Refazer", self._redo_structural_edit),
        ):
            tk.Button(chart_edit_bar, text=_txt, bg="#202735", fg="#ffffff",
                      activebackground="#2a3344", activeforeground="#ffffff",
                      font=("Segoe UI", 8), relief="flat", padx=8, pady=2,
                      cursor="hand2", command=_cmd).pack(side="left", padx=2)

        btn_recolor_chart = tk.Button(
            chart_edit_bar,
            text="🎨 Re-colorir",
            bg="#202735",
            fg="#ffd166",
            activebackground="#2a3344",
            activeforeground="#ffd166",
            font=("Segoe UI", 8),
            relief="flat",
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._apply_chart_syntax_highlighting
        )
        btn_recolor_chart.pack(side="left", padx=3)

        # --- Seletor de Tom (transposição da cifra inteira) ---
        tk.Label(chart_edit_bar, text="  🎚 Tom:", bg="#181d26", fg="#00d2ff",
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(6, 2))
        self.combo_key = ttk.Combobox(
            chart_edit_bar, width=8, state="readonly", font=("Segoe UI", 8),
            values=self._all_key_options()
        )
        self.combo_key.pack(side="left", padx=2)
        self.combo_key.bind("<<ComboboxSelected>>", self._on_change_key)

        # --- Operações de Bloco (clique para selecionar; depois duplica/deleta/move) ---
        for _txt, _cmd, _fg in (
            ("⧉ Duplicar", self._on_copy_block, "#00d2ff"),
            ("🗑 Deletar", self._on_delete_block, "#ff8a80"),
            ("▲", lambda: self._on_move_block(-1), "#ffffff"),
            ("▼", lambda: self._on_move_block(1), "#ffffff"),
        ):
            tk.Button(chart_edit_bar, text=_txt, bg="#202735", fg=_fg,
                      activebackground="#2a3344", activeforeground=_fg,
                      font=("Segoe UI", 8), relief="flat", padx=8, pady=2,
                      cursor="hand2", command=_cmd).pack(side="left", padx=2)

        self.lbl_chart_edit_status = tk.Label(
            chart_edit_bar,
            text="✓ Sincronizado",
            bg="#181d26",
            fg="#00e676",
            font=("Segoe UI", 8, "italic")
        )
        self.lbl_chart_edit_status.pack(side="right", padx=6)

        # Visualizador e Editor de Cifra e Letra (Cancioneiro Interativo)
        chart_view_frame = tk.Frame(right_pane, bg="#141820", bd=1, relief="solid")
        chart_view_frame.pack(fill="both", expand=True, pady=(0, 4))

        self.text_chart_view = tk.Text(
            chart_view_frame,
            bg="#0d1015",
            fg="#ffffff",
            insertbackground="#00d2ff",
            selectbackground="#0077b6",
            font=("Consolas", 11),
            wrap="none",
            bd=0,
            padx=10,
            pady=8,
            undo=True,
            autoseparators=True,
            maxundo=-1
        )
        chart_scroll_y = ttk.Scrollbar(chart_view_frame, orient="vertical", command=self.text_chart_view.yview)
        chart_scroll_x = ttk.Scrollbar(chart_view_frame, orient="horizontal", command=self.text_chart_view.xview)
        self.text_chart_view.configure(yscrollcommand=chart_scroll_y.set, xscrollcommand=chart_scroll_x.set)

        chart_scroll_y.pack(side="right", fill="y")
        chart_scroll_x.pack(side="bottom", fill="x")
        self.text_chart_view.pack(side="left", fill="both", expand=True)

        self.text_chart_view.tag_configure("chord_symbol", foreground="#00d2ff", font=("Consolas", 11, "bold"))
        self.text_chart_view.tag_configure("section_tag", foreground="#ffd166", font=("Segoe UI", 11, "bold"))
        self.text_chart_view.tag_configure("lyric_line", foreground="#f0f4f8", font=("Consolas", 11))
        self.text_chart_view.tag_configure("active_chord", background="#0f4c81", foreground="#ffffff", font=("Consolas", 12, "bold"))
        self.text_chart_view.tag_configure("active_line", background="#1a2738")

        # Tags de FUNDO por TIPO de seção (Intro/Verso/Refrão/Ponte…): o BLOCO inteiro
        # recebe um fundo tingido na cor da seção (suave para o corpo, mais forte no
        # cabeçalho e no bloco SELECIONADO), para diferenciar visualmente as partes.
        _bg = "#0d1015"
        for _sec_type, _color in SECTION_COLORS.items():
            self.text_chart_view.tag_configure(
                f"secbg_{_sec_type}", background=self._blend_hex(_color, _bg, 0.20))
            self.text_chart_view.tag_configure(
                f"sechd_{_sec_type}", background=self._blend_hex(_color, _bg, 0.40),
                foreground=_color, font=("Segoe UI", 11, "bold"))
            self.text_chart_view.tag_configure(
                f"secsel_{_sec_type}", background=self._blend_hex(_color, _bg, 0.55))
            self.text_chart_view.tag_configure(
                f"secselhd_{_sec_type}", background=self._blend_hex(_color, _bg, 0.72),
                foreground="#ffffff", font=("Segoe UI", 11, "bold"))
        # Índice do bloco atualmente selecionado por clique (-1 = nenhum)
        self._selected_block_index = -1
        self.text_chart_view.bind("<Button-1>", self._on_chart_click, add="+")

        # Eventos do Editor de Cifra
        self.text_chart_view.bind("<KeyRelease>", self._on_chart_key_release)
        self.text_chart_view.bind("<Control-s>", lambda e: (self._on_save_chart_direct(), "break")[1])
        self.text_chart_view.bind("<Control-S>", lambda e: (self._on_save_chart_direct(), "break")[1])
        self.text_chart_view.bind("<Control-z>", self._on_undo_shortcut)
        self.text_chart_view.bind("<Control-Shift-z>", self._on_redo_shortcut)
        self.text_chart_view.bind("<Control-Shift-Z>", self._on_redo_shortcut)

        # Barra de Controles Manuais para Ensaio
        rehearsal_bar = tk.Frame(right_pane, bg="#141820", padx=6, pady=3)
        rehearsal_bar.pack(fill="x")

        tk.Label(rehearsal_bar, text="NAVEGAÇÃO MANUAL:", bg="#141820", fg="#8c9ba5",
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 6))

        btn_rew = tk.Button(rehearsal_bar, text="⏮ Início", bg="#202735", fg="#ffffff", relief="flat",
                            font=("Segoe UI", 8), command=lambda: self._on_seek_bar_manual(to_start=True))
        btn_rew.pack(side="left", padx=2)

        btn_prev_bar = tk.Button(rehearsal_bar, text="◀ Compasso", bg="#202735", fg="#ffffff", relief="flat",
                                 font=("Segoe UI", 8), command=lambda: self._on_seek_bar_manual(-1))
        btn_prev_bar.pack(side="left", padx=2)

        btn_next_bar = tk.Button(rehearsal_bar, text="Compasso ▶", bg="#202735", fg="#ffffff", relief="flat",
                                 font=("Segoe UI", 8), command=lambda: self._on_seek_bar_manual(+1))
        btn_next_bar.pack(side="left", padx=2)

        btn_prev_ch = tk.Button(rehearsal_bar, text="◀ Acorde", bg="#202735", fg="#00d2ff", relief="flat",
                                font=("Segoe UI", 8), command=lambda: self._on_seek_chord_manual(-1))
        btn_prev_ch.pack(side="left", padx=2)

        btn_next_ch = tk.Button(rehearsal_bar, text="Acorde ▶", bg="#202735", fg="#00d2ff", relief="flat",
                                font=("Segoe UI", 8), command=lambda: self._on_seek_chord_manual(+1))
        btn_next_ch.pack(side="left", padx=2)

        btn_prev_sec = tk.Button(rehearsal_bar, text="◀ Seção", bg="#202735", fg="#ffd166", relief="flat",
                                 font=("Segoe UI", 8), command=lambda: self._on_seek_section_manual(-1))
        btn_prev_sec.pack(side="left", padx=2)

        btn_next_sec = tk.Button(rehearsal_bar, text="Seção ▶", bg="#202735", fg="#ffd166", relief="flat",
                                 font=("Segoe UI", 8), command=lambda: self._on_seek_section_manual(+1))
        btn_next_sec.pack(side="left", padx=2)

        self._refresh_setlist_listbox()

    def _refresh_setlist_listbox(self) -> None:
        """Atualiza a lista visual de músicas no setlist."""
        self.listbox_setlist.delete(0, "end")
        setlist = self.project_manager.project.get_active_setlist()
        active_song = setlist.get_active_song()
        active_idx = 0

        for i, song in enumerate(setlist.songs):
            marker = "▶ " if (active_song and song.id == active_song.id) else "   "
            display_txt = f"{marker}{i + 1:02d} - {song.title}"
            self.listbox_setlist.insert("end", display_txt)
            if active_song and song.id == active_song.id:
                active_idx = i

        if setlist.songs:
            self.listbox_setlist.selection_set(active_idx)
            self.listbox_setlist.see(active_idx)

        self.lbl_setlist_proj_title.config(text=f"📁 {self.project_manager.project.name.upper()}")
        self.lbl_active_setlist_name.config(text=f"SETLIST: {setlist.name}")
        if active_song:
            self.lbl_playalong_song_title.config(text=f"{active_song.title} — {active_song.artist} ({active_song.key})")
            self._render_chart_text(active_song)
        else:
            self.lbl_playalong_song_title.config(text="Nenhuma música na setlist")
            self.text_chart_view.delete("1.0", "end")

    def _on_setlist_select(self, event=None) -> None:
        """Manipula seleção manual de música na listbox."""
        sel = self.listbox_setlist.curselection()
        if len(sel) != 1:
            return
        idx = sel[0]
        setlist = self.project_manager.project.get_active_setlist()
        if 0 <= idx < len(setlist.songs):
            target_song = setlist.songs[idx]
            if (self.project_manager.active_session is None or
                    self.project_manager.active_session.song.id != target_song.id):
                self._switch_to_song(target_song)

    def _on_delete_selected_songs(self, event=None):
        """Delete remove da setlist as entradas marcadas, preservando os arquivos."""
        setlist = self.project_manager.project.get_active_setlist()
        indices = sorted(set(self.listbox_setlist.curselection()))
        selected = [setlist.songs[index] for index in indices
                    if 0 <= index < len(setlist.songs)]
        if not selected:
            return "break"
        selected_ids = [song.id for song in selected]
        active = self.project_manager.active_session
        removing_active = active is not None and active.song.id in selected_ids
        if removing_active:
            self.player.stop()
            self.virtual_players.reset_all()
        try:
            removed = self.project_manager.remove_songs_from_active_setlist(selected_ids)
        except Exception as exc:
            messagebox.showerror("Erro ao remover música", str(exc), parent=self.root)
            return "break"
        next_song = setlist.get_active_song()
        if removing_active and next_song is not None:
            self._switch_to_song(next_song)
        elif removing_active:
            self.player.unload_source()
            self._current_source = None
            self.analyzer.reset_musical_history()
            self.waveform_view.clear()
            self.seek_var.set(0.0)
            self.seek_slider.config(to=0.0)
            self.lbl_time_cur.config(text="00:00")
            self.lbl_time_total.config(text="00:00")
            self.lbl_file_name.config(text="Nenhum áudio carregado")
            self.lbl_meta_details.config(text="Importe uma música para começar.")
            self._update_transport_state(PlaybackState.STOPPED)
            self._has_unsaved_chart_edits = False
            self._refresh_setlist_listbox()
        else:
            self._refresh_setlist_listbox()
        self.lbl_status_msg.config(
            text=f"{len(removed)} música(s) removida(s) da setlist. Arquivos preservados.")
        return "break"

    def _switch_to_song(self, song: Song) -> None:
        """Executa a transição hermética para uma nova música."""
        self._audio_generation += 1
        self.player.unload_source()
        self._current_source = None
        self.analyzer.reset_musical_history()
        self.analyzer.set_harmonic_expectation(None, None)
        self.waveform_view.clear()
        self.seek_var.set(0.0)
        self.seek_slider.config(to=0.0)
        self.lbl_file_name.config(text="Nenhum áudio carregado")
        self.lbl_meta_details.config(text="Selecione um arquivo de áudio para reproduzir.")
        self.lbl_time_cur.config(text="00:00")
        self.lbl_time_total.config(text="00:00")
        self._update_transport_state(PlaybackState.STOPPED)
        session = self.project_manager.open_song(song)
        self._sync_reference_sources(song)

        if song.audio_path and os.path.exists(song.audio_path):
            self._load_audio_file(song.audio_path)

        self._refresh_setlist_listbox()
        self.lbl_status_msg.config(text=f"Música selecionada: {song.title}")

    def _on_next_song(self) -> None:
        next_s = self.project_manager.setlist_manager.next_song()
        if next_s:
            self._switch_to_song(next_s)

    def _on_prev_song(self) -> None:
        prev_s = self.project_manager.setlist_manager.prev_song()
        if prev_s:
            self._switch_to_song(prev_s)

    def _on_first_song(self) -> None:
        first_s = self.project_manager.setlist_manager.first_song()
        if first_s:
            self._switch_to_song(first_s)

    def _on_last_song(self) -> None:
        last_s = self.project_manager.setlist_manager.last_song()
        if last_s:
            self._switch_to_song(last_s)

    def _toggle_follow_mode(self) -> None:
        self._follow_mode_enabled = not self._follow_mode_enabled
        if self._follow_mode_enabled:
            self.btn_follow_mode.config(text="FOLLOW MODE: ATIVADO", bg="#0f3424", fg="#00e676")
        else:
            self.btn_follow_mode.config(text="FOLLOW MODE: DESATIVADO", bg="#3a1c1c", fg="#ff7b72")

    def _on_new_song_dialog(self) -> None:
        """Diálogo avançado para importação de cifras em múltiplos formatos (TXT, DOCX, PNG, JPG, WEBP ou Colar)."""
        dlg = tk.Toplevel(self.root)
        dlg.title("🎼 Importar Cifra — Multi-Formato (TXT, DOCX, PDF, Imagem ou Colar)")
        dlg.geometry("720x640")
        dlg.minsize(640, 520)
        dlg.configure(bg="#141820")
        dlg.transient(self.root)
        dlg.grab_set()

        var_title = tk.StringVar(value=f"Música {len(self.project_manager.project.get_active_setlist().songs) + 1:02d}")
        var_artist = tk.StringVar(value="Artista")
        var_audio_path = tk.StringVar(value="")
        var_source_info = tk.StringVar(value="Nenhum arquivo selecionado (Você pode digitar ou colar a cifra abaixo)")
        var_conf_warning = tk.StringVar(value="")

        # 1. Cabeçalho de Metadados e Seleção de Arquivo
        top_box = tk.Frame(dlg, bg="#181d26", bd=1, relief="solid", padx=12, pady=10)
        top_box.pack(side="top", fill="x", padx=12, pady=10)

        # 2. Rodapé de Ações — EMPACOTADO COM side="bottom" ANTES da área central
        # Isso garante no Tkinter que a barra inferior com o botão OK NUNCA suma da tela!
        bot_bar = tk.Frame(dlg, bg="#141820", padx=14, pady=10, bd=1, relief="solid")
        bot_bar.pack(side="bottom", fill="x")

        # 3. Aviso de baixa confiança (se houver)
        lbl_warning = tk.Label(dlg, textvariable=var_conf_warning, bg="#141820", fg="#ffd166",
                               font=("Segoe UI", 8, "bold"), wraplength=680, justify="left")
        lbl_warning.pack(side="top", fill="x", padx=14, pady=(0, 4))

        # 4. Área Central: Visualização e Edição Manual da Cifra (Preenche o espaço restante)
        mid_box = tk.Frame(dlg, bg="#181d26", bd=1, relief="solid", padx=8, pady=6)
        mid_box.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 8))

        tk.Label(mid_box, text="Cifra e Letra (Edição Livre / Cole seu texto aqui se desejar):",
                 bg="#181d26", fg="#00d2ff", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))

        txt_frame = tk.Frame(mid_box, bg="#0d1015")
        txt_frame.pack(fill="both", expand=True)

        txt_preview = tk.Text(
            txt_frame,
            bg="#0d1015",
            fg="#ffffff",
            insertbackground="#00d2ff",
            selectbackground="#0077b6",
            font=("Consolas", 11),
            bd=0,
            wrap="none",
            padx=8,
            pady=6
        )
        sb_y = ttk.Scrollbar(txt_frame, orient="vertical", command=txt_preview.yview)
        sb_x = ttk.Scrollbar(txt_frame, orient="horizontal", command=txt_preview.xview)
        txt_preview.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)

        sb_y.pack(side="right", fill="y")
        sb_x.pack(side="bottom", fill="x")
        txt_preview.pack(side="left", fill="both", expand=True)

        sample_init = "[Intro]\nC        G\n\n[Verso]\nAm       F\nEu estava pensando em você\n\n[Refrão]\nC        G        Am       F x2\nVem cantar com a banda virtual"
        txt_preview.insert("1.0", sample_init)

        # Lógica de Confirmação e Inserção no Setlist
        def _on_confirm_import():
            cifra_content = txt_preview.get("1.0", "end").strip()
            if not cifra_content:
                messagebox.showwarning("Cifra Vazia", "Por favor, informe ou cole o texto da cifra.", parent=dlg)
                return

            t_val = var_title.get().strip() or "Nova Canção"
            a_val = var_artist.get().strip() or "Artista"
            audio_val = var_audio_path.get().strip()

            new_s = self.project_manager.import_song(
                title=t_val,
                artist=a_val,
                audio_path=audio_val,
                chart_text=cifra_content,
                auto_add_to_setlist=True
            )
            self._switch_to_song(new_s)
            self.main_notebook.select(self.playalong_tab)
            dlg.destroy()
            messagebox.showinfo("Importação Concluída", f"Canção '{t_val}' importada com sucesso para o repertório!")

        # Botões do Rodapé (Fixos e Impossíveis de Sumir)
        btn_cancel = tk.Button(bot_bar, text="Cancelar", bg="#202735", fg="#ffffff", relief="flat",
                               font=("Segoe UI", 9), padx=14, pady=5, cursor="hand2", command=dlg.destroy)
        btn_cancel.pack(side="right", padx=(8, 0))

        btn_confirm = tk.Button(
            bot_bar,
            text="✓ OK — INSERIR NO SETLIST",
            bg="#00e676",
            fg="#0d1015",
            activebackground="#00c853",
            activeforeground="#0d1015",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=18,
            pady=5,
            cursor="hand2",
            command=_on_confirm_import
        )
        btn_confirm.pack(side="right")

        # Conteúdo do Cabeçalho (Linha 1: Seleção de Arquivo + Botão OK no Topo)
        row_file = tk.Frame(top_box, bg="#181d26")
        row_file.pack(fill="x", pady=(0, 8))

        def _on_choose_chart_file():
            path = filedialog.askopenfilename(
                parent=dlg,
                title="Selecionar Arquivo de Cifra (TXT, DOCX, PDF, PNG, JPG, WEBP)",
                filetypes=[
                    ("Todos os Formatos Suportados", "*.txt;*.docx;*.pdf;*.png;*.jpg;*.jpeg;*.webp"),
                    ("Documentos PDF (*.pdf)", "*.pdf"),
                    ("Documentos Word (*.docx)", "*.docx"),
                    ("Cifras de Texto (*.txt)", "*.txt"),
                    ("Imagens de Cifra (*.png;*.jpg;*.jpeg;*.webp)", "*.png;*.jpg;*.jpeg;*.webp"),
                    ("Todos os Arquivos (*.*)", "*.*")
                ]
            )
            if not path:
                return

            ext = os.path.splitext(path)[1].lower()
            base_name = os.path.splitext(os.path.basename(path))[0]
            if var_title.get().startswith("Música "):
                var_title.set(base_name)

            try:
                doc = None
                if ext == ".docx":
                    doc = DocxChartSource().load(path)
                    var_source_info.set(f"📄 DOCX carregado: {os.path.basename(path)}")
                elif ext == ".pdf":
                    dlg.config(cursor="watch")
                    dlg.update_idletasks()
                    doc = PdfChartSource().load(path)
                    dlg.config(cursor="")
                    var_source_info.set(f"📑 PDF carregado: {os.path.basename(path)}")
                elif ext in (".png", ".jpg", ".jpeg", ".webp"):
                    var_source_info.set(f"🖼️ Imagem carregada (OCR Espacial): {os.path.basename(path)}")
                    dlg.config(cursor="watch")
                    dlg.update_idletasks()
                    doc = ImageChartSource().load(path)
                    dlg.config(cursor="")
                else:
                    doc = TextChartSource().load(path)
                    var_source_info.set(f"📝 Texto TXT carregado: {os.path.basename(path)}")

                if doc:
                    txt_preview.delete("1.0", "end")
                    txt_preview.insert("1.0", doc.text)
                    if doc.has_low_confidence_chords:
                        var_conf_warning.set(
                            f"⚠️ Atenção: {len(doc.low_confidence_tokens)} acorde(s) com confiança moderada (< 70%). "
                            "Por favor, revise o texto abaixo antes de confirmar."
                        )
                    else:
                        var_conf_warning.set("")
            except Exception as ex:
                dlg.config(cursor="")
                messagebox.showerror("Erro ao Carregar Cifra", f"Falha ao processar arquivo:\n{ex}", parent=dlg)

        btn_pick_file = tk.Button(
            row_file,
            text="📂 Escolher Arquivo (TXT / DOCX / PDF / PNG / JPG / WEBP)",
            bg="#202735",
            fg="#00d2ff",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            command=_on_choose_chart_file
        )
        btn_pick_file.pack(side="left", padx=(0, 8))

        # Botão OK duplicado no cabeçalho para máxima facilidade de acesso
        btn_top_ok = tk.Button(
            row_file,
            text="✓ OK — INSERIR NO SETLIST",
            bg="#00e676",
            fg="#0d1015",
            activebackground="#00c853",
            activeforeground="#0d1015",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=12,
            pady=4,
            cursor="hand2",
            command=_on_confirm_import
        )
        btn_top_ok.pack(side="right", padx=(8, 0))

        lbl_src_status = tk.Label(row_file, textvariable=var_source_info, bg="#181d26", fg="#8c9ba5",
                                  font=("Segoe UI", 8), anchor="w")
        lbl_src_status.pack(side="left", fill="x", expand=True)

        # Linha 2 do Cabeçalho: Título e Artista
        row_meta = tk.Frame(top_box, bg="#181d26")
        row_meta.pack(fill="x", pady=(0, 6))

        tk.Label(row_meta, text="Título:", bg="#181d26", fg="#f0f4f8", font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 4))
        ent_title = ttk.Entry(row_meta, textvariable=var_title, width=28, font=("Segoe UI", 9))
        ent_title.pack(side="left", padx=(0, 14))

        tk.Label(row_meta, text="Artista:", bg="#181d26", fg="#f0f4f8", font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 4))
        ent_artist = ttk.Entry(row_meta, textvariable=var_artist, width=24, font=("Segoe UI", 9))
        ent_artist.pack(side="left")

        # Linha 3 do Cabeçalho: Áudio de Acompanhamento (Opcional)
        row_audio = tk.Frame(top_box, bg="#181d26")
        row_audio.pack(fill="x")

        tk.Label(row_audio, text="Áudio (Opcional):", bg="#181d26", fg="#8c9ba5", font=("Segoe UI", 8)).pack(side="left", padx=(0, 4))
        ent_audio = ttk.Entry(row_audio, textvariable=var_audio_path, font=("Segoe UI", 8))
        ent_audio.pack(side="left", fill="x", expand=True, padx=(0, 6))

        def _on_pick_audio():
            ap = filedialog.askopenfilename(
                parent=dlg,
                title="Selecionar Áudio de Acompanhamento (Opcional)",
                filetypes=[("Arquivos de Áudio (*.wav;*.mp3)", "*.wav;*.mp3"), ("Todos os Arquivos (*.*)", "*.*")]
            )
            if ap:
                var_audio_path.set(ap)

        btn_pick_audio = tk.Button(row_audio, text="Buscar...", bg="#202735", fg="#ffffff", relief="flat",
                                   font=("Segoe UI", 8), cursor="hand2", command=_on_pick_audio)
        btn_pick_audio.pack(side="right")

        dlg.bind("<Control-Return>", lambda e: _on_confirm_import())
        dlg.bind("<Escape>", lambda e: dlg.destroy())

    def _on_save_project(self) -> None:
        try:
            path = self.project_manager.save_project()
            messagebox.showinfo("Projeto Salvo", f"Projeto salvo com sucesso em:\n{path}")
            self.lbl_status_msg.config(text=f"Projeto salvo: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Erro ao Salvar Projeto", f"Falha ao salvar:\n{e}")

    def _on_seek_bar_manual(self, delta: int = 0, to_start: bool = False) -> None:
        session = self.project_manager.active_session
        if not session:
            return
        if to_start:
            session.seek_to_start()
        else:
            session.seek_to_bar(session.current_bar + delta)
        self._update_playalong_hud_from_session(session)

    def _on_seek_chord_manual(self, delta: int = 0) -> None:
        session = self.project_manager.active_session
        if not session:
            return
        if delta > 0:
            session.next_chord_jump()
        else:
            session.prev_chord_jump()
        self._update_playalong_hud_from_session(session)

    def _on_seek_section_manual(self, delta: int = 0) -> None:
        session = self.project_manager.active_session
        if not session:
            return
        if delta > 0:
            session.next_section_jump()
        else:
            session.prev_section()
        self._update_playalong_hud_from_session(session)

    def _on_chart_key_release(self, event=None) -> None:
        """Rastreia digitação no editor de cifra para alertar sobre alterações pendentes."""
        if event and event.keysym in ("Control_L", "Control_R", "Shift_L", "Shift_R", "Alt_L", "Alt_R", "Caps_Lock", "Escape"):
            return
        self._has_unsaved_chart_edits = True
        if hasattr(self, "lbl_chart_edit_status"):
            self.lbl_chart_edit_status.config(text="● Alterações não salvas (Ctrl+S)", fg="#ffd166")
        if hasattr(self, "btn_save_chart"):
            self.btn_save_chart.config(bg="#ffd166", fg="#0d1015")

    def _on_save_chart_direct(self, event=None) -> None:
        """Salva o texto editado diretamente no text_chart_view para a música ativa e recalcula harmonia."""
        setlist = self.project_manager.project.get_active_setlist()
        active_song = setlist.get_active_song()
        if not active_song:
            messagebox.showinfo("Nenhuma Música Ativa", "Não há nenhuma música selecionada para salvar.")
            return

        new_text = self.text_chart_view.get("1.0", "end-1c").strip()
        if not new_text:
            messagebox.showwarning("Cifra Vazia", "A cifra não pode ficar completamente vazia.", parent=self.root)
            return

        selected_block = getattr(self, "_selected_block_index", -1)
        before = ChartEditState(active_song.chart_text, selected_block)
        try:
            updated_song = self.project_manager.update_song_chart(active_song.id, new_text)
        except Exception as exc:
            self._has_unsaved_chart_edits = True
            self._set_chart_status("Falha ao salvar; alterações mantidas no editor", "#ff5252")
            messagebox.showerror("Erro ao Salvar Cifra", str(exc), parent=self.root)
            return
        if hasattr(self, "_chart_edit_history"):
            self._chart_edit_history.record(
                before, ChartEditState(new_text, selected_block))
        self._has_unsaved_chart_edits = False
        if hasattr(self, "lbl_chart_edit_status"):
            self.lbl_chart_edit_status.config(text="✓ Cifra salva e sincronizada!", fg="#00e676")
        if hasattr(self, "btn_save_chart"):
            self.btn_save_chart.config(bg="#00e676", fg="#0d1015")
        self.lbl_status_msg.config(text=f"Cifra de '{active_song.title}' salva com sucesso!")

        # Reaplica coloração sintática de acordes e seções
        self._apply_chart_syntax_highlighting()

        # Atualiza título do cabeçalho caso tom/bpm tenham mudado
        if updated_song:
            self.lbl_playalong_song_title.config(
                text=f"{updated_song.title} — {updated_song.artist} ({updated_song.key})"
            )

        # Atualiza HUD da sessão ativa
        if self.project_manager.active_session:
            self._update_playalong_hud_from_session(self.project_manager.active_session)

    def _on_discard_chart_edits(self) -> None:
        """Descarta alterações não salvas e recarrega o texto original da canção ativa."""
        setlist = self.project_manager.project.get_active_setlist()
        active_song = setlist.get_active_song()
        if active_song:
            self._render_chart_text(active_song)
            self._has_unsaved_chart_edits = False
            if hasattr(self, "lbl_chart_edit_status"):
                self.lbl_chart_edit_status.config(text="✓ Alterações descartadas", fg="#8c9ba5")
            if hasattr(self, "btn_save_chart"):
                self.btn_save_chart.config(bg="#00e676", fg="#0d1015")

    def _apply_chart_syntax_highlighting(self) -> None:
        """Colore os acordes/letras e dá a cada BLOCO um FUNDO tingido pelo tipo de seção.

        O bloco selecionado (por clique) recebe um fundo mais forte para indicar a seleção.
        """
        v = self.text_chart_view
        for tag in ("section_tag", "chord_symbol", "lyric_line"):
            v.tag_remove(tag, "1.0", "end")
        for sec_type in SECTION_COLORS:
            for pref in ("secbg_", "sechd_", "secsel_", "secselhd_"):
                v.tag_remove(f"{pref}{sec_type}", "1.0", "end")

        text = v.get("1.0", "end-1c")
        total_lines = int(v.index("end-1c").split(".")[0])
        _pre, ranges = self._block_line_ranges(text)

        # Mapa linha -> (bloco, tipo) para saber o fundo de cada linha
        line_block = {}
        for idx, (s, e, t) in enumerate(ranges):
            for ln in range(s, e + 1):
                line_block[ln] = (idx, t)

        def full_line(ln):
            # Inclui a quebra de linha para o fundo cobrir a largura toda do bloco
            end = f"{ln + 1}.0" if ln < total_lines else f"{ln}.end"
            return f"{ln}.0", end

        for line_num in range(1, total_lines + 1):
            line_str = v.get(f"{line_num}.0", f"{line_num}.end")
            s_line = line_str.strip()

            # 1. Fundo do bloco (mesmo em linhas vazias, para o bloco ficar contínuo)
            if line_num in line_block:
                blk_idx, sec_type = line_block[line_num]
                selected = (blk_idx == self._selected_block_index)
                is_header = bool(ChartParser.parse_section_header(s_line))
                if selected:
                    bg_tag = f"secselhd_{sec_type}" if is_header else f"secsel_{sec_type}"
                else:
                    bg_tag = f"sechd_{sec_type}" if is_header else f"secbg_{sec_type}"
                a, b = full_line(line_num)
                v.tag_add(bg_tag, a, b)

            # 2. Cor do texto (acordes/letra) sobre o fundo
            if not s_line:
                continue
            if ChartParser.parse_section_header(s_line):
                continue  # cabeçalho já formatado pelo fundo/negrito
            if ChartParser.is_chord_line(s_line):
                v.tag_add("chord_symbol", f"{line_num}.0", f"{line_num}.end")
            else:
                v.tag_add("lyric_line", f"{line_num}.0", f"{line_num}.end")

        # Mantém os destaques de reprodução acima dos fundos de bloco
        for t in ("active_line", "active_chord"):
            try:
                v.tag_raise(t)
            except Exception:
                pass

    # ==================================================================
    # Tom (transposição) e Operações de Bloco
    # ==================================================================
    @staticmethod
    def _blend_hex(color: str, bg: str, t: float) -> str:
        """Mistura ``color`` sobre ``bg`` na proporção ``t`` (0..1). Retorna hex."""
        def rgb(h):
            h = h.lstrip("#")
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        cr, cg, cb = rgb(color)
        br, bg_, bb = rgb(bg)
        r = round(br + t * (cr - br))
        g = round(bg_ + t * (cg - bg_))
        b = round(bb + t * (cb - bb))
        return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"

    def _block_line_ranges(self, text: str):
        """Retorna (preamble_len, [(start,end,section_type), ...]) em linhas 1-based."""
        lines = text.split("\n")
        ranges = []
        preamble_len = 0
        cur_start = None
        cur_type = "UNKNOWN"
        seen_header = False
        for i, ln in enumerate(lines):
            sec = ChartParser.parse_section_header(ln.strip())
            if sec:
                if cur_start is not None:
                    ranges.append((cur_start, i, cur_type))  # i = linha 1-based do fim (exclusivo+1? veja abaixo)
                cur_start = i + 1
                cur_type = (sec[1] or "UNKNOWN").upper()
                if cur_type not in SECTION_COLORS:
                    cur_type = "UNKNOWN"
                seen_header = True
            elif not seen_header:
                preamble_len = i + 1
        if cur_start is not None:
            ranges.append((cur_start, len(lines), cur_type))
        # Ajusta os fins: cada bloco vai de start até (próximo start - 1)
        fixed = []
        for idx, (s, _e, t) in enumerate(ranges):
            end = (ranges[idx + 1][0] - 1) if idx + 1 < len(ranges) else len(lines)
            fixed.append((s, end, t))
        return preamble_len, fixed

    def _on_chart_click(self, event=None) -> None:
        """Seleciona o bloco clicado (destaca o fundo) para deletar/duplicar/mover."""
        try:
            clicked = int(self.text_chart_view.index(f"@{event.x},{event.y}").split(".")[0])
        except Exception:
            return
        _pre, ranges = self._block_line_ranges(self.text_chart_view.get("1.0", "end-1c"))
        self._selected_block_index = -1
        for idx, (s, e, _t) in enumerate(ranges):
            if s <= clicked <= e:
                self._selected_block_index = idx
                break
        self._apply_chart_syntax_highlighting()
        if self._selected_block_index >= 0:
            self._set_chart_status(f"Bloco {self._selected_block_index + 1} selecionado", "#00d2ff")

    @staticmethod
    def _all_key_options() -> list:
        """Lista de tons para o seletor (12 maiores + 12 menores)."""
        roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return [f"{r} Major" for r in roots] + [f"{r} Minor" for r in roots]

    def _active_song(self) -> Optional[Song]:
        """Retorna a música ativa (da sessão ou do setlist ativo), se houver."""
        try:
            if self.project_manager.active_session:
                return self.project_manager.active_session.song
            active_set = self.project_manager.get_project().get_active_setlist()
            if active_set:
                return active_set.get_active_song()
        except Exception:
            pass
        return None

    @staticmethod
    def _source_label(options: dict, value: str) -> str:
        return next((label for label, code in options.items() if code == value), "")

    def _sync_reference_sources(self, song: Optional[Song] = None) -> None:
        """Reflete no Laboratório as preferências persistidas da música ativa."""
        song = song or self._active_song()
        if song is None:
            return
        settings = song.performance_settings
        if hasattr(self, "combo_bpm_source"):
            self.combo_bpm_source.set(self._source_label(
                self.BPM_SOURCES, getattr(settings, "bpm_source", "AUDIO").upper()))
        if hasattr(self, "combo_key_source"):
            self.combo_key_source.set(self._source_label(
                self.KEY_SOURCES, getattr(settings, "key_source", "CHART").upper()))

    def _save_reference_sources(self) -> None:
        if self.project_manager.project.settings.auto_save:
            self.project_manager.save_project()

    def _on_bpm_source_changed(self, event=None) -> None:
        session = self.project_manager.active_session
        source = self.BPM_SOURCES.get(self.combo_bpm_source.get())
        if session is None or source is None:
            return
        session.set_bpm_source(source)
        self._save_reference_sources()
        description = "áudio da música" if source == "AUDIO" else "BPM da cifra"
        self.lbl_status_msg.config(text=f"BPM seguindo {description}.")

    def _on_key_source_changed(self, event=None) -> None:
        session = self.project_manager.active_session
        source = self.KEY_SOURCES.get(self.combo_key_source.get())
        if session is None or source is None:
            return
        session.set_key_source(source)
        self._save_reference_sources()
        description = "áudio da música" if source == "AUDIO" else "tom da cifra"
        self.lbl_status_msg.config(text=f"Tom seguindo {description}.")

    def _set_chart_status(self, text: str, fg: str) -> None:
        if hasattr(self, "lbl_chart_edit_status"):
            self.lbl_chart_edit_status.config(text=text, fg=fg)

    def _on_change_key(self, event=None) -> None:
        """Transpõe a cifra inteira da música ativa para o tom selecionado."""
        song = self._active_song()
        target = self.combo_key.get().strip()
        if not song or not target:
            self._set_chart_status("Nenhuma música ativa para transpor", "#ffb703")
            return
        try:
            before = ChartEditState(song.chart_text, self._selected_block_index)
            self.project_manager.transpose_song(song, target)
            after = ChartEditState(song.chart_text, self._selected_block_index)
            self._chart_edit_history.record(before, after)
            self._render_chart_text(song, reset_history=False)
            self._set_chart_status(f"✓ Transposto para {target}", "#00e676")
        except Exception as exc:
            self._set_chart_status(f"Erro ao transpor: {exc}", "#ff5252")

    def _split_text_blocks(self, text: str):
        """Divide o texto em blocos que começam num cabeçalho de seção.

        Retorna (preamble_lines, blocks) onde cada bloco é uma lista de linhas
        iniciada pelo seu cabeçalho [ ]. Linhas antes do 1º cabeçalho ficam no preâmbulo.
        """
        lines = text.split("\n")
        preamble = []
        blocks = []
        current = None
        for ln in lines:
            if ChartParser.parse_section_header(ln.strip()):
                if current is not None:
                    blocks.append(current)
                current = [ln]
            elif current is None:
                preamble.append(ln)
            else:
                current.append(ln)
        if current is not None:
            blocks.append(current)
        return preamble, blocks

    def _current_block_index(self, blocks) -> int:
        """Índice do bloco onde está o cursor no editor (0-based)."""
        try:
            cursor_line = int(self.text_chart_view.index("insert").split(".")[0])
        except Exception:
            return len(blocks) - 1 if blocks else -1
        # Reconstrói limites de linha a partir do preâmbulo + blocos
        preamble, _ = self._split_text_blocks(self.text_chart_view.get("1.0", "end-1c"))
        line = len(preamble) + 1
        for idx, blk in enumerate(blocks):
            start = line
            end = line + len(blk) - 1
            if start <= cursor_line <= end:
                return idx
            line = end + 1
        return len(blocks) - 1 if blocks else -1

    def _rebuild_text_from_blocks(self, preamble, blocks) -> str:
        parts = []
        if any(l.strip() for l in preamble):
            parts.append("\n".join(preamble).rstrip())
            parts.append("")
        for blk in blocks:
            parts.append("\n".join(blk).rstrip())
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    def _persist_chart_text(self, song: Song, new_text: str) -> None:
        """Grava o novo texto da cifra na música ativa e re-renderiza."""
        self.project_manager.update_song_chart(song.id, new_text)
        self._render_chart_text(song)

    def _target_block_index(self, blocks) -> int:
        """Índice do bloco alvo: o selecionado por clique, senão o do cursor."""
        if 0 <= self._selected_block_index < len(blocks):
            return self._selected_block_index
        return self._current_block_index(blocks)

    def _commit_blocks(self, preamble, blocks, status: str) -> None:
        """Reconstrói o texto, persiste (se houver música) e re-renderiza, mantendo a seleção."""
        keep_sel = self._selected_block_index
        before_text = self.text_chart_view.get("1.0", "end-1c")
        new_text = self._rebuild_text_from_blocks(preamble, blocks)
        song = self._active_song()
        before = ChartEditState(before_text, self._target_block_index(blocks) if blocks else -1)
        after = ChartEditState(new_text, keep_sel if keep_sel is not None else -1)
        if song:
            self.project_manager.update_song_chart(song.id, new_text)
            self._render_chart_text(song, reset_history=False)  # zera a seleção
        else:
            self.text_chart_view.delete("1.0", "end")
            self.text_chart_view.insert("1.0", new_text)
        self._chart_edit_history.record(before, after)
        # Restaura a seleção do bloco resultante e re-destaca
        self._selected_block_index = keep_sel if keep_sel is not None else -1
        self._apply_chart_syntax_highlighting()
        self._set_chart_status(status, "#00e676")

    def _apply_chart_edit_state(self, state: ChartEditState, status: str) -> None:
        """Aplica snapshot pela persistência oficial, reconstruindo toda a sessão."""
        song = self._active_song()
        if song:
            self.project_manager.update_song_chart(song.id, state.chart_text)
            self._render_chart_text(song, reset_history=False)
        else:
            self.text_chart_view.delete("1.0", "end")
            self.text_chart_view.insert("1.0", state.chart_text)
        self._selected_block_index = state.selected_block_index
        self._apply_chart_syntax_highlighting()
        self._has_unsaved_chart_edits = False
        self._set_chart_status(status, "#00e676")

    def _undo_structural_edit(self) -> bool:
        state = self._chart_edit_history.undo()
        if state is None:
            self._set_chart_status("Nada para desfazer", "#8c9ba5")
            return False
        try:
            self._apply_chart_edit_state(state, "✓ Alteração desfeita e sincronizada")
        except Exception as exc:
            self._chart_edit_history.redo()
            self._set_chart_status(f"Falha ao desfazer: {exc}", "#ff5252")
            return False
        return True

    def _redo_structural_edit(self) -> bool:
        state = self._chart_edit_history.redo()
        if state is None:
            self._set_chart_status("Nada para refazer", "#8c9ba5")
            return False
        try:
            self._apply_chart_edit_state(state, "✓ Alteração refeita e sincronizada")
        except Exception as exc:
            self._chart_edit_history.undo()
            self._set_chart_status(f"Falha ao refazer: {exc}", "#ff5252")
            return False
        return True

    def _on_undo_shortcut(self, event=None):
        """Undo nativo no texto; histórico estrutural no restante da aplicação."""
        focused = self.root.focus_get()
        if focused is self.text_chart_view:
            try:
                self.text_chart_view.edit_undo()
                self._on_chart_key_release()
                self._apply_chart_syntax_highlighting()
                return "break"
            except tk.TclError:
                pass
        elif isinstance(focused, (tk.Text, tk.Entry, ttk.Entry)):
            return None
        self._undo_structural_edit()
        return "break"

    def _on_redo_shortcut(self, event=None):
        """Redo nativo no texto; histórico estrutural no restante da aplicação."""
        focused = self.root.focus_get()
        if focused is self.text_chart_view:
            try:
                self.text_chart_view.edit_redo()
                self._on_chart_key_release()
                self._apply_chart_syntax_highlighting()
                return "break"
            except tk.TclError:
                pass
        elif isinstance(focused, (tk.Text, tk.Entry, ttk.Entry)):
            return None
        self._redo_structural_edit()
        return "break"

    def _on_copy_block(self) -> None:
        """Duplica o bloco selecionado (ou o do cursor), inserindo a cópia logo após."""
        preamble, blocks = self._split_text_blocks(self.text_chart_view.get("1.0", "end-1c"))
        if not blocks:
            self._set_chart_status("Clique num bloco para duplicar", "#ffb703")
            return
        idx = self._target_block_index(blocks)
        if idx < 0:
            idx = len(blocks) - 1
        blocks.insert(idx + 1, list(blocks[idx]))
        self._selected_block_index = idx + 1  # segue a cópia
        self._commit_blocks(preamble, blocks, "✓ Bloco duplicado")

    def _on_delete_block(self) -> None:
        """Remove o bloco selecionado (ou o do cursor)."""
        preamble, blocks = self._split_text_blocks(self.text_chart_view.get("1.0", "end-1c"))
        if not blocks:
            self._set_chart_status("Clique num bloco para deletar", "#ffb703")
            return
        idx = self._target_block_index(blocks)
        if idx < 0:
            self._set_chart_status("Nenhum bloco selecionado", "#ffb703")
            return
        del blocks[idx]
        self._selected_block_index = -1
        self._commit_blocks(preamble, blocks, "✓ Bloco deletado")

    def _on_move_block(self, delta: int) -> None:
        """Move o bloco selecionado (ou o do cursor) para cima (-1) ou para baixo (+1)."""
        preamble, blocks = self._split_text_blocks(self.text_chart_view.get("1.0", "end-1c"))
        if len(blocks) < 2:
            self._set_chart_status("Nada para reordenar", "#ffb703")
            return
        idx = self._target_block_index(blocks)
        new_idx = idx + delta
        if idx < 0 or not (0 <= new_idx < len(blocks)):
            self._set_chart_status("Bloco no limite", "#ffb703")
            return
        blocks[idx], blocks[new_idx] = blocks[new_idx], blocks[idx]
        self._selected_block_index = new_idx  # a seleção acompanha o bloco
        self._commit_blocks(preamble, blocks, f"✓ Bloco movido {'▲' if delta < 0 else '▼'}")

    def _render_chart_text(self, song: Song, reset_history: bool = True) -> None:
        """Renderiza o texto da cifra formatado com tags coloridas mantendo o editor liberado para edição direta."""
        self._selected_block_index = -1  # nova renderização começa sem bloco selecionado
        self.text_chart_view.config(state="normal")
        self.text_chart_view.delete("1.0", "end")

        txt = song.chart_text
        if not txt and song.chart_data:
            lines = []
            for sec in song.chart_data.get("sections", []):
                lines.append(f"\n[{sec.get('name', 'Seção')}]")
                chords_str = "        ".join(c.get("symbol", {}).get("original_symbol", "") for c in sec.get("chords", []))
                lines.append(chords_str)
                for l in sec.get("lyrics", []):
                    lines.append(l.get("text", ""))
            txt = "\n".join(lines)

        if not txt:
            txt = "(Sem cifra cadastrada para esta canção)\nDigite sua cifra diretamente aqui ou use o botão '➕ Importar / Trocar Cifra'."

        self.text_chart_view.insert("1.0", txt)
        self._apply_chart_syntax_highlighting()
        try:
            self.text_chart_view.edit_reset()
        except Exception:
            pass
        # Sincroniza o seletor de tom com a tonalidade atual da música
        if hasattr(self, "combo_key") and getattr(song, "key", None):
            try:
                key = song.key if song.key in self._all_key_options() else None
                if key is None:
                    root, mode = parse_key_root_mode(song.key)
                    key = f"{root} {'Minor' if mode == 'minor' else 'Major'}"
                self.combo_key.set(key)
            except Exception:
                pass
        self._sync_reference_sources(song)
        self._has_unsaved_chart_edits = False
        if reset_history:
            self._chart_edit_history.reset(ChartEditState(txt, -1))
        if hasattr(self, "lbl_chart_edit_status"):
            self.lbl_chart_edit_status.config(text="✓ Sincronizado", fg="#00e676")
        if hasattr(self, "btn_save_chart"):
            self.btn_save_chart.config(bg="#00e676", fg="#0d1015")
        # Mantém state="normal" para o usuário poder editar diretamente!

    def _highlight_chart_position(self, chart_pos: Any) -> None:
        """Localiza e destaca visualmente a posição e o acorde atual no texto da cifra."""
        if not chart_pos:
            return

        self.text_chart_view.tag_remove("active_chord", "1.0", "end")
        self.text_chart_view.tag_remove("active_line", "1.0", "end")

        line_idx = max(1, getattr(chart_pos, "line_index", 1))
        target_chord = getattr(chart_pos, "current_chord", "--")

        found_chord = False
        if target_chord and target_chord != "--":
            # 1. Procura primeiro na linha exata da posição
            line_start = f"{line_idx}.0"
            line_end = f"{line_idx}.end"
            idx = self.text_chart_view.search(target_chord, line_start, nocase=False, stopindex=line_end)
            if idx:
                end_idx = f"{idx}+{len(target_chord)}c"
                self.text_chart_view.tag_add("active_chord", idx, end_idx)
                found_chord = True
            else:
                # 2. Janela local de +/- 1 linha
                for offset in (-1, 1):
                    cand_line = line_idx + offset
                    if cand_line >= 1:
                        c_start = f"{cand_line}.0"
                        c_end = f"{cand_line}.end"
                        c_idx = self.text_chart_view.search(target_chord, c_start, nocase=False, stopindex=c_end)
                        if c_idx:
                            end_idx = f"{c_idx}+{len(target_chord)}c"
                            self.text_chart_view.tag_add("active_chord", c_idx, end_idx)
                            found_chord = True
                            break

        # Se for linha de letra avulsa ou intervalo sem acorde, destaca a linha ativa suavemente
        if not found_chord:
            line_start = f"{line_idx}.0"
            line_end = f"{line_idx}.end"
            self.text_chart_view.tag_add("active_line", line_start, line_end)

        # Auto-scroll contínuo baseado na linha real
        if self._follow_mode_enabled:
            try:
                if self.root.focus_get() != self.text_chart_view:
                    self.text_chart_view.see(f"{line_idx}.0")
            except Exception:
                self.text_chart_view.see(f"{line_idx}.0")

    def _update_playalong_hud_from_session(self, session: SongSession) -> None:
        """Atualiza os indicadores do HUD a partir do estado corrente da SongSession."""
        self.lbl_playalong_section.config(text=session.current_section)
        self.lbl_playalong_chord.config(text=session.current_chord)
        self.lbl_playalong_next_chord.config(text=session.next_chord)
        self.lbl_playalong_bar_beat.config(
            text=f"Comp. {session.current_bar}  |  Tempo {session.current_beat} / {session.clock.beats_per_bar}"
        )
        self.lbl_playalong_bpm.config(text=f"{session.clock.bpm:.1f} BPM")

        # Indicador visual avançado de Rastreamento (TrackingState)
        tracking_st = getattr(session, "tracking_state", "TRACKING")
        pos_conf = getattr(session, "position_confidence", 0.90)
        performance_st = getattr(session, "performance_state", "PLAYING")

        if tracking_st == "TRACKING":
            st_color = "#00e676"
            st_text = f"TRACKING ({int(pos_conf * 100)}%)"
        elif tracking_st == "UNCERTAIN":
            st_color = "#ffd166"
            st_text = f"INCERTO ({int(pos_conf * 100)}%)"
        elif tracking_st == "RECOVERING":
            st_color = "#00d2ff"
            st_text = f"RECUPERANDO ({int(pos_conf * 100)}%)"
        else: # LOST
            st_color = "#ff7b72"
            st_text = f"TEMPORAL ({int(pos_conf * 100)}%)"

        if performance_st != "PLAYING":
            st_text = f"{performance_st} | {st_text}"

        if session.fusion.discrepancies and session.fusion.discrepancies[-1].classification == "possible_performance_variation":
            last_disc = session.fusion.discrepancies[-1]
            if last_disc.duration > 1.0:
                self.lbl_playalong_fusion.config(text=st_text, fg=st_color)
                self.lbl_playalong_fusion_sub.config(text=f"Variação: {last_disc.detected_chord} (Esperado: {last_disc.expected_chord})", fg="#ffd166")
            else:
                self.lbl_playalong_fusion.config(text=st_text, fg=st_color)
                self.lbl_playalong_fusion_sub.config(text=f"Áudio: {session.detected_chord} | Cifra: {session.expected_chord}", fg="#8c9ba5")
        else:
            self.lbl_playalong_fusion.config(text=st_text, fg=st_color)
            self.lbl_playalong_fusion_sub.config(text=f"Áudio: {session.detected_chord} | Cifra: {session.expected_chord}", fg="#8c9ba5")

        if self._follow_mode_enabled:
            self._highlight_chart_position(session.chart_position)
        if getattr(self, "_bass_debug_visible", False):
            self.lbl_status_msg.config(text=session.format_position_diagnostics() + " | " +
                                       session.format_tempo_diagnostics())


    def _on_close(self) -> None:
        try:
            self.player.close()
        except Exception:
            pass
        self.root.destroy()
