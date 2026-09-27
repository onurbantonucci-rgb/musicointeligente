"""Painel do Baixista Virtual (extraído de MainWindow v0.4 — split v0.8).

Controla e visualiza o BassPlayer: ativação, padrão rítmico, fonte harmônica,
volume, e o log de decisões (debug). Todo o estado e os widgets deste painel
são internos à classe; MainWindow só chama ``update(ctx)`` a cada tick e
``reset_debug_table()`` ao carregar um novo áudio.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from app.instruments import BassPatternType, BassNoteValue, BassHarmonySource
from app.music.theory import midi_to_hz


class BassPlayerPanel(tk.Frame):
    """Aba "🎸 Baixista Virtual" — widgets, handlers e atualização por ciclo."""

    PATTERNS = [
        "FUNDAMENTAIS",
        "AUTO",
        "ROOT + FIFTH",
        "ROOT + FIFTH + OCTAVE",
        "SUSTAINED",
    ]

    PATTERN_MAP = {
        "FUNDAMENTAIS": BassPatternType.FUNDAMENTALS,
        "AUTO": BassPatternType.AUTO,
        "ROOT": BassPatternType.ROOT,
        "ROOT + FIFTH": BassPatternType.ROOT_FIFTH,
        "ROOT + FIFTH + OCTAVE": BassPatternType.ROOT_FIFTH_OCTAVE,
        "SUSTAINED": BassPatternType.SUSTAINED,
    }

    NOTE_VALUES = {
        "Semibreve — a cada 4 tempos": BassNoteValue.WHOLE,
        "Mínima — a cada 2 tempos": BassNoteValue.HALF,
        "Semínima — a cada tempo": BassNoteValue.QUARTER,
        "Colcheia — 2x por tempo": BassNoteValue.EIGHTH,
        "Semicolcheia — 4x por tempo": BassNoteValue.SIXTEENTH,
    }

    HARMONY_SOURCES = {
        "Seguir sempre a cifra": BassHarmonySource.CHART,
        "Cifra + confirmação do áudio": BassHarmonySource.FOLLOW,
    }

    def __init__(self, parent: tk.Widget, analyzer, status_callback: Optional[Callable[[str], None]] = None):
        super().__init__(parent, bg="#181d26", bd=0, padx=8, pady=4)
        self._analyzer = analyzer
        self._status_callback = status_callback or (lambda msg: None)
        self._bass_debug_visible = True
        self._last_rendered_events_count = 0
        self._build()

    def _set_status(self, msg: str) -> None:
        self._status_callback(msg)

    # ------------------------------------------------------------------
    # Construção dos widgets (idêntica à antiga MainWindow._build_bass_player_content)
    # ------------------------------------------------------------------
    def _build(self) -> None:
        self.pack(fill="both", expand=True)

        # -------------------------------------------------------------
        # 1. Barra Superior de Controles do Baixo
        # -------------------------------------------------------------
        ctrl_bar = tk.Frame(self, bg="#181d26")
        ctrl_bar.pack(fill="x", pady=(0, 4))

        lbl_bass_title = tk.Label(
            ctrl_bar,
            text="🎸 BAIXISTA VIRTUAL (BASSPLAYER)",
            bg="#181d26",
            fg="#00e676",
            font=("Segoe UI", 9, "bold")
        )
        lbl_bass_title.pack(side="left", padx=(0, 10))

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

        tk.Label(ctrl_bar, text="Padrão:", bg="#181d26", fg="#8c9ba5",
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 4))

        self.combo_bass_pattern = ttk.Combobox(
            ctrl_bar,
            values=self.PATTERNS,
            state="readonly",
            width=22,
            font=("Segoe UI", 8)
        )
        self.combo_bass_pattern.set("FUNDAMENTAIS")
        self.combo_bass_pattern.pack(side="left", padx=(0, 14))
        self.combo_bass_pattern.bind("<<ComboboxSelected>>", self._on_pattern_changed)

        tk.Label(ctrl_bar, text="Volume Baixo:", bg="#181d26", fg="#8c9ba5",
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 4))

        self.bass_vol_var = tk.DoubleVar(value=85.0)
        self.scale_bass_vol = ttk.Scale(
            ctrl_bar,
            from_=0.0,
            to=100.0,
            orient="horizontal",
            variable=self.bass_vol_var,
            command=self._on_volume_changed,
            length=100
        )
        self.scale_bass_vol.pack(side="left", padx=(0, 6))

        self.lbl_bass_vol_val = tk.Label(ctrl_bar, text="85%", bg="#181d26", fg="#00d2ff",
                                         font=("Consolas", 8, "bold"), width=4)
        self.lbl_bass_vol_val.pack(side="left", padx=(0, 10))

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
            command=self._toggle_debug
        )
        self.btn_bass_debug_toggle.pack(side="right")

        rhythm_bar = tk.Frame(self, bg="#181d26")
        rhythm_bar.pack(fill="x", pady=(0, 5))
        tk.Label(rhythm_bar, text="Fonte das notas:", bg="#181d26", fg="#8c9ba5",
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 6))
        self.combo_bass_harmony_source = ttk.Combobox(
            rhythm_bar, values=list(self.HARMONY_SOURCES), state="readonly",
            width=27, font=("Segoe UI", 8))
        self.combo_bass_harmony_source.set("Seguir sempre a cifra")
        self.combo_bass_harmony_source.pack(side="left", padx=(0, 12))
        self.combo_bass_harmony_source.bind("<<ComboboxSelected>>", self._on_harmony_source_changed)
        tk.Label(rhythm_bar, text="Repetição da tônica:", bg="#181d26", fg="#8c9ba5",
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 6))
        self.combo_bass_note_value = ttk.Combobox(
            rhythm_bar, values=list(self.NOTE_VALUES), state="readonly",
            width=29, font=("Segoe UI", 8))
        self.combo_bass_note_value.set("Mínima — a cada 2 tempos")
        self.combo_bass_note_value.pack(side="left", padx=(0, 12))
        self.combo_bass_note_value.bind("<<ComboboxSelected>>", self._on_note_value_changed)
        tk.Label(rhythm_bar, text="O BPM e a batida do músico definem os instantes.",
                 bg="#181d26", fg="#8c9ba5", font=("Segoe UI", 8)).pack(side="left")

        # -------------------------------------------------------------
        # 2. Quatro Cartões de Métricas do Baixista
        # -------------------------------------------------------------
        cards_frame = tk.Frame(self, bg="#181d26")
        cards_frame.pack(fill="x", pady=2)
        cards_frame.columnconfigure((0, 1, 2, 3), weight=1, uniform="bass_cards")

        c1 = tk.Frame(cards_frame, bg="#141820", bd=1, relief="solid", padx=8, pady=4)
        c1.grid(row=0, column=0, sticky="nsew", padx=2)
        tk.Label(c1, text="NOTA SOANDO DO BAIXO", bg="#141820", fg="#00e676", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_bass_note = tk.Label(c1, text="--", bg="#141820", fg="#5a6677", font=("Segoe UI", 18, "bold"))
        self.lbl_bass_note.pack()
        self.lbl_bass_freq = tk.Label(c1, text="MIDI --  |  -- Hz", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 7))
        self.lbl_bass_freq.pack()

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

        c3 = tk.Frame(cards_frame, bg="#141820", bd=1, relief="solid", padx=8, pady=4)
        c3.grid(row=0, column=2, sticky="nsew", padx=2)
        tk.Label(c3, text="PADRÃO EM EXECUÇÃO", bg="#141820", fg="#00d2ff", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_bass_pattern_active = tk.Label(c3, text="FUNDAMENTAIS", bg="#141820", fg="#ffffff", font=("Segoe UI", 11, "bold"))
        self.lbl_bass_pattern_active.pack(pady=2)
        self.lbl_bass_sub_pattern = tk.Label(c3, text="Adaptativo às mudanças", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 7))
        self.lbl_bass_sub_pattern.pack()

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
        self.frame_bass_debug = tk.Frame(self, bg="#141820", bd=1, relief="solid")
        self.frame_bass_debug.pack(fill="x", pady=(4, 0))

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

    # ------------------------------------------------------------------
    # Handlers (idênticos à antiga MainWindow, adaptados para self._analyzer)
    # ------------------------------------------------------------------
    def _toggle_bass(self) -> None:
        new_state = not self._analyzer.bass_player.enabled
        self._analyzer.bass_player.enabled = new_state
        if new_state:
            self.btn_bass_toggle.config(
                text="🎸 BAIXO: ATIVADO",
                bg="#0f3424",
                fg="#00e676",
                activebackground="#144d34",
                activeforeground="#00e676"
            )
            self._set_status("Baixista Virtual ATIVADO e acompanhando a harmonia.")
        else:
            self.btn_bass_toggle.config(
                text="🔇 BAIXO: SILENCIADO",
                bg="#341818",
                fg="#ff5252",
                activebackground="#4a2222",
                activeforeground="#ff5252"
            )
            self._set_status("Baixista Virtual SILENCIADO.")

    def _on_pattern_changed(self, event=None) -> None:
        val = self.combo_bass_pattern.get()
        p = self.PATTERN_MAP.get(val, BassPatternType.AUTO)
        self._analyzer.bass_player.pattern = p
        self._set_status(f"Padrão rítmico do baixo alterado para: {val}")

    def _on_note_value_changed(self, event=None) -> None:
        label = self.combo_bass_note_value.get()
        self._analyzer.bass_player.note_value = self.NOTE_VALUES[label]
        self._set_status(f"Repetição da tônica: {label}")

    def _on_harmony_source_changed(self, event=None) -> None:
        label = self.combo_bass_harmony_source.get()
        self._analyzer.bass_player.harmony_source = self.HARMONY_SOURCES[label]
        self._set_status(f"Fonte das notas do baixo: {label}")

    def _on_volume_changed(self, val) -> None:
        try:
            vol_val = float(val)
            self._analyzer.bass_player.volume = vol_val / 100.0
            self.lbl_bass_vol_val.config(text=f"{int(vol_val)}%")
        except ValueError:
            pass

    def _toggle_debug(self) -> None:
        if self._bass_debug_visible:
            self.frame_bass_debug.pack_forget()
            self._bass_debug_visible = False
            self.btn_bass_debug_toggle.config(text="📋 Log de Decisões (Debug) ▶")
        else:
            self.frame_bass_debug.pack(fill="x", pady=(4, 0))
            self._bass_debug_visible = True
            self.btn_bass_debug_toggle.config(text="📋 Log de Decisões (Debug) ▼")

    # ------------------------------------------------------------------
    # Atualização por ciclo (chamada pelo _ui_update_loop de MainWindow)
    # ------------------------------------------------------------------
    def refresh_debug_table(self) -> None:
        """Atualiza a tabela gráfica com os eventos do Baixista Virtual."""
        events = self._analyzer.bass_player.get_recent_events()
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
                chord_txt = self._analyzer.context.chord if self._analyzer.context.chord != "--" else "--"
                self.tree_bass_debug.insert("", "end", values=(
                    f"#{ev.bar}",
                    f"{ev.beat} / 4",
                    chord_txt,
                    ev.note,
                    f"{ev.midi_note} ({freq:.1f}Hz)",
                    self._analyzer.bass_player.pattern.value.upper(),
                    ev.reason
                ))

            updated_children = self.tree_bass_debug.get_children()
            if updated_children:
                self.tree_bass_debug.see(updated_children[-1])

            self._last_rendered_events_count = total_count

    def reset_debug_table(self) -> None:
        """Limpa a tabela de decisões ao carregar um novo áudio."""
        for item in self.tree_bass_debug.get_children():
            self.tree_bass_debug.delete(item)
        self._last_rendered_events_count = 0

    def update(self, ctx) -> None:
        """Atualiza os cartões de estado do baixista a partir do MusicalContext atual."""
        bass = self._analyzer.bass_player
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
        else:
            self.lbl_bass_note.config(text="MUTED", fg="#ff5252")
            self.lbl_bass_freq.config(text="Sintetizador desativado")
            self.lbl_bass_reason.config(text="Baixista silenciado pelo usuário", fg="#8c9ba5")
            self.lbl_bass_next_note.config(text="--", fg="#5a6677")
            self.lbl_bass_next_timing.config(text="Silenciado", fg="#5a6677")

    def reset_display(self) -> None:
        """Reseta os cartões de estado (usado quando a reprodução para)."""
        self.lbl_bass_note.config(text="--", fg="#5a6677")
        self.lbl_bass_freq.config(text="MIDI --  |  -- Hz", fg="#5a6677")
        self.lbl_bass_reason.config(text="Aguardando reprodução...", fg="#5a6677")
        self.lbl_bass_next_note.config(text="--", fg="#5a6677")
        self.lbl_bass_next_timing.config(text="Aguardando harmonia", fg="#5a6677")
