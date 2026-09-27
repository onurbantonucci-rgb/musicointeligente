"""Painel de Estrutura Musical & Predição (extraído de MainWindow v0.3 — split v0.8).

Mostra a seção/padrão atual, a previsão da próxima seção e dos próximos acordes
(look-ahead harmônico), a forma global da música, e as tabelas de padrões
conhecidos (PatternMemory) e da matriz de transição entre seções.

Nota: a timeline colorida de estrutura (``canvas_timeline``) fica em outra aba
("Laboratório de Áudio & DSP") e continua sendo desenhada por MainWindow, que é
quem possui aquele widget — este painel expõe só o que é seu.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, Optional


class StructurePredictionPanel(tk.Frame):
    """Aba "🏛️ Estrutura & Previsão" — widgets, handlers e atualização por ciclo."""

    def __init__(self, parent: tk.Widget, analyzer, status_callback: Optional[Callable[[str], None]] = None):
        super().__init__(parent, bg="#181d26", padx=8, pady=4)
        self._analyzer = analyzer
        self._status_callback = status_callback or (lambda msg: None)
        self._build()

    def _set_status(self, msg: str) -> None:
        self._status_callback(msg)

    # ------------------------------------------------------------------
    # Construção dos widgets (idêntica à antiga MainWindow._build_structure_prediction_content)
    # ------------------------------------------------------------------
    def _build(self) -> None:
        self.pack(fill="both", expand=True)

        # 1. Barra superior: Título e Botão de Exportar
        top_bar = tk.Frame(self, bg="#181d26")
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
            command=self._on_export_report
        )
        self.btn_export_report.pack(side="right")

        # 2. Quatro Cartões de Resumo
        cards_frame = tk.Frame(self, bg="#181d26")
        cards_frame.pack(fill="x", pady=2)
        cards_frame.columnconfigure((0, 1, 2, 3), weight=1, uniform="struct_cards")

        sc1 = tk.Frame(cards_frame, bg="#141820", bd=1, relief="solid", padx=6, pady=3)
        sc1.grid(row=0, column=0, sticky="nsew", padx=2)
        tk.Label(sc1, text="SEÇÃO ATUAL & PADRÃO", bg="#141820", fg="#00d2ff", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_struct_cur_section = tk.Label(sc1, text="--", bg="#141820", fg="#5a6677", font=("Segoe UI", 16, "bold"))
        self.lbl_struct_cur_section.pack(pady=1)
        self.lbl_struct_progress = tk.Label(sc1, text="Progresso: --%  |  Comp. --", bg="#141820", fg="#00e676", font=("Segoe UI", 8))
        self.lbl_struct_progress.pack()

        sc2 = tk.Frame(cards_frame, bg="#141820", bd=1, relief="solid", padx=6, pady=3)
        sc2.grid(row=0, column=1, sticky="nsew", padx=2)
        tk.Label(sc2, text="PRÓXIMA SEÇÃO (PREVISTA)", bg="#141820", fg="#ffd166", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_pred_next_section = tk.Label(sc2, text="--", bg="#141820", fg="#5a6677", font=("Segoe UI", 16, "bold"))
        self.lbl_pred_next_section.pack(pady=1)
        self.lbl_pred_timing = tk.Label(sc2, text="Em -- compassos", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 8))
        self.lbl_pred_timing.pack()

        sc3 = tk.Frame(cards_frame, bg="#141820", bd=1, relief="solid", padx=6, pady=3)
        sc3.grid(row=0, column=2, sticky="nsew", padx=2)
        tk.Label(sc3, text="LOOK-AHEAD HARMÔNICO (1-4C)", bg="#141820", fg="#00e676", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_pred_lookahead = tk.Label(sc3, text="--", bg="#141820", fg="#5a6677", font=("Segoe UI", 13, "bold"))
        self.lbl_pred_lookahead.pack(pady=2)
        self.lbl_pred_reason = tk.Label(sc3, text="Aguardando harmonia", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 7))
        self.lbl_pred_reason.pack()

        sc4 = tk.Frame(cards_frame, bg="#141820", bd=1, relief="solid", padx=6, pady=3)
        sc4.grid(row=0, column=3, sticky="nsew", padx=2)
        tk.Label(sc4, text="FORMA GLOBAL DA MÚSICA", bg="#141820", fg="#c77dff", font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.lbl_struct_form = tk.Label(sc4, text="--", bg="#141820", fg="#5a6677", font=("Segoe UI", 14, "bold"))
        self.lbl_struct_form.pack(pady=1)
        self.lbl_struct_summary = tk.Label(sc4, text="0 seções  |  0 padrões", bg="#141820", fg="#8c9ba5", font=("Segoe UI", 7))
        self.lbl_struct_summary.pack()

        # 3. Tabelas Lado a Lado: Padrões Conhecidos & Matriz de Transições
        tables_frame = tk.Frame(self, bg="#181d26")
        tables_frame.pack(fill="both", expand=True, pady=(3, 0))
        tables_frame.columnconfigure((0, 1), weight=1, uniform="struct_tables")

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

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _on_export_report(self) -> None:
        """Salva o relatório estrutural JSON completo da música."""
        try:
            filepath = filedialog.asksaveasfilename(
                title="Exportar Relatório de Estrutura Musical (JSON)",
                defaultextension=".json",
                initialfile="music_structure_report.json",
                filetypes=[("Arquivos JSON (*.json)", "*.json"), ("Todos os Arquivos (*.*)", "*.*")]
            )
            if filepath:
                self._analyzer.export_structure_report(filepath)
                messagebox.showinfo(
                    "Exportação Concluída",
                    f"Relatório de estrutura exportado com sucesso para:\n{filepath}"
                )
                self._set_status(f"Relatório exportado: {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Erro ao Exportar Relatório", f"Falha ao salvar relatório JSON:\n{e}")

    # ------------------------------------------------------------------
    # Atualização por ciclo (chamada pelo _ui_update_loop de MainWindow)
    # ------------------------------------------------------------------
    def refresh_tables(self) -> None:
        """Atualiza tabelas de padrões conhecidos e matriz de transições."""
        patterns = self._analyzer.pattern_memory.get_known_patterns()
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

        transitions = self._analyzer.pattern_memory.get_transitions()
        self.tree_transitions.delete(*self.tree_transitions.get_children())
        for tr in transitions:
            prob_txt = f"{tr.probability * 100:.0f}%"
            self.tree_transitions.insert(
                "", "end",
                values=(tr.from_section, tr.to_section, f"{tr.from_pattern_id} → {tr.to_pattern_id}", prob_txt, tr.count)
            )

    def reset_tables(self) -> None:
        """Limpa as tabelas ao carregar um novo áudio."""
        for item in self.tree_patterns.get_children():
            self.tree_patterns.delete(item)
        for item in self.tree_transitions.get_children():
            self.tree_transitions.delete(item)

    def update(self, ctx) -> None:
        """Atualiza os cartões de seção/previsão a partir do MusicalContext atual."""
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

        struct = self._analyzer.structure_analyzer.structure
        if struct.abstract_sequence:
            self.lbl_struct_form.config(text=" - ".join(struct.abstract_sequence), fg="#c77dff")
            self.lbl_struct_summary.config(
                text=f"{len(struct.sections)} seções  |  {len(struct.patterns)} padrões",
                fg="#ffffff"
            )
        else:
            self.lbl_struct_form.config(text="--", fg="#5a6677")
            self.lbl_struct_summary.config(text="0 seções  |  0 padrões", fg="#5a6677")

    def reset_display(self) -> None:
        """Reseta os cartões de estado (usado quando a reprodução para)."""
        self.lbl_struct_cur_section.config(text="--", fg="#5a6677")
        self.lbl_struct_progress.config(text="Progresso: --%  |  Comp. --", fg="#5a6677")
        self.lbl_pred_next_section.config(text="--", fg="#5a6677")
        self.lbl_pred_timing.config(text="Em -- compassos", fg="#5a6677")
        self.lbl_pred_lookahead.config(text="--", fg="#5a6677")
        self.lbl_pred_reason.config(text="Aguardando harmonia", fg="#5a6677")
        self.lbl_struct_form.config(text="--", fg="#5a6677")
        self.lbl_struct_summary.config(text="0 seções  |  0 padrões", fg="#5a6677")
