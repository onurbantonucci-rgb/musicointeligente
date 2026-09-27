"""Widget Canvas para visualização em tempo real das 12 classes cromáticas de notas."""

import tkinter as tk
from typing import List

from app.music.theory import PITCH_CLASSES


class ChromaView(tk.Canvas):
    """Componente gráfico que renderiza as 12 barras cromáticas (C, C#, D... B).
    
    Permite visualizar instantaneamente a energia harmônica presente em cada nota musical.
    """

    def __init__(self, parent, height: int = 70, bg_color: str = "#141820",
                 active_color: str = "#00e676", inactive_color: str = "#242c3d", **kwargs):
        super().__init__(parent, height=height, bg=bg_color, highlightthickness=0, **kwargs)
        self._bg_color = bg_color
        self._active_color = active_color
        self._inactive_color = inactive_color
        self._energies = [0.0] * 12

        self.bind("<Configure>", lambda e: self.redraw())

    def update_chroma(self, energies: List[float]) -> None:
        """Atualiza os valores de energia das 12 notas e redesenha."""
        if len(energies) >= 12:
            self._energies = energies[:12]
            self.redraw()

    def clear(self) -> None:
        self._energies = [0.0] * 12
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()

        if w <= 1 or h <= 1:
            return

        bar_spacing = 4
        total_bars = 12
        bar_width = max(8, (w - (bar_spacing * (total_bars + 1))) / total_bars)

        label_area_h = 16
        graph_h = max(10, h - label_area_h - 4)

        for i, note in enumerate(PITCH_CLASSES):
            x1 = bar_spacing + i * (bar_width + bar_spacing)
            x2 = x1 + bar_width

            val = max(0.0, min(1.0, self._energies[i]))
            bar_height = val * graph_h

            y2 = graph_h + 2
            y1 = y2 - bar_height

            # Cor dinâmica: ciano para baixo, verde para notas ativas (> 0.45)
            if val > 0.45:
                color = "#00e676" if val > 0.75 else "#00d2ff"
                label_color = "#ffffff"
                label_font = ("Segoe UI", 7, "bold")
            else:
                color = self._inactive_color
                label_color = "#5a6677"
                label_font = ("Segoe UI", 7)

            # Barra de energia
            if bar_height > 1:
                self.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
            else:
                # Marcador mínimo
                self.create_line(x1, y2, x2, y2, fill=self._inactive_color, width=1)

            # Nome da nota abaixo da barra
            self.create_text((x1 + x2) / 2, h - 8, text=note, fill=label_color, font=label_font)
