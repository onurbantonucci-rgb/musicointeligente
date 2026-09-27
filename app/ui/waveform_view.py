"""Widget Canvas de alta performance para renderização de Waveform e Seek interativo."""

import tkinter as tk
from typing import Optional, Callable
import numpy as np


class WaveformView(tk.Canvas):
    """Componente gráfico Tkinter que desenha a forma de onda do áudio e o cursor de reprodução.
    
    Permite clique direto para navegar (seek) pelo áudio.
    """

    def __init__(self, parent, height: int = 120, bg_color: str = "#151922", wave_color: str = "#00bcd4",
                 playhead_color: str = "#ff5252", grid_color: str = "#242c3d", **kwargs):
        super().__init__(parent, height=height, bg=bg_color, highlightthickness=0, **kwargs)
        self._bg_color = bg_color
        self._wave_color = wave_color
        self._playhead_color = playhead_color
        self._grid_color = grid_color

        self._mins: np.ndarray = np.array([], dtype=np.float32)
        self._maxs: np.ndarray = np.array([], dtype=np.float32)
        self._duration: float = 0.0
        self._playhead_time: float = 0.0

        # Callback chamado quando o usuário clica na waveform: callback(target_time_seconds)
        self.on_seek_requested: Optional[Callable[[float], None]] = None

        # Eventos de mouse e redimensionamento
        self.bind("<Configure>", self._on_resize)
        self.bind("<Button-1>", self._on_mouse_click)
        self.bind("<B1-Motion>", self._on_mouse_drag)

    def set_waveform_data(self, mins: np.ndarray, maxs: np.ndarray, duration: float) -> None:
        """Carrega os dados decimados da forma de onda e a duração em segundos."""
        self._mins = mins
        self._maxs = maxs
        self._duration = max(0.001, duration)
        self._playhead_time = 0.0
        self.redraw()

    def set_playhead_position(self, position_seconds: float) -> None:
        """Atualiza a posição do cursor de reprodução (playhead) em segundos."""
        self._playhead_time = max(0.0, min(position_seconds, self._duration))
        self._draw_playhead()

    def clear(self) -> None:
        """Limpa o visualizador."""
        self._mins = np.array([], dtype=np.float32)
        self._maxs = np.array([], dtype=np.float32)
        self._duration = 0.0
        self._playhead_time = 0.0
        self.delete("all")

    def _on_resize(self, event) -> None:
        self.redraw()

    def _on_mouse_click(self, event) -> None:
        self._handle_mouse_seek(event.x)

    def _on_mouse_drag(self, event) -> None:
        self._handle_mouse_seek(event.x)

    def _handle_mouse_seek(self, mouse_x: int) -> None:
        width = self.winfo_width()
        if width <= 0 or self._duration <= 0:
            return

        ratio = max(0.0, min(1.0, mouse_x / float(width)))
        target_seconds = ratio * self._duration
        self.set_playhead_position(target_seconds)

        if self.on_seek_requested:
            self.on_seek_requested(target_seconds)

    def redraw(self) -> None:
        """Redesenha completamente o fundo, a grade, a forma de onda e o playhead."""
        self.delete("all")
        width = self.winfo_width()
        height = self.winfo_height()

        if width <= 1 or height <= 1:
            return

        mid_y = height / 2.0

        # Grade horizontal central (linha de zero)
        self.create_line(0, mid_y, width, mid_y, fill=self._grid_color, width=1, dash=(3, 3))

        if len(self._mins) == 0:
            # Mensagem de estado vazio
            self.create_text(
                width / 2, mid_y,
                text="Nenhum arquivo de áudio carregado. Clique em 'ABRIR ÁUDIO' para começar.",
                fill="#5c6b84", font=("Segoe UI", 9, "italic")
            )
            return

        # Desenhar forma de onda
        num_points = len(self._mins)
        step_x = width / float(num_points)

        for i in range(num_points):
            x = i * step_x
            min_val = self._mins[i]
            max_val = self._maxs[i]

            # Mapear amplitude [-1.0, 1.0] para coordenadas de tela [height, 0]
            # Deixar 4px de margem superior e inferior
            usable_height = (height - 8) / 2.0
            y1 = mid_y - (max_val * usable_height)
            y2 = mid_y - (min_val * usable_height)

            # Evitar linhas invisíveis quando min == max
            if abs(y2 - y1) < 1.0:
                y1 = mid_y - 1
                y2 = mid_y + 1

            self.create_line(x, y1, x, y2, fill=self._wave_color, width=max(1, int(step_x) + 1))

        self._draw_playhead()

    def _draw_playhead(self) -> None:
        """Desenha ou move a linha indicadora do playhead sobre o canvas."""
        self.delete("playhead")
        width = self.winfo_width()
        height = self.winfo_height()

        if width <= 0 or self._duration <= 0:
            return

        ratio = self._playhead_time / self._duration
        x = ratio * width

        # Linha vertical do playhead
        self.create_line(x, 0, x, height, fill=self._playhead_color, width=2, tags="playhead")
        # Pequeno marcador triangular no topo
        self.create_polygon(x - 5, 0, x + 5, 0, x, 8, fill=self._playhead_color, tags="playhead")
