"""Virtual Band AI — Ponto de Entrada Principal (v0.1-A).

Executa a interface gráfica e o motor de áudio.
"""

import tkinter as tk
from app.utils.logging_config import configure_logging
from app.ui.main_window import MainWindow


def main():
    """Inicializa a aplicação Virtual Band AI."""
    configure_logging()
    root = tk.Tk()
    _app = MainWindow(root)  # mantido vivo pelo laço de eventos do Tk
    root.mainloop()


if __name__ == "__main__":
    main()
