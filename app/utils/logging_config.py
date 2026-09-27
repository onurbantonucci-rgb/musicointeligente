"""Configuração central de logging do Virtual Band AI.

Substitui os `print()` de depuração espalhados pelo código por `logging`, permitindo
silenciar, redirecionar para arquivo ou ajustar o nível de verbosidade sem tocar em
cada módulo. Cada módulo usa ``logging.getLogger(__name__)`` e herda esta configuração.
"""

import logging

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configura o logging raiz da aplicação. Idempotente (chamadas extras são ignoradas)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True
