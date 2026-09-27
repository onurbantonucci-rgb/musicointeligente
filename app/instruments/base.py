"""Contratos e interfaces base para instrumentos virtuais autônomos."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from app.music.musical_context import MusicalContext

logger = logging.getLogger(__name__)


class VirtualInstrument(ABC):
    """Interface fundamental para qualquer músico virtual (baixo, bateria, teclado, guitarra).
    
    Cada instrumento consome unicamente o MusicalContext e toma suas próprias
    decisões musicais de arranjo e performance.
    """

    @abstractmethod
    def on_musical_context(self, context: MusicalContext) -> Optional[object]:
        """Recebe o estado musical atualizado e decide se deve gerar uma nota/evento."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reinicia o estado interno e memória musical do instrumento."""
        pass

    @property
    @abstractmethod
    def instrument_name(self) -> str:
        """Nome do instrumento (ex: 'Virtual Bassist', 'Virtual Drummer')."""
        pass


class VirtualBand:
    """Orquestrador da Banda Virtual que despacha o MusicalContext para todos os músicos cadastrados."""

    def __init__(self):
        self._members: List[VirtualInstrument] = []

    def add_member(self, instrument: VirtualInstrument) -> None:
        self._members.append(instrument)

    def dispatch_context(self, context: MusicalContext) -> None:
        """Propaga o contexto musical instantâneo para todos os membros da banda."""
        for member in self._members:
            try:
                member.on_musical_context(context)
            except Exception as e:
                logger.error("Erro no integrante %s: %s", member.instrument_name, e)

    def reset_all(self) -> None:
        for member in self._members:
            member.reset()
