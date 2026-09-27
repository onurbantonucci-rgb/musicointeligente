"""Registro de Músicos Virtuais e Abstrações de Acompanhamento (v0.4).

Gerencia a Banda Virtual conectando o BassPlayer funcional e expondo interfaces
e contratos para os futuros integrantes (Bateria, Teclado e Guitarra).
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from app.instruments.base import VirtualInstrument
from app.instruments.bass_player import BassPlayer
from app.music.musical_context import MusicalContext

logger = logging.getLogger(__name__)


class AdaptiveMusicalClock(ABC):
    """Interface conceitual para futuro acompanhamento adaptativo de andamento ao vivo.
    
    Permitirá que o músico dite o tempo (acelerando, desacelerando ou sustentando)
    e o relógio musical adapte-se continuamente sem impor um BPM rígido.
    """

    @abstractmethod
    def on_musician_pulse(self, detected_onset_timestamp: float, expected_beat: int) -> None:
        """Notifica o relógio de um pulso executado pelo músico líder."""
        pass

    @abstractmethod
    def get_current_tempo(self) -> float:
        """Retorna o andamento dinamicamente ajustado em BPM."""
        pass


class VirtualPlayerRegistry:
    """Repositório e orquestrador de todos os músicos virtuais da banda."""

    def __init__(self, sample_rate: int = 44100, bass_player: Optional[BassPlayer] = None):
        self._sample_rate = sample_rate
        self._players: Dict[str, VirtualInstrument] = {}
        self._player_statuses: Dict[str, str] = {
            "bass": "READY",
            "drums": "STANDBY",
            "keyboard": "STANDBY",
            "guitar": "STANDBY",
        }

        # Inicializa e cadastra o BassPlayer existente
        self._bass_player = bass_player if bass_player is not None else BassPlayer(sample_rate=sample_rate)
        self.register_player("bass", self._bass_player)

    def register_player(self, slot: str, player: VirtualInstrument) -> None:
        """Cadastra um instrumento virtual em um slot da banda."""
        self._players[slot.lower()] = player
        self._player_statuses[slot.lower()] = "READY"

    def unregister_player(self, slot: str) -> None:
        s = slot.lower()
        if s in self._players:
            del self._players[s]
            self._player_statuses[s] = "STANDBY"

    def get_player(self, slot: str) -> Optional[VirtualInstrument]:
        return self._players.get(slot.lower())

    @property
    def bass_player(self) -> Optional[BassPlayer]:
        p = self.get_player("bass")
        return p if isinstance(p, BassPlayer) else None

    @property
    def registered_slots(self) -> List[str]:
        return list(self._players.keys())

    @property
    def all_statuses(self) -> Dict[str, str]:
        """Status operacional de cada posto da banda para exibição na UI."""
        return dict(self._player_statuses)

    def dispatch_context(self, context: MusicalContext) -> Dict[str, Any]:
        """Dispara o MusicalContext para todos os músicos ativos e coleta decisões."""
        events = {}
        for slot, player in self._players.items():
            try:
                ev = player.on_musical_context(context)
                if ev is not None:
                    events[slot] = ev
            except Exception as e:
                logger.error("Erro ao despachar contexto para %s: %s", slot, e)
        return events

    def reset_all(self) -> None:
        """Reinicia o estado interno e memória de todos os músicos da banda."""
        for player in self._players.values():
            player.reset()
