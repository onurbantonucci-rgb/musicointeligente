"""Modelo de Dados Song e Configurações de Performance (v0.4).

Define uma música como uma entidade independente contendo áudio, cifra, letra,
estrutura, memória de padrões e configurações, sem reter estado volátil de execução.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid


@dataclass
class PerformanceSettings:
    """Configurações de performance e acompanhamento específicas da música."""
    bpm_override: Optional[float] = None
    key_override: Optional[str] = None
    # Fontes independentes: andamento regula a grade rítmica; tom regula a
    # interpretação tonal. Valores persistidos por música.
    bpm_source: str = "AUDIO"       # AUDIO | CHART
    key_source: str = "CHART"       # CHART | AUDIO
    meter_override: Optional[str] = None
    transpose_semitones: int = 0
    master_volume: float = 1.0
    mute: bool = False
    solo: bool = False
    bass_enabled: bool = True
    bass_style: str = "Root-Fifth"       # "Root-Fifth", "Walking", "Arpeggio", "Rock-Eighths"
    drums_enabled: bool = True
    drums_style: str = "Standard-Rock"
    keyboard_enabled: bool = True
    keyboard_style: str = "Comping"
    guitar_enabled: bool = True
    guitar_style: str = "Acoustic-Strum"
    humanization: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bpm_override": self.bpm_override,
            "key_override": self.key_override,
            "bpm_source": self.bpm_source,
            "key_source": self.key_source,
            "meter_override": self.meter_override,
            "transpose_semitones": self.transpose_semitones,
            "master_volume": self.master_volume,
            "mute": self.mute,
            "solo": self.solo,
            "bass_enabled": self.bass_enabled,
            "bass_style": self.bass_style,
            "drums_enabled": self.drums_enabled,
            "drums_style": self.drums_style,
            "keyboard_enabled": self.keyboard_enabled,
            "keyboard_style": self.keyboard_style,
            "guitar_enabled": self.guitar_enabled,
            "guitar_style": self.guitar_style,
            "humanization": self.humanization,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PerformanceSettings":
        if not data:
            return cls()
        return cls(
            bpm_override=data.get("bpm_override"),
            key_override=data.get("key_override"),
            bpm_source=(str(data.get("bpm_source", "AUDIO")).upper()
                        if str(data.get("bpm_source", "AUDIO")).upper() in ("AUDIO", "CHART") else "AUDIO"),
            key_source=(str(data.get("key_source", "CHART")).upper()
                        if str(data.get("key_source", "CHART")).upper() in ("AUDIO", "CHART") else "CHART"),
            meter_override=data.get("meter_override"),
            transpose_semitones=int(data.get("transpose_semitones", 0)),
            master_volume=float(data.get("master_volume", 1.0)),
            mute=bool(data.get("mute", False)),
            solo=bool(data.get("solo", False)),
            bass_enabled=bool(data.get("bass_enabled", True)),
            bass_style=str(data.get("bass_style", "Root-Fifth")),
            drums_enabled=bool(data.get("drums_enabled", True)),
            drums_style=str(data.get("drums_style", "Standard-Rock")),
            keyboard_enabled=bool(data.get("keyboard_enabled", True)),
            keyboard_style=str(data.get("keyboard_style", "Comping")),
            guitar_enabled=bool(data.get("guitar_enabled", True)),
            guitar_style=str(data.get("guitar_style", "Acoustic-Strum")),
            humanization=float(data.get("humanization", 0.05)),
        )


@dataclass
class Song:
    """Representação conceitual persistente de uma canção na banda virtual."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = "Sem Título"
    artist: str = "Artista Desconhecido"
    audio_path: str = ""
    duration: float = 0.0
    
    # Definições musicais nominais
    key: str = "C Major"
    bpm: float = 120.0
    meter: str = "4/4"
    time_signature: str = "4/4"
    
    # Cifra estruturada e Letra
    chart_text: str = ""                # Texto cru original da cifra
    chart_data: Optional[Dict[str, Any]] = None  # Estrutura parseada ChordChart
    lyrics_text: str = ""               # Letra completa da música
    
    # Modelos e Históricos analíticos consolidados
    structure_data: Optional[Dict[str, Any]] = None
    pattern_memory_data: Optional[Dict[str, Any]] = None
    prediction_state_data: Optional[Dict[str, Any]] = None
    chord_history_summary: List[Dict[str, Any]] = field(default_factory=list)
    key_history_summary: List[Dict[str, Any]] = field(default_factory=list)
    
    # Configurações de performance e Metadados
    performance_settings: PerformanceSettings = field(default_factory=PerformanceSettings)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if not self.time_signature:
            self.time_signature = self.meter
        elif not self.meter:
            self.meter = self.time_signature

    def update_timestamp(self) -> None:
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a música em dicionário para persistência local."""
        return {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "audio_path": self.audio_path,
            "duration": round(float(self.duration), 3),
            "key": self.key,
            "bpm": round(float(self.bpm), 2),
            "meter": self.meter,
            "time_signature": self.time_signature,
            "chart_text": self.chart_text,
            "chart_data": self.chart_data,
            "lyrics_text": self.lyrics_text,
            "structure_data": self.structure_data,
            "pattern_memory_data": self.pattern_memory_data,
            "prediction_state_data": self.prediction_state_data,
            "chord_history_summary": self.chord_history_summary,
            "key_history_summary": self.key_history_summary,
            "performance_settings": self.performance_settings.to_dict(),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Song":
        """Reconstitui uma instância de Song a partir de dicionário deserializado."""
        perf_dict = data.get("performance_settings", {})
        perf_obj = PerformanceSettings.from_dict(perf_dict)
        
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            title=data.get("title", "Sem Título"),
            artist=data.get("artist", "Artista Desconhecido"),
            audio_path=data.get("audio_path", ""),
            duration=float(data.get("duration", 0.0)),
            key=data.get("key", "C Major"),
            bpm=float(data.get("bpm", 120.0)),
            meter=data.get("meter", data.get("time_signature", "4/4")),
            time_signature=data.get("time_signature", data.get("meter", "4/4")),
            chart_text=data.get("chart_text", ""),
            chart_data=data.get("chart_data"),
            lyrics_text=data.get("lyrics_text", ""),
            structure_data=data.get("structure_data"),
            pattern_memory_data=data.get("pattern_memory_data"),
            prediction_state_data=data.get("prediction_state_data"),
            chord_history_summary=list(data.get("chord_history_summary", [])),
            key_history_summary=list(data.get("key_history_summary", [])),
            performance_settings=perf_obj,
            metadata=dict(data.get("metadata", {})),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )
