"""Fonte de áudio em tempo real (Live Audio Input).

Captura áudio de microfone, entrada de linha ou interface USB
e entrega blocos de áudio através da mesma interface AudioSource.
"""

import logging
import threading
from typing import Optional, List
import numpy as np
import sounddevice as sd

from app.audio.audio_source import AudioSource

logger = logging.getLogger(__name__)


class LiveAudioSource(AudioSource):
    """Fonte de captura de áudio ao vivo (Microfone / Linha / Interface de Áudio).
    
    Implementa a mesma interface AudioSource que FileAudioSource, permitindo
    que o AudioAnalyzer processe o sinal sem saber se a origem é arquivo ou ao vivo.
    """

    def __init__(self, sample_rate: int = 44100, channels: int = 1, device_index: Optional[int] = None):
        self._sample_rate = sample_rate
        self._channels = channels
        self._device_index = device_index
        self._name = "Live Audio Input"
        self._lock = threading.Lock()
        self._active = False

        # Buffer circular para armazenar amostras em tempo real
        self._buffer_size = sample_rate * 5  # 5 segundos de histórico
        self._ring_buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._write_pos = 0
        self._read_pos = 0
        self._total_recorded_frames = 0

        self._stream: Optional[sd.InputStream] = None

    @classmethod
    def list_available_devices(cls) -> List[dict]:
        """Lista todos os dispositivos de entrada de áudio disponíveis no Windows."""
        devices = []
        try:
            device_list = sd.query_devices()
            for idx, dev in enumerate(device_list):
                if dev.get("max_input_channels", 0) > 0:
                    devices.append({
                        "index": idx,
                        "name": dev.get("name", f"Device {idx}"),
                        "channels": dev.get("max_input_channels"),
                        "default_samplerate": dev.get("default_samplerate")
                    })
        except Exception as e:
            logger.error("Erro ao enumerar dispositivos: %s", e)
        return devices

    def start(self) -> None:
        """Inicia a captura de áudio da entrada de hardware."""
        with self._lock:
            if self._active:
                return
            try:
                self._stream = sd.InputStream(
                    device=self._device_index,
                    samplerate=self._sample_rate,
                    channels=self._channels,
                    dtype="float32",
                    callback=self._input_callback,
                    blocksize=1024,
                )
                self._stream.start()
                self._active = True
                logger.info("Captura ao vivo iniciada em %dHz", self._sample_rate)
            except Exception as e:
                self._active = False
                logger.error("Erro ao iniciar captura: %s", e)
                raise e

    def _input_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback do driver de gravação do Windows / PortAudio."""
        if not self._active:
            return

        # Converter para mono se for multicanal
        if indata.ndim > 1 and indata.shape[1] > 1:
            mono = np.mean(indata, axis=1).astype(np.float32)
        else:
            mono = indata.flatten().astype(np.float32)

        with self._lock:
            # Escrita no buffer circular
            for sample in mono:
                self._ring_buffer[self._write_pos] = sample
                self._write_pos = (self._write_pos + 1) % self._buffer_size
            self._total_recorded_frames += frames

    def read_chunk(self, chunk_size: int) -> np.ndarray:
        """Lê os últimos `chunk_size` samples gravados para o pipeline de análise."""
        with self._lock:
            if not self._active or chunk_size <= 0:
                return np.zeros(chunk_size, dtype=np.float32)

            # Obter os últimos chunk_size elementos antes de _write_pos
            indices = (np.arange(self._write_pos - chunk_size, self._write_pos)) % self._buffer_size
            return self._ring_buffer[indices].copy()

    def read_playback_chunk(self, chunk_size: int) -> np.ndarray:
        """Para áudio ao vivo, retorna o chunk mono replicado nos canais para monitoramento."""
        mono = self.read_chunk(chunk_size)
        if self._channels == 1:
            return mono[:, np.newaxis]
        return np.column_stack([mono] * self._channels)

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def get_duration(self) -> float:
        # Fluxos ao vivo não possuem duração fixa pré-determinada
        return float("inf")

    def get_position(self) -> float:
        with self._lock:
            return self._total_recorded_frames / float(self._sample_rate) if self._sample_rate > 0 else 0.0

    def seek(self, position_seconds: float) -> None:
        # Áudio ao vivo não aceita seek arbitrário
        pass

    def is_active(self) -> bool:
        return self._active

    def close(self) -> None:
        with self._lock:
            self._active = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def name(self) -> str:
        return self._name
