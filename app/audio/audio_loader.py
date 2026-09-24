"""Carregamento e gerenciamento de arquivos de áudio (WAV e MP3)."""

import os
import threading
from typing import Tuple, Optional
import numpy as np
import soundfile as sf
import librosa

from app.audio.audio_source import AudioSource


class FileAudioSource(AudioSource):
    """Fonte de áudio baseada em arquivo local (WAV, MP3, etc.).
    
    Carrega o áudio para a memória em ponto flutuante, oferecendo leitura
    em chunks síncrona, busca (seek), conversão mono/estéreo e geração
    de envelope para renderização rápida de waveform.
    """

    def __init__(self, file_path: str):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {file_path}")

        self._file_path = os.path.abspath(file_path)
        self._name = os.path.basename(self._file_path)
        self._lock = threading.Lock()
        self._active = False

        # Dados decodificados
        self._sample_rate: int = 0
        self._duration: float = 0.0
        self._total_frames: int = 0
        self._channels: int = 1
        self._analysis_channel: Optional[int] = None
        self._position_frame: int = 0

        # Áudio em memória
        # _playback_data: shape (frames, channels) float32
        # _mono_data: shape (frames,) float32
        self._playback_data: np.ndarray = np.array([], dtype=np.float32)
        self._mono_data: np.ndarray = np.array([], dtype=np.float32)

        self._load_file()

    def _load_file(self) -> None:
        """Decodifica o arquivo de áudio para float32 com fallback se necessário."""
        try:
            # Leitura primária de alta performance via soundfile (suporta WAV e MP3 nativamente)
            data, sr = sf.read(self._file_path, dtype="float32", always_2d=True)
        except Exception as primary_err:
            try:
                # Fallback via librosa (que usa audioread/av)
                data_librosa, sr = librosa.load(self._file_path, sr=None, mono=False)
                if data_librosa.ndim == 1:
                    data = data_librosa[:, np.newaxis]
                else:
                    data = data_librosa.T  # Transpor para formato (frames, channels)
                data = data.astype(np.float32)
            except Exception as secondary_err:
                raise RuntimeError(
                    f"Falha ao decodificar áudio {self._name}: {primary_err} / Fallback: {secondary_err}"
                )

        self._sample_rate = int(sr)
        self._playback_data = data
        self._total_frames = data.shape[0]
        self._channels = data.shape[1]
        self._duration = self._total_frames / float(self._sample_rate) if self._sample_rate > 0 else 0.0

        # Canal mono normalizado para análise harmônica/DSP
        if self._channels == 1:
            self._mono_data = self._playback_data[:, 0].copy()
        else:
            self._mono_data = np.mean(self._playback_data, axis=1).astype(np.float32)

        # Prevenção de clipping e normalização segura se ultrapassar 1.0
        max_val = np.max(np.abs(self._mono_data)) if self._mono_data.size > 0 else 0.0
        if max_val > 1.0:
            self._mono_data /= max_val
            self._playback_data /= max_val

        self._position_frame = 0
        self._active = True

    def read_chunk(self, chunk_size: int) -> np.ndarray:
        """Lê os próximos `chunk_size` frames do áudio mono. Se o arquivo acabar, preenche com zeros."""
        with self._lock:
            if not self._active or self._total_frames == 0:
                return np.zeros(chunk_size, dtype=np.float32)

            start = self._position_frame
            end = min(start + chunk_size, self._total_frames)
            count = end - start

            if count <= 0:
                return np.zeros(chunk_size, dtype=np.float32)

            chunk = self._mono_data[start:end]
            self._position_frame = end

            if count < chunk_size:
                padded = np.zeros(chunk_size, dtype=np.float32)
                padded[:count] = chunk
                return padded

            return chunk.copy()

    def read_playback_chunk(self, chunk_size: int) -> np.ndarray:
        """Lê os próximos `chunk_size` frames em formato multicanal para reprodução."""
        with self._lock:
            if not self._active or self._total_frames == 0:
                return np.zeros((chunk_size, self._channels), dtype=np.float32)

            start = self._position_frame
            end = min(start + chunk_size, self._total_frames)
            count = end - start

            if count <= 0:
                return np.zeros((chunk_size, self._channels), dtype=np.float32)

            chunk = self._playback_data[start:end, :]
            self._position_frame = end

            if count < chunk_size:
                padded = np.zeros((chunk_size, self._channels), dtype=np.float32)
                padded[:count, :] = chunk
                return padded

            return chunk.copy()

    def get_chunk_at(self, start_frame: int, chunk_size: int) -> np.ndarray:
        """Lê um bloco mono em uma posição arbitrária sem alterar o cursor de reprodução."""
        with self._lock:
            if not self._active or self._total_frames == 0:
                return np.zeros(chunk_size, dtype=np.float32)

            start = max(0, start_frame)
            end = min(start + chunk_size, self._total_frames)
            count = end - start

            if count <= 0:
                return np.zeros(chunk_size, dtype=np.float32)

            chunk = self._mono_data[start:end]
            if count < chunk_size:
                padded = np.zeros(chunk_size, dtype=np.float32)
                padded[:count] = chunk
                return padded
            return chunk.copy()

    def get_analysis_chunk_at(self, start_frame: int, chunk_size: int) -> np.ndarray:
        """Lê o canal de análise escolhido sem alterar o cursor de reprodução."""
        with self._lock:
            if not self._active or self._total_frames == 0:
                return np.zeros(chunk_size, dtype=np.float32)
            start = max(0, start_frame)
            end = min(start + chunk_size, self._total_frames)
            if self._analysis_channel is None or self._channels == 1:
                chunk = self._mono_data[start:end]
            else:
                chunk = self._playback_data[start:end, self._analysis_channel]
            if len(chunk) < chunk_size:
                padded = np.zeros(chunk_size, dtype=np.float32)
                padded[:len(chunk)] = chunk
                return padded
            return chunk.copy()

    def set_analysis_channel(self, channel: Optional[int]) -> None:
        """Seleciona um canal fixo para o ouvido; ``None`` usa a mistura mono."""
        with self._lock:
            if channel is None or self._channels == 1:
                self._analysis_channel = None
                return
            value = int(channel)
            if value < 0 or value >= self._channels:
                raise ValueError(f"Canal de análise inválido: {value}")
            self._analysis_channel = value

    @property
    def analysis_channel(self) -> Optional[int]:
        return self._analysis_channel

    def seek(self, position_seconds: float) -> None:
        """Move o cursor de leitura para o tempo especificado."""
        with self._lock:
            if self._sample_rate <= 0 or self._total_frames == 0:
                return
            target_frame = int(position_seconds * self._sample_rate)
            self._position_frame = max(0, min(target_frame, self._total_frames))

    def get_position(self) -> float:
        """Retorna o tempo atual do cursor em segundos."""
        with self._lock:
            if self._sample_rate <= 0:
                return 0.0
            return self._position_frame / float(self._sample_rate)

    def get_position_frame(self) -> int:
        """Retorna o frame atual do cursor."""
        with self._lock:
            return self._position_frame

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def get_duration(self) -> float:
        return self._duration

    def is_active(self) -> bool:
        return self._active

    def close(self) -> None:
        with self._lock:
            self._active = False

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def name(self) -> str:
        return self._name

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def mono_data(self) -> np.ndarray:
        """Acesso somente-leitura ao sinal mono completo."""
        return self._mono_data

    @property
    def multichannel_data(self) -> np.ndarray:
        """Acesso somente-leitura aos canais originais para calibração."""
        return self._playback_data

    def get_waveform_envelope(self, num_points: int = 800) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula min/max decimados para renderização gráfica instantânea de waveform.
        
        Retorna duas matrizes 1D: (mins, maxs) com comprimento num_points.
        """
        if self._mono_data.size == 0 or num_points <= 0:
            return np.zeros(num_points, dtype=np.float32), np.zeros(num_points, dtype=np.float32)

        total_samples = len(self._mono_data)
        if total_samples < num_points:
            mins = self._mono_data
            maxs = self._mono_data
            return mins, maxs

        # Dividir em fatias e extrair picos
        step = total_samples / float(num_points)
        mins = np.zeros(num_points, dtype=np.float32)
        maxs = np.zeros(num_points, dtype=np.float32)

        for i in range(num_points):
            idx_start = int(i * step)
            idx_end = int((i + 1) * step)
            if idx_start < idx_end and idx_start < total_samples:
                slice_data = self._mono_data[idx_start:idx_end]
                mins[i] = np.min(slice_data)
                maxs[i] = np.max(slice_data)

        return mins, maxs
