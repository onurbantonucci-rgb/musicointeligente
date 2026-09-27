"""Reprodutor de áudio em tempo real com controle de transporte e sincronização."""

import logging
from enum import Enum
import threading
import time
from typing import Callable, Optional
import numpy as np
import sounddevice as sd

from app.audio.audio_source import AudioSource

logger = logging.getLogger(__name__)


class PlaybackState(Enum):
    STOPPED = "STOPPED"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"


class AudioPlayer:
    """Reprodutor de áudio thread-safe utilizando sounddevice.
    
    Permite Play, Pause, Stop, Seek e acoplamento com o analisador de DSP.
    """
    OUTPUT_BLOCK_SIZE = 2048

    def __init__(self):
        self._source: Optional[AudioSource] = None
        self._stream: Optional[sd.OutputStream] = None
        self._state: PlaybackState = PlaybackState.STOPPED
        self._lock = threading.RLock()

        # Callbacks para interface e pipeline
        self.on_position_change: Optional[Callable[[float], None]] = None
        self.on_state_change: Optional[Callable[[PlaybackState], None]] = None
        self.on_playback_finished: Optional[Callable[[], None]] = None
        self.on_audio_chunk: Optional[Callable[[np.ndarray, float], None]] = None
        self.on_mix_audio: Optional[Callable[[int, int, float], Optional[np.ndarray]]] = None

        # Thread de monitoramento de posição (desacoplada do stream de hardware)
        self._monitor_running = False
        self._monitor_thread: Optional[threading.Thread] = None

    @property
    def state(self) -> PlaybackState:
        return self._state

    @property
    def source(self) -> Optional[AudioSource]:
        return self._source

    def load_source(self, source: AudioSource) -> None:
        """Carrega uma nova fonte de áudio e reinicia o estado do player."""
        with self._lock:
            self.unload_source()
            self._source = source
            self._init_stream()

    def unload_source(self) -> None:
        """Interrompe a reprodução e libera a fonte anterior."""
        with self._lock:
            self.stop()
            self._close_stream()
            if self._source is not None:
                self._source.close()
                self._source = None

    def _init_stream(self) -> None:
        """Inicializa o fluxo de saída de áudio com a taxa de amostragem e canais da fonte."""
        if self._source is None:
            return

        sr = self._source.get_sample_rate()
        channels = self._source.channels

        if sr <= 0:
            return

        try:
            self._stream = sd.OutputStream(
                samplerate=sr,
                channels=channels,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=self.OUTPUT_BLOCK_SIZE,
            )
            self._stream.start()
        except Exception as e:
            logger.error("Erro ao iniciar stream de saída de áudio: %s", e)
            self._stream = None

    @property
    def estimated_output_latency_ms(self) -> float:
        """Estimativa do buffer configurado; o atraso do driver permanece desconhecido."""
        if self._source is None:
            return 0.0
        sample_rate = self._source.get_sample_rate()
        return (self.OUTPUT_BLOCK_SIZE / float(sample_rate)) * 1000.0 if sample_rate > 0 else 0.0

    def _close_stream(self) -> None:
        """Fecha o stream do sounddevice se estiver ativo."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback de hardware de áudio do PortAudio (deve ser ágil e não-bloqueante)."""
        if self._state != PlaybackState.PLAYING or self._source is None:
            outdata.fill(0)
            return

        current_pos = self._source.get_position()
        duration = self._source.get_duration()

        # Fim do arquivo
        if duration > 0 and current_pos >= duration:
            outdata.fill(0)
            # Notificação em thread separada para não bloquear o callback de hardware
            threading.Thread(target=self._handle_eof, daemon=True).start()
            return

        # Leitura do chunk para saída dos alto-falantes
        chunk = self._source.read_playback_chunk(frames)
        outdata[:] = chunk

        # Mixagem em tempo real de instrumentos virtuais (ex: Baixista Virtual)
        if self.on_mix_audio is not None:
            sr = self._source.get_sample_rate()
            try:
                instrument_chunk = self.on_mix_audio(frames, sr, current_pos)
                if instrument_chunk is not None:
                    instrument_chunk = np.asarray(instrument_chunk)
                    if instrument_chunk.ndim == 1:
                        instrument_chunk = instrument_chunk[:, np.newaxis]
                    if instrument_chunk.ndim == 2 and instrument_chunk.shape[0] == frames:
                        if outdata.shape[1] == 1:
                            instrument_chunk = np.mean(instrument_chunk, axis=1, keepdims=True)
                        elif instrument_chunk.shape[1] == 1:
                            instrument_chunk = np.repeat(instrument_chunk, outdata.shape[1], axis=1)
                        if instrument_chunk.shape == outdata.shape:
                            outdata[:] = np.clip(outdata + instrument_chunk, -1.0, 1.0)
            except Exception:
                pass

        # Se houver consumidor de análise cadastrado, repassar uma fatia mono

        if self.on_audio_chunk is not None:
            mono_chunk = np.mean(chunk, axis=1).astype(np.float32)
            try:
                self.on_audio_chunk(mono_chunk, current_pos)
            except Exception:
                pass

    def _handle_eof(self) -> None:
        """Trata o término da faixa de áudio."""
        with self._lock:
            if self._state == PlaybackState.PLAYING:
                self._set_state(PlaybackState.STOPPED)
                if self._source:
                    self._source.seek(0.0)
                if self.on_playback_finished:
                    self.on_playback_finished()

    def play(self) -> None:
        """Inicia ou retoma a reprodução."""
        with self._lock:
            if self._source is None:
                return

            if self._stream is None or not self._stream.active:
                self._init_stream()

            if self._state == PlaybackState.PLAYING:
                return

            self._set_state(PlaybackState.PLAYING)
            self._start_monitor()

    def pause(self) -> None:
        """Pausa a reprodução mantendo a posição atual."""
        with self._lock:
            if self._state == PlaybackState.PLAYING:
                self._set_state(PlaybackState.PAUSED)

    def stop(self) -> None:
        """Interrompe a reprodução e retorna o cursor ao início."""
        with self._lock:
            self._set_state(PlaybackState.STOPPED)
            if self._source is not None:
                self._source.seek(0.0)
            if self.on_position_change:
                self.on_position_change(0.0)

    def seek(self, position_seconds: float) -> None:
        """Reposiciona o cursor de áudio de forma segura."""
        with self._lock:
            if self._source is not None:
                self._source.seek(position_seconds)
                current = self._source.get_position()
                if self.on_position_change:
                    self.on_position_change(current)

    def get_position(self) -> float:
        """Retorna o tempo de reprodução atual em segundos."""
        if self._source is not None:
            return self._source.get_position()
        return 0.0

    def get_duration(self) -> float:
        """Retorna a duração total do áudio em segundos."""
        if self._source is not None:
            return self._source.get_duration()
        return 0.0

    def _set_state(self, new_state: PlaybackState) -> None:
        self._state = new_state
        if self.on_state_change:
            self.on_state_change(new_state)

    def _start_monitor(self) -> None:
        """Inicia uma thread leve para despachar eventos periódicos de posição para a UI."""
        if self._monitor_running:
            return
        self._monitor_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        while self._monitor_running:
            if self._state == PlaybackState.PLAYING and self._source is not None:
                pos = self._source.get_position()
                if self.on_position_change:
                    self.on_position_change(pos)
            time.sleep(0.05)  # 20 atualizações por segundo para o cursor da UI

    def close(self) -> None:
        """Libera todos os recursos e threads."""
        self._monitor_running = False
        self.stop()
        self._close_stream()
        if self._source is not None:
            self._source.close()
            self._source = None
    OUTPUT_BLOCK_SIZE = 2048
