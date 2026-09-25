"""Orquestrador do 'Ouvido' e Contexto Musical do Virtual Band AI.

Executa a cadeia completa e integrada de DSP, MIR e Memória Temporal:
  1. Detecção de Pitch Monofônico Fundamental (f0)
  2. Extração de Cromagrama (12 Classes de Notas Temperadas)
  3. Análise Harmônica e Acordes Polifônicos com Inversões
  4. Estimativa de Tonalidade (Krumhansl-Schmuckler Adaptativo)
  5. Rastreamento de Andamento e Batidas (BPM & Beat Tracker)
  6. Medição Rigorosa de Latência da CPU (Nanossegundos)
  7. Gerenciador de Contexto Musical (MusicalContextManager) com:
     - Estabilização temporal de acordes (debounce / histerese)
     - Registro de eventos discretos de acordes (ChordHistory)
     - Rastreamento temporal de tonalidades (KeyHistory)
     - Sincronização com o Relógio Musical (MusicalClock)
"""

import threading
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from app.song.song_session import SongSession
    from app.instruments.registry import VirtualPlayerRegistry
import numpy as np

from app.analysis.pitch_detector import PitchDetector, PitchResult, create_pitch_detector, AVAILABLE_PITCH_DETECTORS
from app.analysis.chroma_extractor import ChromaExtractor
from app.analysis.active_notes import ActiveNoteTracker
from app.analysis.harmonic_analyzer import HarmonicAnalyzer, DefaultHarmonicAnalyzer
from app.analysis.chord_detector import Chord, ChordDetector
from app.analysis.chord_history import ChordHistory, ChordEvent
from app.analysis.key_detector import KeyDetector, KeyResult, KrumhanslSchmucklerKeyDetector
from app.analysis.key_history import KeyHistory, KeyEvent
from app.analysis.tempo_detector import OnsetTempoDetector, TempoResult
from app.music.musical_context import MusicalContext
from app.music.chord_chart import parse_chord
from app.music.musical_clock import MusicalClock
from app.music.context_manager import MusicalContextManager
from app.music.music_structure import MusicStructure
from app.analysis.music_structure_analyzer import MusicStructureAnalyzer
from app.analysis.structure_report import export_structure_report
from app.instruments.base import VirtualBand
from app.instruments.bass_player import BassPlayer
from app.utils.timing import LatencyTracker


class AudioAnalyzer:
    """Orquestrador central de análise em tempo real do sinal de áudio."""

    def __init__(self, pitch_algorithm: str = "Autocorrelação", sample_rate: int = 44100,
                 chunk_size: int = 4096, chroma_method: str = "harmonic", detect_extensions: bool = False):
        self._lock = threading.RLock()
        self._active_session: Optional["SongSession"] = None
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size

        # Módulos especializados de processamento DSP / MIR
        self._pitch_detector: PitchDetector = create_pitch_detector(pitch_algorithm)
        self._chroma_extractor: ChromaExtractor = ChromaExtractor(method=chroma_method)
        self._active_note_tracker = ActiveNoteTracker()
        # Extrator FFT dedicado à detecção de TOM (usado quando o de acorde é harmônico)
        self._fft_chroma: ChromaExtractor = ChromaExtractor(method="fft")
        self._harmonic_analyzer: HarmonicAnalyzer = DefaultHarmonicAnalyzer(detect_extensions=detect_extensions)
        self._key_detector: KeyDetector = KrumhanslSchmucklerKeyDetector()
        self._tempo_detector: OnsetTempoDetector = OnsetTempoDetector()

        # Medidor rigoroso de latência
        self._latency_tracker: LatencyTracker = LatencyTracker(
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            stabilization_ms=20.0
        )

        # Gerenciador de Contexto Musical e Memória Temporal
        self._context_manager = MusicalContextManager(
            sample_rate=sample_rate,
            bpm=120.0,
            meter="4/4"
        )

        # Analisador de Estrutura Musical, Memória de Padrões e Predição (v0.3)
        self._structure_analyzer = MusicStructureAnalyzer()

        # Banda Virtual (Músicos Autônomos que consomem o MusicalContext)
        self._bass_player: BassPlayer = BassPlayer(sample_rate=sample_rate)
        self._band: VirtualBand = VirtualBand()
        self._band.add_member(self._bass_player)

        # Expectativa harmônica ativa (opcional): serve de PRIOR musical para a detecção
        # de acordes ("ouvir esperando o que a cifra prevê"). Sem expectativa, ouvido cego.
        # É EMPURRADA pelo lado que conhece a posição real (SongSession/UI), evitando
        # acoplar-se a um relógio interno possivelmente dessincronizado.
        self._expected_chord = None
        self._key_hint = None


    @property
    def current_algorithm_name(self) -> str:
        return self._pitch_detector.algorithm_name

    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        return list(AVAILABLE_PITCH_DETECTORS.keys())

    @staticmethod
    def recommend_analysis_channel(audio: np.ndarray, sample_rate: int,
                                   probe_seconds: float = 30.0) -> Optional[int]:
        """Escolhe um canal estéreo fixo com evidência harmônica mais nítida.

        A escolha é feita uma vez por arquivo. Isso evita cancelamento de fase e
        também evita trocar de canal no meio da execução, o que pareceria uma
        mudança musical inexistente. ``None`` preserva a mistura mono quando os
        canais são equivalentes.
        """
        data = np.asarray(audio)
        if data.ndim != 2 or data.shape[1] < 2 or sample_rate <= 0:
            return None
        usable = min(len(data), int(max(1.0, probe_seconds) * sample_rate))
        if usable < 512:
            return None
        chunk_size = min(4096, usable)
        probe_step = max(chunk_size, sample_rate // 4)
        starts = np.arange(0, max(1, usable - chunk_size + 1), probe_step,
                           dtype=int)[:120]
        scores = []
        for channel in range(min(2, data.shape[1])):
            extractor = ChromaExtractor(method="harmonic")
            detector = ChordDetector()
            confidences = []
            for start in starts:
                chunk = data[start:start + chunk_size, channel]
                if float(np.sqrt(np.mean(np.square(chunk.astype(np.float64))))) < 0.005:
                    continue
                chroma = extractor.extract(chunk, sample_rate)
                result = detector.detect(chroma, chunk, sample_rate,
                                         timestamp=float(start) / sample_rate)
                if result.symbol != "--":
                    confidences.append(result.confidence)
            clarity = float(np.mean(confidences)) if confidences else 0.0
            scores.append(clarity)
        if len(scores) < 2 or max(scores) <= 0.0:
            return None
        best = int(np.argmax(scores))
        return best if scores[best] - min(scores) >= 0.0025 else None

    @property
    def context(self) -> MusicalContext:
        """Acesso thread-safe ao contexto musical consolidado."""
        return self._active_session.context if self._active_session is not None else self._context_manager.context

    @property
    def context_manager(self) -> MusicalContextManager:
        return self._context_manager

    @property
    def clock(self) -> MusicalClock:
        return self._context_manager.clock

    @property
    def chord_history(self) -> ChordHistory:
        """Histórico temporal estabilizado de acordes."""
        return self._context_manager.chord_history

    @property
    def key_history(self) -> KeyHistory:
        """Histórico temporal estabilizado de tonalidades."""
        return self._context_manager.key_history

    @property
    def structure_analyzer(self) -> MusicStructureAnalyzer:
        """Analisador de forma e estrutura musical."""
        return self._active_session.structure_analyzer if self._active_session is not None else self._structure_analyzer

    @property
    def pattern_memory(self):
        """Memória de padrões harmônicos e transições."""
        return self.structure_analyzer.pattern_memory

    @property
    def prediction_engine(self):
        """Motor probabilístico de predição e antecipação."""
        return self.structure_analyzer.prediction_engine

    @property
    def position_estimator(self):
        """Estimador de posicionamento musical e progresso de seção."""
        return self.structure_analyzer.position_estimator

    @property
    def music_structure(self) -> MusicStructure:
        """Estrutura musical global consolidada."""
        return self.structure_analyzer.structure

    @property
    def bass_player(self) -> BassPlayer:
        """Acesso direto ao Baixista Virtual."""
        return self._bass_player

    @property
    def band(self) -> VirtualBand:
        """Acesso ao orquestrador da Banda Virtual."""
        return self._band

    @property
    def tempo_detector(self) -> OnsetTempoDetector:
        return self._tempo_detector

    def set_detect_extensions(self, enabled: bool) -> None:
        """Liga/desliga tétrades/suspensos (7,7M,m7,sus) sem mexer no cromagrama harmônico."""
        with self._lock:
            if hasattr(self._harmonic_analyzer, "set_detect_extensions"):
                self._harmonic_analyzer.set_detect_extensions(enabled)

    def set_harmonic_resolution(self, enabled: bool) -> None:
        """Ativa/desativa o modo de ALTA RESOLUÇÃO HARMÔNICA.

        Quando ligado, usa cromagrama por soma harmônica (suprime vazamento de overtones)
        + reconhecimento de tétrades/suspensos (7, 7M, m7, sus). Recomendado para
        acompanhamento de cifra, onde a qualidade do acorde importa. Desligado, mantém o
        cromagrama FFT clássico e apenas tríades (mais leve).
        """
        with self._lock:
            self._chroma_extractor = ChromaExtractor(method="harmonic" if enabled else "fft")
            self._active_note_tracker.reset()
            if hasattr(self._harmonic_analyzer, "set_detect_extensions"):
                self._harmonic_analyzer.set_detect_extensions(enabled)

    def set_harmonic_expectation(self, expected_chord=None, key=None) -> None:
        """Define o prior musical do próximo frame: o acorde esperado da cifra e/ou o tom.

        Deve ser chamado pelo lado que conhece a posição real (ex.: SongSession) antes de
        ``analyze_chunk``. Passe ``None`` para voltar à audição cega (modo livre).
        """
        with self._lock:
            self._expected_chord = expected_chord if expected_chord not in ("", "--") else None
            self._key_hint = key if key not in ("", "--") else None

    def _harmonic_prior_for(self, session=None):
        """Libera o prior da cifra apenas quando a posição está estável."""
        active = session if session is not None else self._active_session
        event_count = getattr(getattr(active, "alignment", None), "event_count", 0)
        chart_active = isinstance(event_count, int) and event_count > 0
        expected = active.expected_chord if chart_active else self._expected_chord
        if expected is None or expected == "--":
            return None, False
        if active is None:
            return expected, True
        state = getattr(active, "tracking_state", "TRACKING")
        state = getattr(state, "value", state)
        confidence = float(getattr(active, "position_confidence", 0.0))
        if state != "TRACKING" or confidence < 0.70:
            return None, False
        ctx = active.context if chart_active else None
        competing_candidate = False
        if ctx is not None:
            candidate_root = getattr(ctx, "chord_candidate_root", "--")
            expected_root = parse_chord(expected).root
            competing_candidate = (candidate_root not in ("", "--", expected_root) and
                                   getattr(ctx, "chord_candidate_frames", 0) >= 3 and
                                   getattr(ctx, "chord_candidate_confidence", 0.0) >= 0.55)
        enabled = (not getattr(ctx, "stable_chord_stale", False) and
                   not competing_candidate)
        return (expected if enabled else None), enabled

    def set_pitch_detector(self, algorithm_name: str) -> None:
        """Altera o detector de pitch em tempo de execução."""
        with self._lock:
            self._pitch_detector = create_pitch_detector(algorithm_name)
            print(f"[AudioAnalyzer] Algoritmo de pitch alterado para: {self._pitch_detector.algorithm_name}")

    def pre_analyze_track(self, mono_audio: np.ndarray, sample_rate: int) -> float:
        """Pré-analisa a faixa de áudio para extração global de BPM e grade de batidas."""
        with self._lock:
            self.reset_musical_history()
            bpm = self._tempo_detector.analyze_audio(mono_audio, sample_rate)
            valid_bpm = round(bpm, 1) if bpm > 0 else 120.0
            self._context_manager.clock.bpm = valid_bpm
            self._context_manager.context.bpm = valid_bpm
            return valid_bpm

    def apply_tempo_analysis(self, detector: OnsetTempoDetector) -> float:
        """Aplica apenas o resultado da faixa ativa, sem apagar seu histórico."""
        with self._lock:
            self._tempo_detector = detector
            bpm = round(detector.bpm, 1) if detector.bpm > 0 else 120.0
            self._context_manager.clock.bpm = bpm
            self._context_manager.context.bpm = bpm
            return bpm

    def reset_musical_history(self) -> None:
        """Reinicia acumuladores de histórico, relógio e instrumentos ao carregar nova música ou dar stop."""
        with self._lock:
            self._active_session = None
            self._tempo_detector.reset()
            self._active_note_tracker.reset()
            self._key_detector.reset()
            self._harmonic_analyzer.reset()
            self._context_manager.reset()
            self._structure_analyzer.reset()
            self._band.reset_all()

    def update_audio_format(self, sample_rate: int, chunk_size: int,
                            capture_latency_ms: float = 0.0,
                            output_latency_ms: float = 0.0) -> None:
        """Atualiza a taxa de amostragem e tamanho do bloco para os analisadores e instrumentos."""
        with self._lock:
            self._sample_rate = sample_rate
            self._chunk_size = chunk_size
            self._latency_tracker.update_config(sample_rate, chunk_size)
            self._latency_tracker.set_external_latency(capture_latency_ms, output_latency_ms)
            self._context_manager.context.sample_rate = sample_rate
            self._bass_player.synthesizer.set_sample_rate(sample_rate)

    def analyze_chunk(self, audio_chunk: np.ndarray, sample_rate: int, timestamp: float,
                      session: Optional["SongSession"] = None,
                      players: Optional["VirtualPlayerRegistry"] = None) -> MusicalContext:
        """Executa a cadeia completa de análise musical e despacha o MusicalContext para a Banda Virtual."""
        with self._lock:
            self._latency_tracker.start_measurement()

            # 1. Análise de Pitch Fundamental (f0)
            pitch_res: PitchResult = self._pitch_detector.detect(audio_chunk, sample_rate)
            audio_activity = float(np.sqrt(np.mean(
                np.square(np.asarray(audio_chunk, dtype=np.float64)))))

            # 2. Extração de Cromagrama (12 Classes de Notas)
            raw_chroma: np.ndarray = self._chroma_extractor.extract(audio_chunk, sample_rate)

            # A origem do tom é uma escolha independente da existência da
            # cifra. O prior harmônico da cifra continua ativo nos dois modos.
            has_chart = bool(session is not None and session.alignment.event_count)
            key_source = (getattr(session.song.performance_settings, "key_source", "CHART").upper()
                          if session is not None else "AUDIO")
            chart_key = ((session.song.performance_settings.key_override or session.chart.key)
                         if has_chart and key_source == "CHART" else None)
            key_hint = chart_key or self._key_hint
            active_state = self._active_note_tracker.update(
                raw_chroma, timestamp, key=key_hint or "--", audio_activity=audio_activity)
            chroma = active_state.chroma

            # 2b. No modo de alta resolução, a detecção de TOM continua usando o cromagrama
            #     FFT clássico (calibração validada do Krumhansl-Schmuckler); apenas a
            #     detecção de ACORDE usa o cromagrama harmônico.
            key_chroma = None
            if chart_key is None:
                key_chroma = (self._fft_chroma.extract(audio_chunk, sample_rate)
                              if self._chroma_extractor.method == "harmonic" else raw_chroma)

            # 3. Análise Harmônica & Detecção de Acordes (com Inversões)
            #    Usa a expectativa da cifra (se definida) como prior musical da detecção.
            expected_for_frame, chart_prior_enabled = self._harmonic_prior_for(session)
            chord_res: Chord = self._harmonic_analyzer.analyze_harmony(
                audio_chunk=audio_chunk,
                sample_rate=sample_rate,
                dominant_pitch=pitch_res,
                chroma_vector=chroma,
                timestamp=timestamp,
                expected_chord=expected_for_frame,
                key=key_hint
            )

            # 4. Estimativa de Tonalidade (Krumhansl-Schmuckler Dual-Timeframe)
            key_res: Optional[KeyResult] = (self._key_detector.estimate_key(key_chroma, timestamp=timestamp)
                                            if key_chroma is not None else None)

            # 5. Rastreamento de Andamento & Batida
            self._tempo_detector.observe_chunk(audio_chunk, sample_rate, timestamp)
            tempo_res: TempoResult = self._tempo_detector.get_tempo_at_time(timestamp)

            # 6. Finalizar medição de tempo DSP da CPU
            lat_metrics = self._latency_tracker.end_measurement()

            # 7. Atualização do MusicalContextManager (Estabilização + Memória + Relógio)
            ctx = self._context_manager.update(
                timestamp=timestamp,
                sample_rate=sample_rate,
                pitch_res=pitch_res,
                chord_res=chord_res,
                key_res=key_res,
                tempo_res=tempo_res,
                chroma_vector=chroma,
                raw_chroma_vector=raw_chroma,
                active_notes=active_state.notes,
                lat_metrics=lat_metrics,
                audio_activity=audio_activity,
            )
            ctx.expected_chart_chord = (session.expected_chord if has_chart
                                        else self._expected_chord) or "--"
            ctx.chart_prior_enabled = chart_prior_enabled

            # Localiza primeiro; estrutura, predição e instrumentos só recebem a posição final.
            self._active_session = session
            if session is not None:
                session.update_audio_tick(
                    timestamp=timestamp, detected_chord=ctx.chord,
                    detected_confidence=ctx.chord_confidence, detected_key=ctx.key,
                    detected_bpm=ctx.bpm, source_context=ctx, tempo_result=tempo_res)
                ctx = session.context
            else:
                # No modo livre, não há cifra: coordenadas musicais coincidem com o relógio.
                self._structure_analyzer.update_online(
                    ctx, self._context_manager.chord_history,
                    self._context_manager.key_history, self._context_manager.clock)
            if players is not None:
                players.dispatch_context(ctx)
            else:
                self._band.dispatch_context(ctx)

            return ctx

    def export_structure_report(self, filepath: str) -> None:
        """Exporta o relatório estrutural completo para arquivo JSON."""
        with self._lock:
            export_structure_report(self.structure_analyzer.structure, filepath)

    def analyze_offline_structure(self, chords_with_timing: list, bpm: float = 120.0):
        """Executa a análise estrutural offline completa a partir de eventos de acordes."""
        with self._lock:
            return self._structure_analyzer.analyze_full_song_from_events(chords_with_timing, bpm)


