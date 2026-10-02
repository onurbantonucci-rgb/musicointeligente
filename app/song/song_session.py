"""Sessão de Execução Musical da Música Ativa (SongSession v0.4).

Gerencia exclusivamente o estado de execução temporário e volátil da música selecionada:
MusicalContext, MusicalClock, ChartAlignment, ChartAudioFusion e VirtualPlayers.
Garante isolamento hermético entre trocas de músicas para evitar memory leaks.
"""

from typing import Optional, Dict, Any, List
import copy
from dataclasses import fields
from dataclasses import dataclass, replace
from enum import Enum

from app.song.song import Song
from app.music.musical_context import MusicalContext
from app.music.musical_clock import MusicalClock
from app.music.chord_chart import ChordChart, parse_chord
from app.music.transposition import transpose_chord_symbol
from app.music.theory import note_to_pc
from app.input.chart_parser import ChartParser
from app.input.chart_semantic_classifier import ChartSemanticClassifier
from app.music.chart_alignment import ChartAlignment, ChartPosition, StartAnchor
from app.music.chart_audio_fusion import ChartAudioFusion, FusedMusicalState
from app.analysis.music_structure_analyzer import MusicStructureAnalyzer
from app.music.prediction_engine import PredictionEngine, MusicPrediction
from app.music.position_estimator import PositionEstimator, TrackingState
from app.music.online_chart_follower import OnlineChartFollower
from app.analysis.tempo_detector import TempoResult
from app.music.follow_confidence import calculate_follow_confidence
from app.music.harmonic_rhythm import HarmonicRhythmTracker


@dataclass
class LyricView:
    """O que está sendo cantado agora, para exibir como o acorde atual."""
    state: str = "sem letra"        # "cantando", "instrumental" ou "sem letra"
    line_number: int = 0            # linha física no texto da cifra (1 = primeira)
    line_text: str = ""
    segment: str = ""               # trecho cantado sobre o acorde atual
    segment_start: int = -1         # coluna do trecho na linha (-1 = sem destaque)
    next_line: str = ""


class PerformanceState(str, Enum):
    """Estado da banda; distinto do rastreamento de posição e do relógio."""
    PLAYING = "PLAYING"
    UNCERTAIN = "UNCERTAIN"
    HOLDING = "HOLDING"
    WAITING = "WAITING"
    RECOVERING = "RECOVERING"
    ENDED = "ENDED"


class SongSession:
    """Representa a sessão ativa de execução e reprodução de uma Song.
    
    Toda a interface gráfica e os instrumentos virtuais devem consultar esta classe
    para obter a verdade sobre o momento musical presente.
    """
    _SHORT_PAUSE_SECONDS = 2.5
    _WAITING_SECONDS = 4.0
    _END_SILENCE_SECONDS = 8.0
    _RECOVERY_MIN_SECONDS = 0.30

    def __init__(self, song: Song, mode: str = "PLAYBACK"):
        self._song: Song = song
        self._mode: str = mode          # "PLAYBACK", "FOLLOW", "MANUAL"
        self._is_playing: bool = False
        self._is_paused: bool = False

        # 1. Relógio Musical isolado
        self._clock: MusicalClock = MusicalClock(
            bpm=song.performance_settings.bpm_override or song.bpm,
            meter=song.performance_settings.meter_override or song.meter
        )

        # 2. Contexto Musical isolado
        self._context: MusicalContext = MusicalContext(
            bpm=self._clock.bpm,
            meter=self._clock.meter,
            key=song.performance_settings.key_override or song.key
        )

        # 3. Cifra estruturada (parseada da canção)
        if song.chart_text:
            # O editor e o destaque usam linhas físicas do texto. Dados
            # estruturados antigos podem conter índices defasados após
            # transposição/importação; reconstruí-los do texto evita iniciar
            # numa linha diferente daquela exibida.
            self._chart: ChordChart = ChartParser.parse(
                text=song.chart_text,
                default_key=song.key,
                default_bpm=song.bpm,
                default_meter=song.meter,
                title=song.title,
                artist=song.artist
            )
        elif song.chart_data:
            self._chart = ChordChart.from_dict(song.chart_data)
        else:
            self._chart = ChordChart(
                title=song.title,
                artist=song.artist,
                key=song.key,
                bpm=song.bpm,
                meter=song.meter
            )
        if self._key_source() == "AUDIO":
            self._context.key = "--"

        # 4. Alinhamento e Fusão
        self._alignment: ChartAlignment = ChartAlignment(self._chart)
        self._fusion: ChartAudioFusion = ChartAudioFusion()

        # 5. Motores de Estrutura e Predição isolados
        self._structure_analyzer: MusicStructureAnalyzer = MusicStructureAnalyzer()
        self._prediction_engine: PredictionEngine = self._structure_analyzer.prediction_engine
        self._harmonic_rhythm = HarmonicRhythmTracker(self._structure_analyzer.pattern_memory)

        # 6. Estimador Contínuo de Posição Musical (v0.4)
        self._position_estimator: PositionEstimator = PositionEstimator(
            alignment=self._alignment,
            clock=self._clock,
            structure_analyzer=self._structure_analyzer,
            prediction_engine=self._prediction_engine
        )
        # Informa o capotraste da cifra (CifraClub) para localizar pelo som real
        try:
            self._position_estimator.set_capo(self._chart.capo_semitones)
        except Exception:
            pass

        # Se a música já possui padrões e estruturas salvos, restaura-os
        if song.pattern_memory_data:
            pass # Pode ser deserializado no futuro

        # 7. Estado Instantâneo Consolidado
        self._position_generation = 0
        self._chart_alignment_confidence = max(0.0, min(1.0, float(
            song.metadata.get("chart_overall_confidence", 1.0))))
        self._follow_stability = 0.70
        self._last_follow_timestamp: Optional[float] = None
        self._follow_observations_available = False
        self._last_harmonic_input_diagnostic: Dict[str, Any] = {
            "raw": "--", "semantic_type": "UNKNOWN", "harmonic_eligible": False, "action": "IGNORED"
        }
        self._performance_state = PerformanceState.WAITING
        self._last_activity_time: Optional[float] = None
        self._recovery_started_at: Optional[float] = None
        self._recovery_evidence = 0
        # O tempo 1 do relógio já foi alinhado ao primeiro ataque da música?
        self._downbeat_anchored = False
        # Portão de início: a música só "começa" quando o áudio confirma o PRIMEIRO
        # acorde da cifra (contagem da bateria, ruído ou outra nota são ignorados).
        self._music_started = False
        self._lyrics_status = "Letra: aguardando voz"
        self._start_candidate_since: Optional[float] = None
        self._start_onset: Optional[float] = None
        self._harmonic_without_start = 0.0
        self._last_gate_tick: Optional[float] = None
        # Seguidor probabilístico (HMM causal): alcança a banda quando trocas de acorde
        # não foram detectadas, em vez de esperar o acorde voltar no ciclo seguinte.
        self._chart_follower: Optional[OnlineChartFollower] = None
        self._chart_follower_key = None
        self._follower_cursor = -1
        self._follower_candidate = (-1, 0)
        self._current_chart_pos: ChartPosition = self._position_estimator.refresh_position()
        self._last_confident_chart_pos = self._current_chart_pos
        self._current_fused_state: FusedMusicalState = FusedMusicalState(
            expected_chord=self._current_chart_pos.current_chord,
            effective_chord=self._current_chart_pos.current_chord,
            next_expected_chord=self._current_chart_pos.next_chord,
            current_section=self._current_chart_pos.section_name,
            next_section=self._current_chart_pos.next_section_name
        )
        self._publish_position(update_structure=False)
        self._player_states: Dict[str, Any] = {
            "bass": "READY",
            "drums": "READY",
            "keyboard": "READY",
            "guitar": "READY",
        }

    # ============================================================
    # Propriedades de Consulta Central
    # ============================================================
    @property
    def song(self) -> Song:
        return self._song

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, new_mode: str) -> None:
        self._mode = new_mode

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def performance_state(self) -> str:
        return self._performance_state.value

    @property
    def clock(self) -> MusicalClock:
        return self._clock

    @property
    def context(self) -> MusicalContext:
        return self._context

    @property
    def chart(self) -> ChordChart:
        return self._chart

    @property
    def alignment(self) -> ChartAlignment:
        return self._alignment

    @property
    def fusion(self) -> ChartAudioFusion:
        return self._fusion

    @property
    def structure_analyzer(self) -> MusicStructureAnalyzer:
        return self._structure_analyzer

    @property
    def current_bar(self) -> int:
        """Compasso musical oficial, após localização/reancoragem na cifra."""
        return self._current_chart_pos.current_bar

    @property
    def current_beat(self) -> int:
        return int(self._current_chart_pos.current_beat)

    @property
    def clock_bar(self) -> int:
        """Compasso bruto calculado a partir do tempo do áudio."""
        return self._clock.bar

    @property
    def clock_beat(self) -> int:
        return self._clock.beat

    @property
    def current_chord(self) -> str:
        return self._current_fused_state.effective_chord

    @property
    def expected_chord(self) -> str:
        return self._current_fused_state.expected_chord

    @property
    def detected_chord(self) -> str:
        return self._current_fused_state.detected_chord

    @property
    def next_chord(self) -> str:
        return self._current_fused_state.next_expected_chord

    @property
    def current_section(self) -> str:
        return self._current_fused_state.current_section

    @property
    def next_section(self) -> str:
        return self._current_fused_state.next_section

    @property
    def current_lyric(self) -> str:
        return self._current_fused_state.lyric

    @property
    def chart_position(self) -> ChartPosition:
        return self._current_chart_pos

    @property
    def start_anchor(self) -> Optional[StartAnchor]:
        return self._alignment.start_anchor

    @property
    def position_estimator(self) -> PositionEstimator:
        return self._position_estimator

    @property
    def tracking_state(self) -> str:
        return self._current_chart_pos.tracking_state

    @property
    def position_confidence(self) -> float:
        return self._current_chart_pos.confidence

    @property
    def current_line(self) -> int:
        return self._current_chart_pos.line_index

    @property
    def prediction(self) -> Optional[MusicPrediction]:
        return self._prediction_engine.latest_prediction

    @property
    def player_states(self) -> Dict[str, Any]:
        return dict(self._player_states)

    # ============================================================
    # Ciclo de Vida e Limpeza
    # ============================================================
    def start(self) -> None:
        if self._clock.elapsed_time <= 0.0 and not self._is_paused:
            self._current_chart_pos = self._position_estimator.seek_to_start()
            self._publish_position(update_structure=False)
        self._is_playing = True
        self._is_paused = False
        if self._performance_state == PerformanceState.ENDED:
            self._set_performance_state(PerformanceState.WAITING)

    def pause(self) -> None:
        self._is_paused = True

    def resume(self) -> None:
        self._is_paused = False
        self._is_playing = True

    def stop(self) -> None:
        self._is_playing = False
        self._is_paused = False
        self.reset()

    def reset(self) -> None:
        """Reinicia os relógios, fusão e contextos para o início da música (t=0.0)."""
        self._position_generation += 1
        self._performance_state = PerformanceState.WAITING
        self._last_activity_time = None
        self._recovery_started_at = None
        self._recovery_evidence = 0
        self._downbeat_anchored = False
        self._music_started = False
        self._lyrics_status = "Letra: aguardando voz"
        self._start_candidate_since = None
        self._start_onset = None
        self._harmonic_without_start = 0.0
        self._last_gate_tick = None
        if self._chart_follower is not None:
            self._chart_follower.reset(None)
        self._follower_cursor = -1
        self._follower_candidate = (-1, 0)
        self._follow_stability = 0.70
        self._last_follow_timestamp = None
        self._follow_observations_available = False
        if (self._bpm_source() == "CHART" or self._song.performance_settings.bpm_override or
                not self._follow_observations_available):
            self._clock.bpm = self._chart_bpm()
        self._clock.reset()
        self._context.reset()
        self._context.bpm = self._clock.bpm
        self._context.meter = self._clock.meter
        self._context.key = self._chart_key() if self._key_source() == "CHART" else "--"
        self._fusion.reset()
        self._structure_analyzer.reset()
        self._harmonic_rhythm.reset()
        self._position_estimator.reset()
        self._current_chart_pos = self._position_estimator.refresh_position()
        self._last_confident_chart_pos = self._current_chart_pos
        self._current_fused_state = FusedMusicalState(
            expected_chord=self._current_chart_pos.current_chord,
            effective_chord=self._current_chart_pos.current_chord,
            next_expected_chord=self._current_chart_pos.next_chord,
            current_section=self._current_chart_pos.section_name,
            next_section=self._current_chart_pos.next_section_name
        )
        self._publish_position(update_structure=False)

    def restart(self) -> ChartPosition:
        """Restart completo; Pause/Resume continuam preservando a posição."""
        was_playing = self._is_playing and not self._is_paused
        self.reset()
        if was_playing:
            self.start()
        return self._current_chart_pos

    def _bpm_source(self) -> str:
        return getattr(self._song.performance_settings, "bpm_source", "AUDIO").upper()

    def _key_source(self) -> str:
        return getattr(self._song.performance_settings, "key_source", "CHART").upper()

    def _chart_bpm(self) -> float:
        return self._song.performance_settings.bpm_override or self._chart.bpm or self._song.bpm

    def _chart_key(self) -> str:
        return self._song.performance_settings.key_override or self._chart.key or self._song.key

    def set_bpm_source(self, source: str) -> None:
        """Altera a fonte do andamento sem criar outro relógio musical."""
        source = str(source).upper()
        if source not in ("AUDIO", "CHART"):
            raise ValueError("Fonte de BPM inválida")
        self._song.performance_settings.bpm_source = source
        if source == "CHART":
            self._clock.bpm = self._chart_bpm()
            self._context.bpm = self._clock.bpm

    def set_key_source(self, source: str) -> None:
        """Altera a autoridade da tonalidade publicada no contexto."""
        source = str(source).upper()
        if source not in ("AUDIO", "CHART"):
            raise ValueError("Fonte de tom inválida")
        self._song.performance_settings.key_source = source
        if source == "CHART":
            self._context.key = self._chart_key()
            self._context.key_confidence = 1.0
        else:
            # A tela não deve continuar exibindo o tom da cifra enquanto o
            # detector de áudio ainda junta seus primeiros frames.
            self._context.key = "--"
            self._context.key_confidence = 0.0

    def close(self) -> None:
        """Libera integralmente todos os recursos e estados da sessão anterior."""
        self.stop()
        self.reset()

    # ============================================================
    # Atualização Contínua de Tempo e Áudio
    # ============================================================
    def update_audio_tick(
        self,
        timestamp: float,
        detected_chord: str = "--",
        detected_confidence: float = 0.0,
        detected_key: str = "--",
        detected_bpm: float = 0.0,
        source_context: Optional[MusicalContext] = None,
        tempo_result: Optional[TempoResult] = None,
    ) -> FusedMusicalState:
        """Tempo bruto → estimador → snapshot musical → todos os consumidores."""
        if source_context is not None:
            self._follow_observations_available = True
        if timestamp < self._clock.elapsed_time - 0.01:
            # Seek da fonte invalida o horário de qualquer evento preparado.
            self._position_generation += 1
        if (not self._music_started and timestamp > 5.0 and
                abs(timestamp - self._clock.elapsed_time) > 1.0 and self._clock.elapsed_time > 0.0):
            # Seek para o meio da música: o portão de INÍCIO não se aplica.
            self._music_started = True
        chart_bpm = self._chart_bpm()
        manual_bpm = self._song.performance_settings.bpm_override
        # A fonte de BPM é independente da fonte harmônica. Em Cifra não
        # enviamos pulsos ao PLL, pois eles acabariam alterando a grade que o
        # usuário pediu para manter fixa.
        if manual_bpm:
            bpm_input = manual_bpm
            beat_timestamp = None
            observation_confidence = 0.0
        elif self._bpm_source() == "CHART":
            bpm_input = chart_bpm
            beat_timestamp = None
            observation_confidence = 0.0
        elif tempo_result is not None:
            bpm_input = (tempo_result.bpm if tempo_result.bpm > 0 and
                         tempo_result.confidence >= .65 else None)
            beat_timestamp = tempo_result.beat_timestamp
            observation_confidence = tempo_result.confidence
        else:
            bpm_input = detected_bpm if detected_bpm > 0 else None
            beat_timestamp = None
            observation_confidence = 0.0
        self._clock.update(
            timestamp, bpm=bpm_input,
            beat_timestamp=beat_timestamp,
            observation_confidence=observation_confidence)
        raw_detected_chord = detected_chord
        if (detected_chord not in ("", "--", "UNKNOWN", "N") and
                not ChartSemanticClassifier.is_chord_shaped(detected_chord)):
            self._last_harmonic_input_diagnostic = {
                "raw": raw_detected_chord,
                "semantic_type": ChartSemanticClassifier.classify_line(raw_detected_chord).line_type.value,
                "harmonic_eligible": False,
                "action": "IGNORED",
            }
            detected_chord = "--"
            detected_confidence = 0.0
        else:
            self._last_harmonic_input_diagnostic = {
                "raw": raw_detected_chord,
                "semantic_type": "CHORD" if detected_chord not in ("", "--", "UNKNOWN", "N") else "UNKNOWN",
                "harmonic_eligible": detected_chord not in ("", "--", "UNKNOWN", "N"),
                "action": "ACCEPTED" if detected_chord not in ("", "--", "UNKNOWN", "N") else "IGNORED",
            }
        observed_onset = None
        if (source_context is not None and
                source_context.smoothed_detected_chord == detected_chord and
                detected_chord not in ("", "--", "UNKNOWN", "N") and
                0.0 <= timestamp - source_context.chord_start_time <= 1.5):
            observed_onset = max(0.0, self._clock.total_beats -
                                 (timestamp - source_context.chord_start_time) *
                                 self._clock.bpm / 60.0)
        self._harmonic_rhythm.observe(
            self._clock.total_beats, self._current_chart_pos.section_name,
            detected_chord, detected_confidence,
            active_notes=source_context.active_notes if source_context is not None else None,
            observation_available=source_context is not None,
            transition_beat=observed_onset)
        activity = self._has_musical_activity(source_context, detected_chord,
                                               detected_confidence, tempo_result)
        just_started = False
        if not self._music_started and source_context is not None:
            just_started = self._confirms_song_start(timestamp, detected_chord,
                                                     detected_confidence, source_context)
            if not just_started:
                # Antes do primeiro acorde da cifra, nada conta como música.
                activity = False
        previous_state = self._performance_state
        should_localize = self._update_performance_state(
            timestamp, activity, activity_observable=source_context is not None)
        if just_started:
            # Confirmado o primeiro acorde: entra tocando, sem esperar outro tempo 1.
            self._set_performance_state(PerformanceState.PLAYING)
            should_localize = True
        self._anchor_downbeat_on_music_start(previous_state, timestamp, source_context, tempo_result)
        if should_localize:
            old_offset = self._position_estimator.bar_offset
            self._current_chart_pos = self._position_estimator.update(
                timestamp=timestamp, detected_chord=detected_chord,
                detected_confidence=detected_confidence, detected_key=detected_key,
                # f0 monofônico é diagnóstico: somente acordes/eventos
                # polifônicos estabilizados podem deslocar a cifra.
                detected_note="--", note_confidence=0.0,
                harmonic_rhythm_events=self._harmonic_rhythm.timeline,
                current_chord_elapsed_beats=self._harmonic_rhythm.state.current_elapsed_beats,
                expected_chord_duration_beats=self._harmonic_rhythm.state.expected_chord_duration_beats,
                duration_confidence=self._harmonic_rhythm.state.duration_confidence,
                audio_observable=source_context is not None)
            if self._position_estimator.bar_offset != old_offset:
                self._position_generation += 1
            if self._music_started and source_context is not None:
                self._follow_chart_by_harmony(timestamp, source_context)
            if activity and self._current_chart_pos.confidence >= 0.50:
                self._last_confident_chart_pos = self._current_chart_pos
        else:
            # Mantém a última posição confirmada: pausa não é avanço nem fim da música.
            self._current_chart_pos = replace(self._last_confident_chart_pos,
                                               absolute_time=timestamp)
        return self._publish_position(detected_chord, detected_confidence, detected_key, source_context)

    def _anchor_downbeat_on_music_start(self, previous_state: PerformanceState, timestamp: float,
                                        source_context: Optional[MusicalContext],
                                        tempo_result: Optional[TempoResult]) -> None:
        """Alinha o TEMPO 1 do relógio ao primeiro ataque da música, uma única vez.

        O relógio numera batidas a partir de t=0 do arquivo. Se a música começa
        depois (silêncio, respiração, contagem), sem este alinhamento as barras de
        compasso ficam deslocadas e a cifra projetada troca de acorde fora do lugar.
        Só vale para o início real da música: atividade observada no áudio, logo
        após o estado de espera, com a cifra ainda no evento inicial.
        """
        if (self._downbeat_anchored or source_context is None or
                previous_state != PerformanceState.WAITING or
                self._performance_state not in (PerformanceState.RECOVERING, PerformanceState.PLAYING)):
            return
        anchor = self._alignment.start_anchor
        if anchor is not None and self._current_chart_pos.event_index != anchor.event_index:
            return
        observed_beat = None
        if tempo_result is not None:
            # A atividade é vista com ~90 ms de antecedência (janela à frente da
            # análise): o ataque real é a PRÓXIMA batida pré-analisada, se estiver perto.
            upcoming = tempo_result.next_beat_timestamp
            if upcoming is not None and -0.04 <= upcoming - timestamp <= 0.25:
                observed_beat = upcoming
            elif tempo_result.beat_timestamp is not None:
                observed_beat = tempo_result.beat_timestamp
        if self._start_onset is not None:
            # O primeiro acorde é confirmado alguns décimos depois de soar: o tempo 1
            # é a batida (já rastreada, ex.: na contagem da bateria) mais próxima
            # do instante em que ele SOOU.
            self._clock.align_downbeat_at(self._start_onset)
        elif observed_beat is not None:
            # Batida real da pré-análise: fixa fase e tempo 1 exatamente nela.
            self._clock.align_downbeat_at(observed_beat, snap_phase=True)
        else:
            self._clock.align_downbeat_at(timestamp)
        self._downbeat_anchored = True
        # Previsões feitas na numeração antiga de compassos não valem mais.
        self._position_generation += 1

    # Tempo que o primeiro acorde precisa se manter para abrir o portão, e quanto
    # tempo de harmonia SEM o primeiro acorde faz desistir de esperá-lo (gravação
    # que começa em outro ponto da cifra: a localização global assume).
    _START_HOLD_SECONDS = 0.25
    _START_GIVE_UP_SECONDS = 12.0

    def _start_chords(self) -> List[str]:
        anchor = self._alignment.start_anchor
        if anchor is None:
            return []
        chords = [anchor.chord]
        capo = self._chart.capo_semitones or self._position_estimator.capo_semitones
        if capo:
            chords.append(transpose_chord_symbol(anchor.chord, capo))
        return chords

    def _confirms_song_start(self, timestamp: float, chord: str, confidence: float,
                             source_context: MusicalContext) -> bool:
        """O áudio está tocando o PRIMEIRO acorde da cifra, de forma estável?"""
        dt = 0.0 if self._last_gate_tick is None else max(0.0, timestamp - self._last_gate_tick)
        self._last_gate_tick = timestamp
        expected = self._start_chords()
        if not expected:
            self._music_started = True
            return True
        harmonic = chord not in ("", "--", "UNKNOWN", "N") and confidence >= 0.45
        roots = {parse_chord(c).root for c in expected}
        matches = harmonic and parse_chord(chord).root in roots
        if not matches and not harmonic:
            # Começo dedilhado/arpejado: notas do PRIMEIRO acorde, começando pela
            # fundamental, também confirmam o início.
            note = getattr(source_context, "note", "--") or "--"
            note_ok = getattr(source_context, "note_confidence", 0.0) >= 0.6 and note != "--"
            if note_ok:
                pc = note_to_pc(note.rstrip("0123456789-"))
                tones = {pc_ for c in expected for pc_ in self._chord_pitch_classes(c)}
                root_pcs = {note_to_pc(r) for r in roots}
                matches = pc in root_pcs or (self._start_candidate_since is not None and pc in tones)
                harmonic = matches
        if not matches:
            self._start_candidate_since = None
            if harmonic:
                self._harmonic_without_start += dt
                if self._harmonic_without_start >= self._START_GIVE_UP_SECONDS:
                    # Começou em outro ponto: libera e deixa a localização achar onde.
                    self._music_started = True
                    self._start_onset = None
                    return True
            return False
        if chord not in ("", "--", "UNKNOWN", "N"):
            self._position_estimator.observe_start_chord(chord, confidence)
        onset = source_context.chord_start_time
        if self._start_candidate_since is None:
            self._start_candidate_since = timestamp
            self._start_onset = onset if 0.0 <= timestamp - onset <= 1.5 else timestamp
        if timestamp - self._start_candidate_since + dt < self._START_HOLD_SECONDS:
            return False
        self._music_started = True
        return True

    @staticmethod
    def _chord_pitch_classes(chord: str) -> set:
        symbol = parse_chord(chord)
        root = note_to_pc(symbol.root)
        third = 3 if symbol.quality in ("minor", "dim") else 4
        fifth = 6 if symbol.quality == "dim" else 8 if symbol.quality == "aug" else 7
        return {root, (root + third) % 12, (root + fifth) % 12}

    @property
    def music_started(self) -> bool:
        return self._music_started

    @property
    def lyrics_status(self) -> str:
        return self._lyrics_status

    def _harmonic_refinement(self, target: int) -> int:
        """A letra diz a PARTE da música; o acorde que soa agora escolhe o compasso exato.

        Palavras são cantadas no acorde sob elas ou um pouco antes: entre o
        compasso da palavra e o anterior, fica o do acorde que o áudio toca.
        """
        heard = self._context.smoothed_detected_chord or "--"
        if heard in ("", "--", "UNKNOWN", "N"):
            return max(1, target - 1)
        root = parse_chord(heard).root
        # O cantor canta a palavra NO acorde dela ou um pouco ANTES — nunca um
        # compasso depois. Na dúvida, o compasso anterior: o baixo segue no acorde
        # atual e troca quando a troca for ouvida (trocar cedo soaria errado).
        for bar in (target, target - 1):
            if bar < 1:
                continue
            chord = self._alignment.get_position_at(bar).current_chord
            if chord and chord != "--" and {parse_chord(c).root for c in
                                            [chord, transpose_chord_symbol(chord, self._chart.capo_semitones)]} & {root}:
                return bar
        return max(1, target - 1)

    def lyric_view(self) -> LyricView:
        """Letra no ponto atual da cifra: linha, trecho sob o acorde e próxima linha."""
        entry = self._alignment.event_entry(self._current_chart_pos.event_index)
        if entry is None:
            return LyricView()
        chord, section = entry
        segments = [s for sec in self._chart.sections for s in sec.lyrics]
        if not segments:
            return LyricView()
        same_section = [s for s in section.lyrics]
        current = [s for s in same_section if s.bar_offset == chord.bar]
        current.sort(key=lambda s: (not s.chord_symbol, s.line_number))
        holding = False
        if not current:
            # Acorde sem sílaba nova: a frase anterior ainda está soando (até 2 compassos).
            recent = [s for s in same_section if chord.bar - 2 <= s.bar_offset < chord.bar]
            if not recent:
                return LyricView(state="instrumental",
                                 next_line=self._next_lyric_line(segments, chord.line_number))
            current = [max(recent, key=lambda s: (s.bar_offset, s.line_number))]
            holding = True
        segment = current[0]
        lines = (self._chart.raw_text or "").split("\n")
        line_text = (lines[segment.line_number - 1].rstrip()
                     if 0 < segment.line_number <= len(lines) else "")
        if not line_text:
            line_text = " ".join(s.text.strip() for s in same_section
                                 if s.line_number == segment.line_number)
        piece = segment.text.strip()
        start = -1
        if piece and not holding:
            start = line_text.find(piece, max(0, getattr(chord, "column", 0) - len(piece)))
            if start < 0:
                start = line_text.find(piece)
        return LyricView(state="cantando", line_number=segment.line_number, line_text=line_text,
                         segment="" if holding else piece, segment_start=start,
                         next_line=self._next_lyric_line(segments, segment.line_number))

    def _next_lyric_line(self, segments, after_line: int) -> str:
        following = sorted({s.line_number for s in segments if s.line_number > after_line})
        if not following:
            return ""
        line_number = following[0]
        lines = (self._chart.raw_text or "").split("\n")
        if 0 < line_number <= len(lines) and lines[line_number - 1].strip():
            return lines[line_number - 1].strip()
        return " ".join(s.text.strip() for s in segments if s.line_number == line_number)

    # O seguidor probabilístico só move a cifra com certeza alta e repetida.
    _FOLLOWER_MIN_PROBABILITY = 0.6
    _FOLLOWER_CONFIRMATIONS = 2

    def _harmonic_follower(self) -> OnlineChartFollower:
        capo = self._chart.capo_semitones or self._position_estimator.capo_semitones
        key = (id(self._alignment), capo)
        if self._chart_follower is None or self._chart_follower_key != key:
            self._chart_follower = OnlineChartFollower(self._alignment.event_chords(), capo)
            self._chart_follower_key = key
            self._follower_cursor = -1
        return self._chart_follower

    def _follow_chart_by_harmony(self, timestamp: float, source_context: MusicalContext) -> None:
        """Adianta a cifra quando a harmonia acumulada mostra a banda à frente do cursor."""
        if not self._alignment.event_count:
            return
        follower = self._harmonic_follower()
        cursor = self._position_estimator.chart_cursor_index
        if not follower.active or cursor < self._follower_cursor:
            # Começo da música, repetição de seção ou navegação manual: recomeça no cursor.
            follower.reset(cursor)
        self._follower_cursor = cursor
        result = follower.observe(timestamp, source_context.raw_chroma_vector)
        if result is None:
            return
        best, probability = result
        # Só para frente: o seguidor alcança a banda, nunca faz a cifra voltar.
        if best <= cursor or probability < self._FOLLOWER_MIN_PROBABILITY:
            self._follower_candidate = (-1, 0)
            return
        seen = self._follower_candidate[1] + 1 if self._follower_candidate[0] == best else 1
        self._follower_candidate = (best, seen)
        if seen < self._FOLLOWER_CONFIRMATIONS:
            return
        self._position_estimator.relocate_to_event(
            best, "harmonic follower", f"probabilidade={probability:.2f}")
        self._follower_cursor = best
        self._follower_candidate = (-1, 0)
        self._position_generation += 1
        self._current_chart_pos = self._position_estimator.refresh_position()

    def _after_external_relocation(self) -> None:
        """Publica na hora a posição corrigida (não espera o próximo tick de áudio)."""
        self._position_generation += 1
        self._current_chart_pos = self._position_estimator.refresh_position()
        self._last_confident_chart_pos = self._current_chart_pos
        self._publish_position(update_structure=False)

    def apply_lyrics_evidence(self, evidence, now: float) -> str:
        """A letra cantada diz em que linha da cifra estamos: confirma ou corrige.

        Regras (as mesmas dos acordes): só depois que a música começou; corrige
        PARA FRENTE (a banda está adiante do cursor); para trás, só se a linha é
        o início de uma seção (repetição de refrão, por exemplo). Linhas iguais em
        vários lugares (refrões) escolhem a ocorrência mais próxima à frente.
        """
        if not self._music_started or evidence is None:
            return "ignorada"
        bpb = max(1, self._clock.beats_per_bar)
        elapsed_bars = max(0.0, (self._clock.beats_at(now) - self._clock.beats_at(evidence.time)) / bpb)
        written_bars = [evidence.bar] + [bar for bar, _ in evidence.alternatives]
        candidates = sorted({b for w in written_bars for b in self._alignment.bars_for_chart_bar(w)})
        if not candidates:
            return "sem compasso"
        current = self.current_bar
        # Onde a palavra cantada põe a banda AGORA (compasso da palavra + tempo decorrido).
        projected = [(b, b + int(elapsed_bars)) for b in candidates]
        line = evidence.line
        text = f"“{line.text}” ({line.section_name})"
        ahead = [p for _, p in projected if p >= current]
        if ahead:
            target = min(ahead)
            if target - current <= 1:
                # Cantores antecipam a palavra antes do tempo 1: até 1 compasso à
                # frente é confirmação (avançar aqui faria o baixo trocar cedo).
                self._lyrics_status = f"Letra confirma: {text}"
                return "confirmada"
            # Salto longo (a banda pulou partes que a cifra lista): só com a 1ª linha de
            # OUTRA parte, reconhecida com segurança — trechos com os mesmos acordes
            # (ponte x refrão) só a letra consegue distinguir.
            long_jump_ok = (line.first_in_section and evidence.confidence >= 0.75 and
                            max(evidence.weight, 0.6 * evidence.matched) >= 1.9 and
                            line.section_name != self.current_section)
            if target - current <= 16 or long_jump_ok:
                target = max(current, self._harmonic_refinement(target))
                if target - current <= 1:
                    self._lyrics_status = f"Letra confirma: {text}"
                    return "confirmada"
                self._position_estimator.relocate_to_bar(target, "lyrics", f"letra: {line.text}")
                self._after_external_relocation()
                self._lyrics_status = f"Letra corrigiu → Comp. {target}: {text}"
                return "corrigida"
        behind = [p for _, p in projected if p < current]
        if (behind and line.first_in_section and evidence.confidence >= 0.75 and
                max(evidence.weight, 0.6 * evidence.matched) >= 1.9 and max(behind) < current - 1):
            # Primeira linha de uma seção que já passou: a banda REPETIU a seção.
            target = self._harmonic_refinement(max(behind))
            self._position_estimator.relocate_to_bar(
                target, "lyrics section repeat", f"letra: {line.text}")
            self._after_external_relocation()
            self._lyrics_status = f"Letra: repetição de seção → Comp. {target}: {text}"
            return "repetição"
        if behind and max(behind) == current - 1:
            # Um compasso atrás é o atraso do reconhecimento: confirma.
            self._lyrics_status = f"Letra confirma: {text}"
            return "confirmada"
        self._lyrics_status = f"Letra ignorada (voltaria na cifra): {text}"
        return "ignorada"

    def _has_musical_activity(self, source_context: Optional[MusicalContext],
                               chord: str, chord_confidence: float,
                               tempo_result: Optional[TempoResult]) -> bool:
        if chord not in ("", "--", "UNKNOWN", "N") and chord_confidence >= 0.35:
            return True
        if tempo_result and tempo_result.beat_timestamp is not None and tempo_result.confidence >= 0.60:
            return True
        if source_context is None:
            return False
        return (source_context.audio_activity >= 0.010 or
                source_context.note_confidence >= 0.35 or
                source_context.chord_confidence >= 0.35)

    def _set_performance_state(self, state: PerformanceState) -> None:
        if state != self._performance_state:
            self._performance_state = state
            self._position_generation += 1

    def _near_chart_end(self) -> bool:
        return (self._last_confident_chart_pos.current_bar >=
                max(1, self._alignment.total_bars - 1))

    def _update_performance_state(self, timestamp: float, activity: bool,
                                  activity_observable: bool) -> bool:
        """Atualiza apenas a decisão de tocar; posição e relógio mantêm responsabilidades próprias."""
        if self._performance_state == PerformanceState.ENDED:
            return False
        # Chamadores de teste/importação podem fornecer somente tempo e cifra. Isso não é
        # evidência de silêncio do músico; preserva o modo temporal sem sensor de atividade.
        if not activity_observable:
            self._last_activity_time = timestamp
            if self._performance_state == PerformanceState.WAITING:
                self._set_performance_state(PerformanceState.PLAYING)
            return True
        if activity:
            if self._performance_state == PerformanceState.UNCERTAIN:
                # Uma respiração menor que a pausa real não exige esperar outro
                # downbeat: o músico nunca parou a execução.
                self._set_performance_state(PerformanceState.PLAYING)
            elif self._performance_state in (PerformanceState.WAITING,
                                             PerformanceState.HOLDING):
                self._set_performance_state(PerformanceState.RECOVERING)
                self._recovery_started_at = timestamp
                self._recovery_evidence = 0
            self._last_activity_time = timestamp
            if self._performance_state == PerformanceState.RECOVERING:
                self._recovery_evidence += 1
                elapsed = timestamp - (self._recovery_started_at or timestamp)
                at_downbeat = self._clock.beat == 1 and self._clock.beat_position <= 0.12
                if ((self._recovery_evidence >= 2 or elapsed >= self._RECOVERY_MIN_SECONDS)
                        and self._position_estimator.tracking_state != TrackingState.LOST
                        and at_downbeat):
                    self._set_performance_state(PerformanceState.PLAYING)
            elif self._performance_state == PerformanceState.PLAYING:
                pass
            return True

        if self._last_activity_time is None:
            return False
        silence = max(0.0, timestamp - self._last_activity_time)
        if self._performance_state == PerformanceState.PLAYING and silence >= 0.35:
            self._set_performance_state(PerformanceState.UNCERTAIN)
        if self._performance_state in (PerformanceState.PLAYING, PerformanceState.UNCERTAIN,
                                       PerformanceState.RECOVERING) and silence >= self._SHORT_PAUSE_SECONDS:
            self._set_performance_state(PerformanceState.HOLDING)
        if self._performance_state == PerformanceState.HOLDING and silence >= self._WAITING_SECONDS:
            self._set_performance_state(PerformanceState.WAITING)
        if (self._performance_state == PerformanceState.WAITING and
                silence >= self._END_SILENCE_SECONDS and self._near_chart_end()):
            self._set_performance_state(PerformanceState.ENDED)
        return self._performance_state not in (PerformanceState.HOLDING,
                                                PerformanceState.WAITING,
                                                PerformanceState.ENDED)

    def _publish_position(self, detected_chord: str = "--", detected_confidence: float = 0.0,
                          detected_key: str = "--", source_context: Optional[MusicalContext] = None,
                          update_structure: bool = True) -> FusedMusicalState:
        """Publica o mesmo snapshot na fusão, contexto, estrutura e predição."""
        position = self._current_chart_pos
        timestamp = position.absolute_time
        previous = self._context.chord
        previous_start = self._context.chord_start_time
        previous_time = self._context.timestamp
        if source_context is not None:
            for item in fields(MusicalContext):
                setattr(self._context, item.name, copy.deepcopy(getattr(source_context, item.name)))
        state = self._fusion.fuse(position, detected_chord, detected_confidence, timestamp)
        self._current_fused_state = state
        ctx = self._context
        ctx.timestamp = timestamp
        ctx.bpm = self._clock.bpm
        ctx.meter = self._clock.meter
        ctx.bar = position.current_bar
        ctx.beat = int(position.current_beat)
        ctx.beat_position = position.current_beat - ctx.beat
        ctx.is_beat = self._clock.is_beat
        ctx.initial_bpm = self._clock.initial_bpm
        ctx.target_bpm = self._clock.target_bpm
        ctx.tempo_confidence = self._clock.tempo_confidence
        ctx.phase_confidence = self._clock.phase_confidence
        ctx.phase_error_ms = self._clock.phase_error_ms
        ctx.tempo_tracking_state = self._clock.tracking_state
        ctx.clock_bar = self.clock_bar
        ctx.clock_beat = self.clock_beat
        ctx.bar_offset = self._position_estimator.bar_offset
        ctx.line_index = position.line_index
        ctx.tracking_state = position.tracking_state
        ctx.position_confidence = position.confidence
        ctx.chord = state.effective_chord
        ctx.next_expected_chord = position.next_chord
        ctx.next_change_bar = (position.current_bar + position.bars_until_chord_change
                               if position.next_chord != "--" and position.bars_until_chord_change > 0 else 0)
        ctx.next_change_beat = 1
        ctx.next_expected_section = position.next_section_name
        ctx.position_generation = self._position_generation
        ctx.chart_available = bool(self._chart.sections and position.current_chord != "--")
        # Esta é a única posição de cifra consumida pela UI e pelo baixista em
        # modo CHART. Não usar o acorde fundido aqui: ele pode representar uma
        # variação confirmada pela análise de áudio.
        ctx.chart_published_chord = position.current_chord if ctx.chart_available else "--"
        if self._alignment.event_count:
            # Sem prender antes de somar: no pré-início (relógio no compasso 0, antes do
            # tempo 1 da música) o próximo compasso é o 1 da cifra, não o 2.
            chart_bar = self.clock_bar - self._position_estimator.bar_offset
            ctx.chart_clock_chord = self._alignment.get_position_at(max(1, chart_bar)).current_chord
            ctx.chart_next_clock_chord = self._alignment.get_position_at(max(1, chart_bar + 1)).current_chord
        else:
            ctx.chart_clock_chord = "--"
            ctx.chart_next_clock_chord = "--"
        ctx.chart_alignment_confidence = self._chart_alignment_confidence if ctx.chart_available else 0.0
        ctx.performance_state = self.performance_state
        ctx.confirmed_variation_chord = (detected_chord if state.confirmed_variation
                                         and detected_confidence >= 0.85 else "--")
        ctx.chord_confidence = state.confidence
        rhythm = self._harmonic_rhythm.state
        ctx.current_chord_elapsed_beats = rhythm.current_elapsed_beats
        ctx.harmonic_event_chord = rhythm.current_chord
        ctx.harmonic_event_root = rhythm.current_root
        ctx.harmonic_event_start_beat = rhythm.current_start_beat
        ctx.harmonic_event_confidence = rhythm.event_confidence
        ctx.expected_chord_duration_beats = rhythm.expected_chord_duration_beats
        ctx.beats_until_chord_change = rhythm.beats_until_change
        ctx.duration_confidence = rhythm.duration_confidence
        ctx.pattern_confidence = rhythm.pattern_confidence
        ctx.harmonic_rhythm_pattern = rhythm.pattern_id
        ctx.harmonic_rhythm_observations = rhythm.observation_count
        ctx.rhythmic_next_chord = rhythm.next_chord
        if previous != ctx.chord or timestamp < previous_time:
            ctx.previous_chord = previous
            ctx.chord_start_time = timestamp
        else:
            ctx.chord_start_time = previous_start
        ctx.chord_duration = max(0.0, timestamp - ctx.chord_start_time)
        if ctx.chord not in ("", "--", "UNKNOWN", "N"):
            symbol = parse_chord(ctx.chord, self._chart.key)
            ctx.chord_root = symbol.root
            ctx.chord_quality = symbol.quality
            ctx.bass_note = symbol.bass_note or symbol.root
            ctx.inversion = "slash" if symbol.bass_note and symbol.bass_note != symbol.root else "root"
        ctx.current_section = state.current_section
        ctx.section_progress = position.section_progress
        ctx.current_section_type = position.section_type or "UNKNOWN"
        ctx.section_occurrence = position.section_occurrence
        ctx.bars_until_section_change = position.bars_until_section_change
        ctx.structure_confidence = position.confidence
        ctx.predicted_next_section = state.next_section
        chart_key = self._region_key(position.event_index)
        self._publish_key_changes(ctx, position)
        if (self._key_source() == "CHART" and self._alignment.event_count and
                chart_key not in ("", "--")):
            # Tom explícito da cifra é AUTORITATIVO no modo cifra (inclui modulações).
            ctx.key = chart_key
            ctx.key_confidence = 1.0
            ctx.previous_key = "--"
            ctx.key_start_time = 0.0
            ctx.key_duration = max(0.0, timestamp)
            ctx.key_candidate = "--"
            ctx.candidate_confidence = 0.0
            ctx.candidate_duration = 0.0
        elif detected_key and detected_key != "--":
            ctx.key = detected_key
        if update_structure:
            # Históricos DSP permanecem no AudioAnalyzer; estrutura aprende do contexto efetivo.
            self._structure_analyzer.update_online(
                ctx, None, None, self._clock,
                chart_position=position if self._chart.sections else None)
        self._update_follow_confidence(ctx, state, timestamp)
        ctx.sync_aliases()
        return state

    def _region_key(self, event_index: int) -> str:
        """Tom da cifra no evento (tom inicial ou de uma modulação), com o tom escolhido
        pelo usuário (override) aplicado como deslocamento de toda a música."""
        base = self._chart_key()
        if not self._chart.key_changes or not self._alignment.event_count:
            return base
        region = self._alignment.key_at_event(event_index)
        override = self._song.performance_settings.key_override
        if override and override != self._chart.key:
            from app.music.key_timeline import transpose_key
            from app.music.transposition import semitones_between_keys, key_uses_flats
            shift = semitones_between_keys(self._chart.key, override)
            return transpose_key(region, shift, key_uses_flats(override))
        return region

    def region_key_for_event(self, event_index: int) -> str:
        return self._region_key(event_index)

    @property
    def current_chart_key(self) -> str:
        """Tom vigente da cifra no ponto atual (considera as modulações)."""
        return self._region_key(self._current_chart_pos.event_index)

    def _publish_key_changes(self, ctx: MusicalContext, position: ChartPosition) -> None:
        """Próxima modulação da cifra: o baixista precisa saber ANTES que ela chegue."""
        ctx.next_expected_key = "--"
        ctx.beats_until_key_change = -1.0
        ctx.bars_until_key_change = -1
        ctx.key_change_pending = False
        ctx.key_change_chord = "--"
        ctx.key_region_index = 0
        ctx.key_change_confidence = 0.0
        if not self._chart.key_changes or not self._alignment.event_count:
            return
        event = max(0, position.event_index)
        keys = self._alignment.event_keys()
        ctx.key_region_index = sum(1 for i in range(1, event + 1) if keys[i] != keys[i - 1])
        upcoming = self._alignment.next_key_change(event)
        if upcoming is None:
            return
        target_event, _ = upcoming
        bpb = max(1, self._clock.beats_per_bar)
        bars = max(0, self._alignment.event_bar(target_event) - position.current_bar)
        ctx.next_expected_key = self._region_key(target_event)
        ctx.bars_until_key_change = bars
        ctx.beats_until_key_change = max(0.0, bars * bpb - (position.current_beat - 1.0))
        ctx.key_change_pending = True
        ctx.key_change_chord = self._alignment.event_chords()[target_event]
        ctx.key_change_confidence = 1.0

    def _update_follow_confidence(self, ctx: MusicalContext, state: FusedMusicalState,
                                  timestamp: float) -> None:
        """Publica uma decisão global sem substituir as métricas originais."""
        elapsed = max(0.0, timestamp - self._last_follow_timestamp) if self._last_follow_timestamp is not None else 0.0
        self._last_follow_timestamp = timestamp
        stable = (self._performance_state == PerformanceState.PLAYING and
                  ctx.tracking_state != TrackingState.LOST.value and
                  not state.is_discrepancy)
        target = 0.95 if stable else 0.30
        alpha = min(0.45, 0.12 + elapsed * 0.35)
        self._follow_stability += (target - self._follow_stability) * alpha
        ctx.recent_stability = max(0.0, min(1.0, self._follow_stability))
        # Transporte/cifra sem uma fonte observável é ensaio determinístico, não
        # evidência de acompanhamento ruim. Assim que chega áudio real, as
        # confianças medidas voltam a comandar a classificação.
        tempo = ctx.tempo_confidence
        phase = ctx.phase_confidence
        if not self._follow_observations_available:
            tempo = max(tempo, 0.70)
            phase = max(phase, 0.70)
        follow = calculate_follow_confidence(
            tempo=tempo, phase=phase,
            position=ctx.position_confidence, harmonic=ctx.chord_confidence,
            chart_alignment=ctx.chart_alignment_confidence,
            stability=ctx.recent_stability)
        ctx.confidence = follow.score
        ctx.follow_confidence_level = follow.level
        ctx.refresh_total_latency()

    def get_follow_diagnostics(self) -> Dict[str, Any]:
        ctx = self._context
        return {
            "follow_confidence": ctx.follow_confidence,
            "follow_level": ctx.follow_confidence_level,
            "tempo_confidence": ctx.tempo_confidence,
            "phase_confidence": ctx.phase_confidence,
            "position_confidence": ctx.position_confidence,
            "chord_confidence": ctx.chord_confidence,
            "capture_latency": ctx.capture_latency,
            "analysis_latency": ctx.analysis_latency,
            "scheduling_latency": ctx.scheduling_latency,
            "output_latency": ctx.output_latency,
            "total_latency": ctx.total_estimated_latency,
        }

    def format_follow_diagnostics(self) -> str:
        d = self.get_follow_diagnostics()
        return (f"FOLLOW CONFIDENCE: {d['follow_level']} ({d['follow_confidence']:.2f}) | "
                f"TEMPO CONFIDENCE: {d['tempo_confidence']:.2f} | "
                f"PHASE CONFIDENCE: {d['phase_confidence']:.2f} | "
                f"POSITION CONFIDENCE: {d['position_confidence']:.2f} | "
                f"CHORD CONFIDENCE: {d['chord_confidence']:.2f} | "
                f"CAPTURE LATENCY: {d['capture_latency']:.1f} ms | "
                f"ANALYSIS LATENCY: {d['analysis_latency']:.1f} ms | "
                f"SCHEDULER LATENCY: {d['scheduling_latency']:.1f} ms | "
                f"OUTPUT LATENCY: {d['output_latency']:.1f} ms | "
                f"TOTAL ESTIMATED LATENCY: {d['total_latency']:.1f} ms")

    def get_position_diagnostics(self) -> Dict[str, Any]:
        """Diagnóstico derivado; não mantém outra posição independente."""
        return {
            "clock_bar": self.clock_bar, "clock_beat": self.clock_beat,
            "current_bar": self.current_bar, "current_beat": self.current_beat,
            "line_index": self.chart_position.line_index,
            "section": self.chart_position.section_name, "tracking_state": self.tracking_state,
            "bar_offset": self._position_estimator.bar_offset,
            "event_index": self.chart_position.event_index,
            "section_event_index": self.chart_position.section_event_index,
            "section_occurrence": self.chart_position.section_occurrence,
            "event_state": self.chart_position.event_state,
            "next_event_state": self.chart_position.next_event_state,
            "section_state": self.chart_position.section_state,
            "next_event": self.chart_position.next_chord,
            "event_elapsed_beats": self._position_estimator.current_event_elapsed_beats,
            "expected_duration_beats": self._context.expected_chord_duration_beats,
            "last_position_change_reason": self._position_estimator.last_position_change_reason,
            "search_mode": self._position_estimator.last_search_mode,
            "note_evidence": self._position_estimator.last_note_evidence,
            "expected_chord": self.expected_chord, "detected_chord": self.detected_chord,
            "performance_state": self.performance_state,
        }

    def format_position_diagnostics(self) -> str:
        data = self.get_position_diagnostics()
        return (
            f"CLOCK: Bar {data['clock_bar']} Beat {data['clock_beat']} | "
            f"MUSICAL POSITION: Bar {data['current_bar']} Beat {data['current_beat']} | "
            f"CHART LINE: {data['line_index']} | SECTION: {data['section']} | "
            f"EVENT: {data['event_index']} ({data['event_state']}) → {data['next_event']} "
            f"({data['next_event_state']}) | "
            f"SECTION EVENT: {data['section_event_index']} / occurrence {data['section_occurrence']} "
            f"({data['section_state']}) | NEXT: {data['next_event']} | "
            f"EVENT ELAPSED: {data['event_elapsed_beats']:.2f} beats | "
            f"EXPECTED DURATION: {data['expected_duration_beats']:.2f} beats | "
            f"SEARCH MODE: {data['search_mode']} | "
            f"TRACKING: {data['tracking_state']} | BAR OFFSET: {data['bar_offset']:+d} | "
            f"EXPECTED: {data['expected_chord']} | DETECTED: {data['detected_chord']}"
            f" | PERFORMANCE: {data['performance_state']} | "
            f"LAST POSITION CHANGE: {data['last_position_change_reason']}"
        )

    def get_position_transition_log(self) -> List[Dict[str, Any]]:
        """Log de mudanças reais do cursor; beats, por si só, não entram aqui."""
        return self._position_estimator.position_transition_log

    def get_harmonic_input_diagnostic(self) -> Dict[str, Any]:
        """Última decisão da barreira entre conteúdo visual e motor harmônico."""
        return dict(self._last_harmonic_input_diagnostic)

    def get_tempo_diagnostics(self) -> Dict[str, Any]:
        return {
            "initial_bpm": self._clock.initial_bpm,
            "current_bpm": self._clock.current_bpm,
            "target_bpm": self._clock.target_bpm,
            "tempo_confidence": self._clock.tempo_confidence,
            "beat_phase": self._clock.beat_phase,
            "phase_error_ms": self._clock.phase_error_ms,
            "tracking_state": self._clock.tracking_state,
        }

    def format_tempo_diagnostics(self) -> str:
        d = self.get_tempo_diagnostics()
        return (f"INITIAL BPM: {d['initial_bpm']:.1f} | CURRENT BPM: {d['current_bpm']:.1f} | "
                f"TARGET BPM: {d['target_bpm']:.1f} | TEMPO CONF: {d['tempo_confidence']:.2f} | "
                f"BEAT PHASE: {d['beat_phase']:.2f} | PHASE ERROR: {d['phase_error_ms']:+.0f} ms | "
                f"STATE: {d['tracking_state']}")

    def format_harmonic_rhythm_diagnostics(self) -> str:
        """Snapshot para depurar cifra, detector estabilizado e duração em beats."""
        ctx = self._context
        return (f"SECTION: {ctx.current_section} | CURRENT: {ctx.chord} | "
                f"ELAPSED: {ctx.current_chord_elapsed_beats:.2f} beats | "
                f"EXPECTED DURATION: {ctx.expected_chord_duration_beats:.2f} beats | "
                f"BEATS UNTIL CHANGE: {ctx.beats_until_chord_change:.2f} | "
                f"NEXT: {ctx.next_expected_chord} | DETECTED: {ctx.smoothed_detected_chord} | "
                f"CHORD CONF: {ctx.chord_confidence:.2f} | DURATION CONF: {ctx.duration_confidence:.2f} | "
                f"POSITION CONF: {ctx.position_confidence:.2f} | PATTERN: {ctx.harmonic_rhythm_pattern} | "
                f"OBSERVATIONS: {ctx.harmonic_rhythm_observations}")

    def get_harmonic_rhythm_timeline(self) -> List[Dict[str, Any]]:
        """Log consultável de eventos fechados; não é uma segunda posição musical."""
        return [{
            "start_beat": round(event.start_beat, 2), "section": event.section_key,
            "detected": event.symbol, "raw_duration_beats": round(event.raw_duration_beats, 2),
            "duration_beats": event.quantized_duration_beats,
            "confidence": round(event.confidence, 2),
        } for event in self._harmonic_rhythm.timeline]

    # ============================================================
    # Navegação Manual (Ensaio / Rehearsal Mode)
    # ============================================================
    def transpose_to(self, target_key: str) -> ChordChart:
        """Transpõe a cifra ativa para ``target_key`` e atualiza tudo que dela depende.

        Reconstrói o alinhamento e o estimador, atualiza o contexto e persiste na canção
        (chart_data, chart_text e key), preservando a posição atual de reprodução.
        """
        self._chart.transpose_to_key(target_key)
        self._alignment = ChartAlignment(self._chart)
        self._position_estimator.set_alignment(self._alignment)
        self._context.key = target_key

        # Persiste na canção para que a mudança sobreviva à sessão
        self._song.chart_data = self._chart.to_dict()
        if self._song.chart_text:
            self._song.chart_text = self._chart.raw_text
        self._song.key = target_key

        # Reavalia a posição atual sob a nova cifra
        self._position_generation += 1
        self._current_chart_pos = self._position_estimator.refresh_position()
        self._publish_position()
        return self._chart

    def seek_to_bar(self, bar: int) -> ChartPosition:
        """Salta a reprodução/estudo diretamente para um compasso específico."""
        self._position_generation += 1
        self._current_chart_pos = self._position_estimator.seek_to_bar(bar)
        self._publish_position()
        return self._current_chart_pos

    def seek_to_start(self) -> ChartPosition:
        """Executa Play From Start usando o primeiro evento musical da cifra."""
        return self.restart()

    def next_bar(self) -> ChartPosition:
        return self.seek_to_bar(self.current_bar + 1)

    def prev_bar(self) -> ChartPosition:
        return self.seek_to_bar(max(1, self.current_bar - 1))

    def next_section_jump(self) -> ChartPosition:
        """Avança para o primeiro compasso da próxima seção da cifra."""
        curr_sec_name = self._current_chart_pos.section_name
        for bar in range(self.current_bar + 1, self._alignment.total_bars + 1):
            pos = self._alignment.get_position_at(bar)
            if pos.section_name != curr_sec_name:
                return self.seek_to_bar(bar)
        return self._current_chart_pos

    def prev_section(self) -> ChartPosition:
        """Retrocede para o início da seção atual ou da seção anterior."""
        curr_sec_name = self._current_chart_pos.section_name
        # Primeiro tenta achar o início da seção atual
        first_bar_of_curr = self.current_bar
        for bar in range(self.current_bar - 1, 0, -1):
            pos = self._alignment.get_position_at(bar)
            if pos.section_name == curr_sec_name:
                first_bar_of_curr = bar
            else:
                break
        
        if self.current_bar > first_bar_of_curr:
            return self.seek_to_bar(first_bar_of_curr)

        # Se já estava no início da atual, vai para o início da anterior
        for bar in range(first_bar_of_curr - 1, 0, -1):
            pos = self._alignment.get_position_at(bar)
            if pos.section_name != curr_sec_name:
                # Acha o primeiro compasso daquela seção anterior
                prev_sec_name = pos.section_name
                start_prev = bar
                for b in range(bar - 1, 0, -1):
                    if self._alignment.get_position_at(b).section_name == prev_sec_name:
                        start_prev = b
                    else:
                        break
                return self.seek_to_bar(start_prev)
        return self.seek_to_bar(1)

    def next_chord_jump(self) -> ChartPosition:
        """Salta para o próximo acorde na progressão da cifra."""
        curr_chord = self._current_chart_pos.current_chord
        for bar in range(self.current_bar + 1, self._alignment.total_bars + 1):
            pos = self._alignment.get_position_at(bar)
            if pos.current_chord != curr_chord:
                return self.seek_to_bar(bar)
        return self._current_chart_pos

    def prev_chord_jump(self) -> ChartPosition:
        """Retrocede para o acorde imediatamente anterior."""
        curr_chord = self._current_chart_pos.current_chord
        for bar in range(self.current_bar - 1, 0, -1):
            pos = self._alignment.get_position_at(bar)
            if pos.current_chord != curr_chord:
                return self.seek_to_bar(bar)
        return self.seek_to_bar(1)

    def update_chart_text(self, new_chart_text: str) -> None:
        """Atualiza a cifra em tempo de execução, re-parseando e sincronizando o alinhamento."""
        previous_bar = self.current_bar
        preserve_runtime_position = self._is_playing or self._clock.elapsed_time > 0.0
        self._song.chart_text = new_chart_text
        parsed = ChartParser.parse(
            text=new_chart_text,
            default_key=self._song.key,
            default_bpm=self._song.bpm,
            default_meter=self._song.meter,
            title=self._song.title,
            artist=self._song.artist
        )
        self._song.chart_data = parsed.to_dict()
        self._song.lyrics_text = "\n".join(
            l.get("text", "") if isinstance(l, dict) else getattr(l, "text", str(l))
            for s in self._song.chart_data.get("sections", []) for l in s.get("lyrics", [])
        )
        self._song.update_timestamp()
        self._chart = parsed
        self._song.key = parsed.key
        self._song.bpm = parsed.bpm
        self._song.meter = parsed.meter
        self._song.time_signature = parsed.meter
        if (self._bpm_source() == "CHART" or self._song.performance_settings.bpm_override or
                not self._follow_observations_available):
            self._clock.bpm = self._chart_bpm()
        self._clock.set_meter(self._song.performance_settings.meter_override or parsed.meter)
        self._context.bpm = self._clock.bpm
        self._context.meter = self._clock.meter
        self._context.key = self._chart_key() if self._key_source() == "CHART" else "--"
        self._alignment = ChartAlignment(self._chart)
        self._position_estimator.set_alignment(self._alignment)
        self._position_estimator.set_capo(self._chart.capo_semitones)
        self._position_generation += 1
        if preserve_runtime_position:
            self._current_chart_pos = self._position_estimator.seek_to_bar(
                min(previous_bar, self._alignment.total_bars))
        else:
            self._position_estimator.reset()
            self._current_chart_pos = self._position_estimator.refresh_position()
        self._publish_position()
