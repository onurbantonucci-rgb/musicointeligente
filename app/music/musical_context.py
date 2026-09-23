"""Representação do Contexto Musical Compartilhado do Virtual Band AI (v0.1-Completo).

Este objeto atua como o barramento de dados central do sistema.
O 'ouvido' (módulo de análise + context manager) escreve neste contexto e todos
os futuros instrumentos virtuais (baixista, baterista, etc.) o consumirão
diretamente, sem precisar analisar o sinal de áudio bruto.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class MusicalContext:
    """Estado musical instantâneo e temporal do sistema em um determinado instante."""

    # ============================================================
    # 1. Posição Temporal e Amostragem
    # ============================================================
    timestamp: float = 0.0                      # Posição atual do áudio em segundos
    sample_rate: int = 44100                    # Taxa de amostragem da fonte de áudio

    # ============================================================
    # 2. Análise Monofônica / Nota Dominante (Pitch)
    # ============================================================
    note: str = "--"                            # Nome da nota com oitava (ex: "C3", "G2")
    current_note: str = "--"                    # Alias de compatibilidade
    frequency: float = 0.0                      # Frequência fundamental f0 em Hz (ex: 130.81)
    current_frequency: float = 0.0              # Alias de compatibilidade
    note_confidence: float = 0.0                # Grau de certeza do pitch (0.0 a 1.0)
    cents_deviation: float = 0.0                # Desvio de afinação em cents (-50 a +50)
    raw_pitch_hz: float = 0.0                   # f0 monofônico sem interpretação harmônica
    raw_note: str = "--"
    pitch_confidence: float = 0.0
    chroma_vector: List[float] = field(default_factory=lambda: [0.0] * 12)  # Energia das 12 classes C..B
    raw_chroma_vector: List[float] = field(default_factory=lambda: [0.0] * 12)
    active_notes: Dict[str, float] = field(default_factory=dict)
    audio_activity: float = 0.0                # RMS do bloco de referência, para estado de performance

    # ============================================================
    # 3. Análise Harmônica e Acorde Atual
    # ============================================================
    chord: str = "--"                           # Cifra completa (ex: "C", "G/B", "Am")
    next_expected_chord: str = "--"             # Acorde previsto para o próximo compasso
    next_change_bar: int = 0                   # Próximo compasso de mudança, 0 se desconhecido
    next_change_beat: int = 1
    next_expected_section: str = "--"
    confirmed_variation_chord: str = "--"
    position_generation: int = 0               # Incrementado quando a projeção da cifra muda
    chart_available: bool = False
    chart_clock_chord: str = "--"               # Cifra no compasso do relógio, após reancoragem
    chart_next_clock_chord: str = "--"          # Cifra no próximo compasso do relógio
    performance_state: str = "PLAYING"         # Estado da banda, separado de tracking/clock
    current_chord: str = "--"                   # Alias de compatibilidade
    chord_root: str = "--"                      # Tônica do acorde (ex: "C")
    chord_quality: str = "--"                   # Qualidade (ex: "major", "minor")
    bass_note: str = "--"                       # Nota mais grave detectada (ex: "B" para "G/B")
    inversion: str = "root"                     # "root", "first", "second"
    detected_notes: List[str] = field(default_factory=list)  # Ex: ["C", "E", "G"]
    chord_confidence: float = 0.0               # Confiança do acorde (0.0 a 1.0)
    raw_detected_chord: str = "--"             # Saída do classificador antes da estabilização
    raw_chord_confidence: float = 0.0
    smoothed_detected_chord: str = "--"        # Saída do ChordStabilizer
    stable_chord_confidence: float = 0.0
    stable_chord_duration: float = 0.0
    stable_chord_root: str = "--"
    stable_chord_quality: str = "--"
    chord_candidate: str = "--"
    chord_candidate_root: str = "--"
    chord_candidate_confidence: float = 0.0
    chord_candidate_age_ms: float = 0.0
    chord_candidate_frames: int = 0
    current_chord_support_age_ms: float = 0.0
    stable_chord_stale: bool = False
    harmonic_event_chord: str = "--"
    harmonic_event_root: str = "--"
    harmonic_event_start_beat: float = 0.0
    harmonic_event_confidence: float = 0.0
    expected_chart_chord: str = "--"
    chart_prior_enabled: bool = False

    # Contexto Temporal do Acorde
    previous_chord: str = "--"                  # Acorde imediatamente anterior confirmado
    chord_start_time: float = 0.0               # Momento exato em que o acorde atual começou
    chord_duration: float = 0.0                 # Tempo decorrido no acorde atual em segundos

    # ============================================================
    # 4. Tonalidade (Key) e Modulação
    # ============================================================
    key: str = "--"                             # Tonalidade estimada confirmada (ex: "C Major", "A Minor")
    current_key: str = "--"                     # Alias de compatibilidade
    key_confidence: float = 0.0                 # Grau de correlação estatística consolidado (0.0 a 1.0)
    previous_key: str = "--"                    # Tonalidade confirmada anterior
    key_start_time: float = 0.0                 # Momento em que a tonalidade atual se consolidou
    key_duration: float = 0.0                   # Tempo contínuo na tonalidade atual em segundos
    key_candidate: str = "--"                   # Tonalidade candidata sob observação (ex: "E Minor")
    candidate_confidence: float = 0.0           # Confiança da candidata (0.0 a 1.0)
    candidate_duration: float = 0.0             # Tempo contínuo em que a candidata lidera (segundos)
    local_tonal_center: str = "--"              # Centro tonal local (raiz harmônica imediata)


    # ============================================================
    # 5. Relógio Musical (Musical Clock) e Métrica Rítmica
    # ============================================================
    bpm: float = 0.0                            # Andamento estimado em batidas por minuto
    beat: int = 1                               # Número do tempo no compasso (1, 2, 3, 4...)
    beat_position: float = 0.0                  # Posição fracionária dentro do tempo [0.0 a 1.0)
    bar: int = 1                                # Número do compasso (1, 2, 3... 12...)
    meter: str = "4/4"                          # Fórmula de compasso ativa (ex: "4/4", "3/4")
    time_signature: str = "4/4"                 # Alias de compatibilidade
    is_beat: bool = False                       # Pulso no instante atual do clique
    initial_bpm: float = 120.0
    target_bpm: float = 120.0
    tempo_confidence: float = 0.0
    phase_confidence: float = 0.0
    phase_error_ms: float = 0.0
    tempo_tracking_state: str = "UNINITIALIZED"

    # ============================================================
    # 6. Confiança Global
    # ============================================================
    confidence: float = 0.0                     # Confiança geral consolidada do sistema
    follow_confidence_level: str = "MEDIUM"    # HIGH / MEDIUM / LOW; score é ``confidence``
    chart_alignment_confidence: float = 0.0     # Confiabilidade conhecida da cifra/alinhamento
    recent_stability: float = 0.0               # Estabilidade temporal das evidências recentes

    # ============================================================
    # 7. Métricas de Latência Quadripartida (Sem valores fictícios)
    # ============================================================
    processing_latency: float = 0.0             # Tempo de cálculo DSP do bloco em ms
    analysis_window: float = 0.0                # Duração do bloco de áudio analisado em ms
    analysis_latency: float = 0.0               # Centro temporal efetivo da janela em ms
    stabilization_delay: float = 0.0            # Atraso configurado para confirmação temporal em ms
    estimated_musical_latency: float = 0.0      # Latência perceptível total estimada em ms
    estimated_latency: float = 0.0              # Alias de compatibilidade
    capture_latency: float = 0.0                # Buffer de captura conhecido em ms
    decision_latency: float = 0.0               # Tempo medido de decisão do instrumento em ms
    scheduling_latency: float = 0.0             # Tempo medido ao inserir na fila em ms
    output_latency: float = 0.0                 # Buffer de saída conhecido em ms
    total_estimated_latency: float = 0.0        # Soma estimada das parcelas conhecidas

    # ============================================================
    # 8. Estrutura Musical e Antecipação Preditiva (Fase v0.3)
    # ============================================================
    current_section: str = "UNKNOWN"            # Seção ativa estimada (ex: "VERSE", "CHORUS")
    current_pattern: str = "--"                 # ID do padrão ativo (ex: "P01")
    section_progress: float = 0.0               # Progresso relativo na seção ativa [0.0 a 1.0]
    structure_confidence: float = 0.0           # Confiança da inferência estrutural
    predicted_next_section: str = "UNKNOWN"     # Próxima seção prevista por repetição
    predicted_next_chords: List[str] = field(default_factory=list) # Próximos acordes (Look-ahead)
    bars_until_change: int = 0                  # Quantidade de compassos até a mudança
    beats_until_change: int = 0                 # Quantidade de tempos até a mudança
    prediction_confidence: float = 0.0          # Confiança da predição
    prediction_reason: str = "--"               # Justificativa probabilística da predição
    current_chord_elapsed_beats: float = 0.0     # Duração observada no relógio musical
    expected_chord_duration_beats: float = 0.0   # Perfil temporal aprendido; 0 se ainda desconhecido
    beats_until_chord_change: float = 0.0
    duration_confidence: float = 0.0
    pattern_confidence: float = 0.0
    harmonic_rhythm_pattern: str = "--"
    harmonic_rhythm_observations: int = 0
    rhythmic_next_chord: str = "--"

    # Diagnóstico temporal; bar/beat são sempre as coordenadas musicais publicadas.
    clock_bar: int = 1
    clock_beat: int = 1
    bar_offset: int = 0
    line_index: int = 1
    tracking_state: str = "TRACKING"
    position_confidence: float = 0.95

    @property
    def current_bar(self) -> int:
        return self.bar

    @property
    def current_beat(self) -> int:
        return self.beat

    def sync_aliases(self) -> None:
        """Garante que os aliases estejam sempre estritamente sincronizados."""
        self.current_note = self.note
        self.current_frequency = self.frequency
        self.current_chord = self.chord
        self.current_key = self.key
        self.time_signature = self.meter
        self.estimated_latency = self.estimated_musical_latency

    @property
    def follow_confidence(self) -> float:
        """Alias sem duplicar a medida global consolidada."""
        return self.confidence

    def refresh_total_latency(self) -> None:
        """Recalcula a soma sem modificar o timestamp ou a posição musical."""
        self.total_estimated_latency = max(0.0, self.capture_latency) + max(0.0, self.analysis_latency) + \
            max(0.0, self.processing_latency) + max(0.0, self.stabilization_delay) + \
            max(0.0, self.decision_latency) + max(0.0, self.scheduling_latency) + \
            max(0.0, self.output_latency)
        self.estimated_musical_latency = self.total_estimated_latency
        self.estimated_latency = self.total_estimated_latency

    def reset(self) -> None:
        """Reinicia o contexto para o estado inicial neutro."""
        self.timestamp = 0.0
        self.note = "--"
        self.current_note = "--"
        self.frequency = 0.0
        self.current_frequency = 0.0
        self.note_confidence = 0.0
        self.cents_deviation = 0.0
        self.raw_pitch_hz = 0.0
        self.raw_note = "--"
        self.pitch_confidence = 0.0
        self.chroma_vector = [0.0] * 12
        self.raw_chroma_vector = [0.0] * 12
        self.active_notes.clear()
        self.audio_activity = 0.0

        self.chord = "--"
        self.next_expected_chord = "--"
        self.next_change_bar = 0
        self.next_change_beat = 1
        self.next_expected_section = "--"
        self.confirmed_variation_chord = "--"
        self.position_generation = 0
        self.chart_available = False
        self.chart_clock_chord = "--"
        self.chart_next_clock_chord = "--"
        self.performance_state = "WAITING"
        self.current_chord = "--"
        self.chord_root = "--"
        self.chord_quality = "--"
        self.bass_note = "--"
        self.inversion = "root"
        self.detected_notes.clear()
        self.chord_confidence = 0.0
        self.raw_detected_chord = "--"
        self.raw_chord_confidence = 0.0
        self.smoothed_detected_chord = "--"
        self.stable_chord_confidence = 0.0
        self.stable_chord_duration = 0.0
        self.stable_chord_root = "--"
        self.stable_chord_quality = "--"
        self.chord_candidate = "--"
        self.chord_candidate_root = "--"
        self.chord_candidate_confidence = 0.0
        self.chord_candidate_age_ms = 0.0
        self.chord_candidate_frames = 0
        self.current_chord_support_age_ms = 0.0
        self.stable_chord_stale = False
        self.harmonic_event_chord = "--"
        self.harmonic_event_root = "--"
        self.harmonic_event_start_beat = 0.0
        self.harmonic_event_confidence = 0.0
        self.expected_chart_chord = "--"
        self.chart_prior_enabled = False

        self.previous_chord = "--"
        self.chord_start_time = 0.0
        self.chord_duration = 0.0

        self.key = "--"
        self.current_key = "--"
        self.key_confidence = 0.0
        self.previous_key = "--"
        self.key_start_time = 0.0
        self.key_duration = 0.0
        self.key_candidate = "--"
        self.candidate_confidence = 0.0
        self.candidate_duration = 0.0
        self.local_tonal_center = "--"

        self.bpm = 0.0
        self.beat = 1
        self.beat_position = 0.0
        self.bar = 1
        self.clock_bar = 1
        self.clock_beat = 1
        self.bar_offset = 0
        self.line_index = 1
        self.tracking_state = "TRACKING"
        self.position_confidence = 0.95
        self.meter = "4/4"
        self.time_signature = "4/4"
        self.is_beat = False

        self.confidence = 0.0
        self.follow_confidence_level = "MEDIUM"
        self.chart_alignment_confidence = 0.0
        self.recent_stability = 0.0
        self.processing_latency = 0.0
        self.analysis_window = 0.0
        self.analysis_latency = 0.0
        self.stabilization_delay = 0.0
        self.estimated_musical_latency = 0.0
        self.estimated_latency = 0.0
        self.capture_latency = 0.0
        self.decision_latency = 0.0
        self.scheduling_latency = 0.0
        self.output_latency = 0.0
        self.total_estimated_latency = 0.0

        # Reset Estrutura e Predição (v0.3)
        self.current_section = "UNKNOWN"
        self.current_pattern = "--"
        self.section_progress = 0.0
        self.structure_confidence = 0.0
        self.predicted_next_section = "UNKNOWN"
        self.predicted_next_chords.clear()
        self.bars_until_change = 0
        self.beats_until_change = 0
        self.prediction_confidence = 0.0
        self.prediction_reason = "--"
        self.current_chord_elapsed_beats = 0.0
        self.expected_chord_duration_beats = 0.0
        self.beats_until_chord_change = 0.0
        self.duration_confidence = 0.0
        self.pattern_confidence = 0.0
        self.harmonic_rhythm_pattern = "--"
        self.harmonic_rhythm_observations = 0
        self.rhythmic_next_chord = "--"

    def get_summary_dict(self) -> Dict:
        """Retorna uma visão estruturada pronta para consumo por instrumentos virtuais."""
        return {
            "timestamp": round(self.timestamp, 3),
            "note": self.note,
            "frequency_hz": round(self.frequency, 2),
            "chord": self.chord,
            "harmonic_detection": {
                "raw_chord": self.raw_detected_chord,
                "raw_confidence": round(self.raw_chord_confidence, 2),
                "stable_chord": self.smoothed_detected_chord,
                "stable_confidence": round(self.stable_chord_confidence, 2),
                "stable_root": self.stable_chord_root,
                "stable_quality": self.stable_chord_quality,
                "candidate": self.chord_candidate,
                "candidate_root": self.chord_candidate_root,
                "candidate_confidence": round(self.chord_candidate_confidence, 2),
                "candidate_age_ms": round(self.chord_candidate_age_ms, 1),
                "candidate_frames": self.chord_candidate_frames,
                "support_age_ms": round(self.current_chord_support_age_ms, 1),
                "stale": self.stable_chord_stale,
                "expected_chart_chord": self.expected_chart_chord,
                "chart_clock_chord": self.chart_clock_chord,
                "chart_next_clock_chord": self.chart_next_clock_chord,
                "chart_prior_enabled": self.chart_prior_enabled,
            },
            "previous_chord": self.previous_chord,
            "chord_duration_s": round(self.chord_duration, 2),
            "key": self.key,
            "previous_key": self.previous_key,
            "key_duration_s": round(self.key_duration, 2),
            "key_candidate": self.key_candidate,
            "candidate_confidence": round(self.candidate_confidence, 2),
            "local_tonal_center": self.local_tonal_center,
            "bpm": round(self.bpm, 1),
            "meter": self.meter,
            "bar": self.bar,
            "beat": self.beat,
            "position": {
                "clock_bar": self.clock_bar, "clock_beat": self.clock_beat,
                "bar_offset": self.bar_offset, "line_index": self.line_index,
                "tracking_state": self.tracking_state,
                "confidence": self.position_confidence,
            },
            "confidence": round(self.confidence, 2),
            "follow_confidence": {"level": self.follow_confidence_level,
                                  "stability": round(self.recent_stability, 2)},
            "latency_ms": {
                "capture": round(self.capture_latency, 2),
                "analysis": round(self.analysis_latency, 2),
                "processing": round(self.processing_latency, 2),
                "decision": round(self.decision_latency, 2),
                "scheduling": round(self.scheduling_latency, 2),
                "output": round(self.output_latency, 2),
                "window": round(self.analysis_window, 2),
                "stabilization": round(self.stabilization_delay, 2),
                "estimated_total": round(self.total_estimated_latency, 2),
            },
            "structure": {
                "section": self.current_section,
                "pattern": self.current_pattern,
                "progress": round(self.section_progress, 3),
                "confidence": round(self.structure_confidence, 2),
            },
            "prediction": {
                "next_section": self.predicted_next_section,
                "next_chords": list(self.predicted_next_chords),
                "bars_until_change": self.bars_until_change,
                "beats_until_change": self.beats_until_change,
                "confidence": round(self.prediction_confidence, 2),
                "reason": self.prediction_reason,
            }
        }

