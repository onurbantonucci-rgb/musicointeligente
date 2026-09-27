"""Histórico Temporal e Estabilizador de Acordes (v0.1-C e Contexto Temporal).

Responsável por:
1. Filtrar oscilações e ruídos transitórios do detector de acordes (Debounce / Histerese).
2. Registrar eventos discretos de acordes confirmados (ChordEvent) com start_time, end_time, duration e confidence.
3. Não criar eventos repetidos ou centenas de entradas para um mesmo acorde mantido.
4. Fornecer informações de acorde anterior (previous_chord) e duração do acorde atual (chord_duration).
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import collections
import numpy as np

from app.music.constants import (
    MIN_CHORD_CONFIDENCE,
    MIN_CHORD_STABILITY_TIME,
    CHORD_CHANGE_CONFIRMATION_TIME,
    MAX_CHORD_HISTORY_ENTRIES,
)
from app.music.theory import PITCH_CLASSES, note_to_pc


@dataclass
class ChordEvent:
    """Evento consolidado que representa um acorde que foi tocado com duração definida."""
    chord: str                          # Cifra completa (ex: "C", "G/B", "Am")
    start_time: float                   # Início exato do acorde em segundos
    end_time: float                     # Fim do acorde em segundos
    duration: float                     # Duração total em segundos (end_time - start_time)
    confidence: float                   # Grau médio de confiança (0.0 a 1.0)
    root: str = "--"                    # Fundamental (ex: "C")
    quality: str = "--"                 # Qualidade (ex: "major", "minor")
    bass_note: str = "--"               # Nota no baixo (ex: "B" para "G/B")
    inversion: str = "root"             # Tipo de inversão ("root", "first", "second")

    @property
    def symbol(self) -> str:
        """Alias para o símbolo da cifra."""
        return self.chord

    def to_dict(self) -> Dict:
        return {
            "time_str": self.format_timestamp(self.start_time),
            "chord": self.chord,
            "duration_str": f"{self.duration:.2f} s",
            "confidence_str": f"{int(self.confidence * 100)}%",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "confidence": self.confidence,
        }

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        if seconds < 0 or np.isnan(seconds):
            return "00:00.00"
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m:02d}:{s:05.2f}"


class ChordHistory:
    """Repositório ordenado de eventos discretos de acordes confirmados.
    
    Por padrão armazena o histórico completo da faixa sem descarte (max_entries=None).
    """

    def __init__(self, max_entries: Optional[int] = MAX_CHORD_HISTORY_ENTRIES):
        self._max_entries = max_entries
        self._events: collections.deque = collections.deque(maxlen=max_entries)
        self._version: int = 0

    @property
    def version(self) -> int:
        """Número de versão incrementado a cada novo evento confirmado."""
        return self._version

    def add_event(self, event: ChordEvent) -> None:
        """Adiciona um evento de acorde confirmado ao histórico."""
        if event.chord == "--" or event.duration < 0.05:
            return
        self._events.append(event)
        self._version += 1

    def record(self, timestamp: float, chord: str, confidence: float = 0.8) -> None:
        """Adiciona evento simples por timestamp e símbolo de acorde."""
        if chord == "--":
            return
        self._events.append(ChordEvent(
            chord=chord,
            start_time=timestamp,
            end_time=timestamp + 0.5,
            duration=0.5,
            confidence=confidence
        ))
        self._version += 1


    def update(self, chord) -> None:
        """Método de conveniência para compatibilidade com a interface anterior."""
        if not hasattr(chord, "symbol") or chord.symbol == "--":
            return
        # Se o último evento for o mesmo acorde, incrementa duração
        if self._events and self._events[-1].chord == chord.symbol:
            dt = max(0.0, getattr(chord, "timestamp", 0.0) - self._events[-1].start_time)
            self._events[-1].duration = dt
            self._events[-1].end_time = getattr(chord, "timestamp", 0.0)
        else:
            ts = getattr(chord, "timestamp", 0.0)
            self._events.append(ChordEvent(
                chord=chord.symbol,
                start_time=ts,
                end_time=ts,
                duration=0.05,
                confidence=getattr(chord, "confidence", 0.8),
                root=getattr(chord, "root", "--"),
                quality=getattr(chord, "quality", "--"),
                bass_note=getattr(chord, "bass_note", "--"),
                inversion=getattr(chord, "inversion", "root")
            ))

    def get_events(self) -> List[ChordEvent]:
        """Retorna todos os eventos registrados."""
        return list(self._events)

    def get_recent(self, count: int = 8) -> List[ChordEvent]:
        """Retorna os N eventos mais recentes."""
        return list(self._events)[-count:]

    def get_summary_text(self) -> str:
        """Formata resumo da progressão harmônica recente para exibição na UI."""
        if not self._events:
            return "Aguardando progressão de acordes..."
        items = [f"{e.chord} ({e.duration:.1f}s)" for e in list(self._events)[-5:]]
        return " → ".join(items)

    def clear(self) -> None:
        """Limpa todo o histórico."""
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)


class ChordStabilizer:
    """Motor de estabilização temporal para detecção de acordes.
    
    Equilibra estabilidade (rejeição de falsos positivos transitórios)
    e velocidade de resposta musical.
    """

    def __init__(
        self,
        min_confidence: float = MIN_CHORD_CONFIDENCE,
        min_stability_time: float = MIN_CHORD_STABILITY_TIME,
        confirmation_time: float = CHORD_CHANGE_CONFIRMATION_TIME,
        history: Optional[ChordHistory] = None
    ):
        self._min_confidence = min_confidence
        self._min_stability_time = min_stability_time
        self._confirmation_time = confirmation_time
        self._history = history if history is not None else ChordHistory()

        # Estado atual confirmado
        self._current_chord: str = "--"
        self._chord_start_time: float = 0.0
        self._chord_duration: float = 0.0
        self._chord_root: str = "--"
        self._chord_quality: str = "--"
        self._bass_note: str = "--"
        self._inversion: str = "root"
        self._detected_notes: List[str] = []
        self._chord_confidence: float = 0.0

        # Acorde anterior confirmado
        self._previous_chord: str = "--"

        # Acumulador de confiança do acorde vigente
        self._confidences: List[float] = []

        # Estado do acorde candidato (para debounce / confirmação de transição)
        self._candidate_chord: Optional[str] = None
        self._candidate_start_time: float = 0.0
        self._candidate_raw_event: Optional[Dict] = None
        self._candidate_confidences: List[float] = []
        self._candidate_frame_count: int = 0
        self._candidate_root: str = "--"
        self._candidate_confidence: float = 0.0
        self._candidate_duration: float = 0.0

        # Evidência recente. O acorde atual precisa continuar sendo sustentado
        # pelo áudio; a ausência de um candidato perfeito não o torna imortal.
        self._observation_window: collections.deque = collections.deque(maxlen=32)
        self._support_window_seconds = max(0.60, self._confirmation_time * 5.0)
        self._stale_active_seconds = max(0.75, self._confirmation_time * 6.0)
        self._last_current_support_time: float = -1.0
        self._last_supported_confidence: float = 0.0
        self._unsupported_active_seconds: float = 0.0
        self._is_current_chord_stale: bool = False
        self._awaiting_confirmation: bool = False

        # Rastreamento de saltos no tempo (Seek)
        self._last_timestamp: float = -1.0

    @property
    def history(self) -> ChordHistory:
        return self._history

    @property
    def current_chord(self) -> str:
        return self._current_chord

    @property
    def previous_chord(self) -> str:
        return self._previous_chord

    @property
    def chord_start_time(self) -> float:
        return self._chord_start_time

    @property
    def chord_duration(self) -> float:
        return self._chord_duration

    @property
    def chord_confidence(self) -> float:
        return self._chord_confidence

    @property
    def chord_root(self) -> str:
        return self._chord_root

    @property
    def chord_quality(self) -> str:
        return self._chord_quality

    @property
    def bass_note(self) -> str:
        return self._bass_note

    @property
    def inversion(self) -> str:
        return self._inversion

    @property
    def detected_notes(self) -> List[str]:
        return self._detected_notes

    @property
    def candidate_chord(self) -> str:
        return self._candidate_chord or "--"

    @property
    def candidate_root(self) -> str:
        return self._candidate_root

    @property
    def candidate_confidence(self) -> float:
        return self._candidate_confidence

    @property
    def candidate_duration(self) -> float:
        return self._candidate_duration

    @property
    def candidate_frame_count(self) -> int:
        return self._candidate_frame_count

    @property
    def time_since_current_chord_support(self) -> float:
        if self._last_current_support_time < 0.0 or self._last_timestamp < 0.0:
            return 0.0
        return max(0.0, self._last_timestamp - self._last_current_support_time)

    @property
    def is_current_chord_stale(self) -> bool:
        return self._is_current_chord_stale

    @property
    def stabilization_delay_ms(self) -> float:
        """Tempo de atraso de estabilização configurado em milissegundos."""
        return self._confirmation_time * 1000.0

    def reset(self) -> None:
        """Reinicia o estabilizador e limpa o histórico."""
        self._finalize_current_event_if_active()
        self._current_chord = "--"
        self._chord_start_time = 0.0
        self._chord_duration = 0.0
        self._chord_root = "--"
        self._chord_quality = "--"
        self._bass_note = "--"
        self._inversion = "root"
        self._detected_notes = []
        self._chord_confidence = 0.0
        self._previous_chord = "--"
        self._confidences.clear()
        self._candidate_chord = None
        self._candidate_start_time = 0.0
        self._candidate_raw_event = None
        self._candidate_confidences.clear()
        self._candidate_frame_count = 0
        self._candidate_root = "--"
        self._candidate_confidence = 0.0
        self._candidate_duration = 0.0
        self._observation_window.clear()
        self._last_current_support_time = -1.0
        self._last_supported_confidence = 0.0
        self._unsupported_active_seconds = 0.0
        self._is_current_chord_stale = False
        self._awaiting_confirmation = False
        self._last_timestamp = -1.0
        self._history.clear()

    def _finalize_current_event_if_active(self, end_time: Optional[float] = None) -> None:
        """Encerra o acorde atual e o adiciona ao histórico se tiver duração mínima."""
        if self._current_chord != "--" and self._confidences:
            actual_end = end_time if end_time is not None else (self._chord_start_time + self._chord_duration)
            dur = max(0.01, actual_end - self._chord_start_time)
            if dur >= self._min_stability_time:
                avg_conf = float(np.mean(self._confidences))
                event = ChordEvent(
                    chord=self._current_chord,
                    start_time=self._chord_start_time,
                    end_time=actual_end,
                    duration=dur,
                    confidence=avg_conf,
                    root=self._chord_root,
                    quality=self._chord_quality,
                    bass_note=self._bass_note,
                    inversion=self._inversion
                )
                self._history.add_event(event)

    @staticmethod
    def _normalize_root(root: str, symbol: str = "--") -> str:
        candidate = root
        if not candidate or candidate == "--":
            base = (symbol or "--").split("/")[0]
            candidate = base[:2] if len(base) > 1 and base[1] in ("#", "b") else base[:1]
        pc = note_to_pc(candidate)
        return PITCH_CLASSES[pc] if pc >= 0 else "--"

    @staticmethod
    def _quality_family(quality: str, symbol: str) -> str:
        low = (quality or "").lower()
        if "minor" in low:
            return "minor"
        base = (symbol or "").split("/")[0]
        body = base[2:] if len(base) > 1 and base[1] in ("#", "b") else base[1:]
        return "minor" if body.lower().startswith("m") and not body.lower().startswith("maj") else "major"

    def _chroma_support(self, root: str, quality: str,
                        chroma_vector: Optional[List[float]]) -> float:
        if chroma_vector is None or len(chroma_vector) < 12:
            return 0.5
        values = np.asarray(chroma_vector, dtype=np.float64)
        peak = float(np.max(values))
        root_pc = note_to_pc(root)
        if peak <= 1e-9 or root_pc < 0:
            return 0.0
        intervals = (0, 3, 7) if quality == "minor" else (0, 4, 7)
        return float(np.mean([values[(root_pc + interval) % 12] / peak
                              for interval in intervals]))

    def _clear_candidate(self) -> None:
        self._candidate_chord = None
        self._candidate_root = "--"
        self._candidate_start_time = 0.0
        self._candidate_raw_event = None
        self._candidate_confidences.clear()
        self._candidate_frame_count = 0
        self._candidate_confidence = 0.0
        self._candidate_duration = 0.0

    def _append_observation(self, timestamp: float, symbol: str, root: str,
                            quality: str, bass_note: str, inversion: str,
                            notes: List[str], confidence: float,
                            chroma_vector: Optional[List[float]]) -> Dict:
        normalized_root = self._normalize_root(root, symbol)
        normalized_bass = self._normalize_root(bass_note)
        family = self._quality_family(quality, symbol)
        observation = {
            "timestamp": timestamp,
            "symbol": symbol,
            "root": normalized_root,
            "quality": family,
            "bass_note": normalized_bass,
            "inversion": inversion,
            "notes": list(notes),
            "confidence": float(confidence),
            "chroma_support": self._chroma_support(normalized_root, family, chroma_vector),
        }
        self._observation_window.append(observation)
        while (self._observation_window and
               timestamp - self._observation_window[0]["timestamp"] > self._support_window_seconds):
            self._observation_window.popleft()
        return observation

    def _update_candidate_vote(self, timestamp: float) -> None:
        current_root = self._normalize_root(self._chord_root, self._current_chord)
        root_scores: Dict[str, float] = {}
        contributions = []
        for observation in self._observation_window:
            if observation["root"] == "--" or observation["root"] == current_root:
                continue
            age = max(0.0, timestamp - observation["timestamp"])
            recency = float(np.exp(-age / max(0.10, self._support_window_seconds * 0.45)))
            weight = (observation["confidence"] *
                      (0.70 + 0.30 * observation["chroma_support"]) * recency)
            root_scores[observation["root"]] = root_scores.get(observation["root"], 0.0) + weight
            contributions.append((observation["root"], weight, observation))
            bass = observation["bass_note"]
            if bass not in ("--", observation["root"], current_root):
                bass_weight = weight * 0.30
                root_scores[bass] = root_scores.get(bass, 0.0) + bass_weight
                contributions.append((bass, bass_weight, observation))

        if not root_scores:
            self._clear_candidate()
            return
        best_root, best_weight = max(root_scores.items(), key=lambda item: item[1])
        total_weight = sum(root_scores.values())
        dominance = best_weight / max(1e-9, total_weight)
        family = [(weight, obs) for voted_root, weight, obs in contributions
                  if voted_root == best_root]
        direct = [(weight, obs) for weight, obs in family if obs["root"] == best_root]
        evidence = direct or family
        frames = len(evidence)
        first_time = min(obs["timestamp"] for _, obs in evidence)
        confidences = [obs["confidence"] for _, obs in evidence]
        weighted_confidence = sum(weight * obs["confidence"] for weight, obs in evidence) / \
            max(1e-9, sum(weight for weight, _ in evidence))

        symbol_scores: Dict[str, float] = {}
        quality_scores: Dict[str, float] = {}
        for weight, obs in direct:
            base_symbol = obs["symbol"].split("/")[0]
            symbol_scores[base_symbol] = symbol_scores.get(base_symbol, 0.0) + weight
            quality_scores[obs["quality"]] = quality_scores.get(obs["quality"], 0.0) + weight
        if len(symbol_scores) == 1:
            representative = next(iter(symbol_scores))
        else:
            quality = (max(quality_scores.items(), key=lambda item: item[1])[0]
                       if quality_scores else "major")
            representative = f"{best_root}m" if quality == "minor" else best_root
        latest = max((obs for _, obs in evidence), key=lambda obs: obs["timestamp"])

        self._candidate_root = best_root
        self._candidate_chord = representative
        self._candidate_start_time = first_time
        self._candidate_duration = max(0.0, timestamp - first_time)
        self._candidate_frame_count = frames
        self._candidate_confidences = confidences
        self._candidate_confidence = weighted_confidence * (0.65 + 0.35 * dominance)
        self._candidate_raw_event = {
            "root": best_root,
            "quality": self._quality_family(latest["quality"], representative),
            "bass_note": latest["bass_note"] if latest["bass_note"] != "--" else best_root,
            "inversion": latest["inversion"],
            "notes": latest["notes"],
            "dominance": dominance,
        }

    def _candidate_is_confirmed(self) -> bool:
        dominance = (self._candidate_raw_event or {}).get("dominance", 0.0)
        return bool(
            self._candidate_chord
            and self._candidate_root != "--"
            and self._candidate_frame_count >= 3
            and self._candidate_duration >= self._confirmation_time * 0.5
            and self._candidate_confidence >= self._min_confidence
            and dominance >= 0.55
        )

    def _promote_candidate(self, timestamp: float) -> None:
        transition_time = self._candidate_start_time
        if self._current_chord != "--":
            self._finalize_current_event_if_active(transition_time)
            self._previous_chord = self._current_chord
        self._current_chord = self._candidate_chord or "--"
        self._chord_start_time = transition_time
        self._chord_duration = max(0.0, timestamp - transition_time)
        self._confidences = list(self._candidate_confidences)
        self._chord_confidence = self._candidate_confidence
        self._last_supported_confidence = self._candidate_confidence
        self._last_current_support_time = timestamp
        self._unsupported_active_seconds = 0.0
        self._is_current_chord_stale = False
        self._awaiting_confirmation = False
        if self._candidate_raw_event:
            self._chord_root = self._candidate_raw_event["root"]
            self._chord_quality = self._candidate_raw_event["quality"]
            self._bass_note = self._candidate_raw_event["bass_note"]
            self._inversion = self._candidate_raw_event["inversion"]
            self._detected_notes = self._candidate_raw_event["notes"]
        self._observation_window.clear()
        self._clear_candidate()

    def _expire_current_chord(self, timestamp: float) -> None:
        if self._current_chord == "--":
            return
        self._finalize_current_event_if_active(timestamp)
        self._previous_chord = self._current_chord
        self._current_chord = "--"
        self._chord_start_time = timestamp
        self._chord_duration = 0.0
        self._chord_root = "--"
        self._chord_quality = "--"
        self._bass_note = "--"
        self._inversion = "root"
        self._detected_notes = []
        self._confidences.clear()
        self._chord_confidence = 0.0
        self._is_current_chord_stale = True
        self._awaiting_confirmation = True

    def _apply_unsupported_audio(self, timestamp: float, delta: float,
                                 audio_active: bool) -> None:
        if self._current_chord == "--":
            return
        self._chord_duration = max(0.0, timestamp - self._chord_start_time)
        if audio_active:
            self._unsupported_active_seconds += delta
            remaining = max(0.0, 1.0 - self._unsupported_active_seconds / self._stale_active_seconds)
            self._chord_confidence = self._last_supported_confidence * remaining
            if self._unsupported_active_seconds >= self._stale_active_seconds:
                self._expire_current_chord(timestamp)
        else:
            # Silêncio não prova mudança harmônica, mas a confiança exibida deixa
            # de parecer evidência recente depois de uma pausa prolongada.
            age = self.time_since_current_chord_support
            soft_decay = float(np.exp(-max(0.0, age - 1.5) / 3.0))
            self._chord_confidence = self._last_supported_confidence * soft_decay

    def process(
        self,
        raw_symbol: str,
        confidence: float,
        timestamp: float,
        root: str = "--",
        quality: str = "--",
        bass_note: str = "--",
        inversion: str = "root",
        detected_notes: Optional[List[str]] = None,
        chroma_vector: Optional[List[float]] = None,
        audio_activity: float = 0.0,
    ) -> str:
        """Estabiliza por suporte recente e votação de família harmônica."""
        timestamp = max(0.0, float(timestamp))
        notes = detected_notes or []
        previous_timestamp = self._last_timestamp
        delta = max(0.0, timestamp - previous_timestamp) if previous_timestamp >= 0.0 else 0.0

        # Seek invalida toda evidência temporal anterior.
        if previous_timestamp >= 0.0 and abs(timestamp - previous_timestamp) > 1.5:
            self._finalize_current_event_if_active(previous_timestamp)
            self._current_chord = "--"
            self._confidences.clear()
            self._chord_confidence = 0.0
            self._chord_root = "--"
            self._chord_quality = "--"
            self._bass_note = "--"
            self._inversion = "root"
            self._detected_notes = []
            self._observation_window.clear()
            self._clear_candidate()
            self._awaiting_confirmation = False
            self._last_current_support_time = -1.0
            self._last_supported_confidence = 0.0
            self._unsupported_active_seconds = 0.0
            self._is_current_chord_stale = False
        self._last_timestamp = timestamp

        valid = confidence >= self._min_confidence and raw_symbol not in ("", "--", "UNKNOWN", "N")
        audio_active = bool(valid or audio_activity >= 0.005)
        observation = None
        if valid:
            observation = self._append_observation(
                timestamp, raw_symbol, root, quality, bass_note, inversion,
                notes, confidence, chroma_vector)

        current_root = self._normalize_root(self._chord_root, self._current_chord)
        supports_current = bool(observation and observation["root"] == current_root and current_root != "--")

        if self._current_chord == "--" and valid and not self._awaiting_confirmation:
            # Mantém resposta imediata somente na primeira aquisição/seek. Caso
            # um acorde tenha expirado, a volta exige a mesma votação temporal.
            self._current_chord = raw_symbol
            self._chord_start_time = timestamp
            self._chord_duration = 0.0
            self._chord_root = observation["root"]
            self._chord_quality = observation["quality"]
            self._bass_note = observation["bass_note"] if observation["bass_note"] != "--" else observation["root"]
            self._inversion = inversion
            self._detected_notes = list(notes)
            self._chord_confidence = confidence
            self._last_supported_confidence = confidence
            self._confidences = [confidence]
            self._last_current_support_time = timestamp
            self._unsupported_active_seconds = 0.0
            self._is_current_chord_stale = False
            self._clear_candidate()
            return self._current_chord

        if supports_current:
            self._chord_duration = max(0.0, timestamp - self._chord_start_time)
            self._confidences.append(confidence)
            self._last_supported_confidence = (
                self._last_supported_confidence * 0.75 + confidence * 0.25)
            self._chord_confidence = self._last_supported_confidence
            self._last_current_support_time = timestamp
            self._unsupported_active_seconds = 0.0
            self._is_current_chord_stale = False
            self._chord_root = observation["root"]
            self._chord_quality = observation["quality"]
            self._bass_note = observation["bass_note"] if observation["bass_note"] != "--" else observation["root"]
            self._inversion = inversion
            if notes:
                self._detected_notes = list(notes)
            self._clear_candidate()
            return self._current_chord

        if valid:
            self._update_candidate_vote(timestamp)
            if self._candidate_is_confirmed():
                self._promote_candidate(timestamp)
                return self._current_chord

        self._apply_unsupported_audio(timestamp, delta, audio_active)

        if self._current_chord == "--" and valid:
            self._update_candidate_vote(timestamp)
            if self._candidate_is_confirmed():
                self._promote_candidate(timestamp)
        return self._current_chord
