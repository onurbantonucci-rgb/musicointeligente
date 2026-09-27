"""Histórico Temporal e Estabilizador de Tonalidade (Key) do Virtual Band AI.

Responsável por:
1. Registrar mudanças persistentes de tonalidade (KeyEvent) com start_time, end_time, duration e confidence.
2. Evitar falsas modulações geradas por acordes de passagem ou oscilações transitórias.
3. Rastrear o tempo de vigência da tonalidade atual (key_duration) e a tonalidade anterior (previous_key).
4. Preparar a arquitetura para o futuro KeyChangeDetector.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import collections
import numpy as np

from app.music.constants import (
    MIN_KEY_CONFIDENCE,
    KEY_CHANGE_CONFIRMATION_TIME,
    KEY_SCORE_MARGIN,
    WEIGHT_KEY_INSTANTANEOUS,
    WEIGHT_KEY_HISTORICAL_CHROMA,
    WEIGHT_KEY_CHORD_CONTEXT,
    WEIGHT_KEY_STABILITY,
    MAX_KEY_HISTORY_ENTRIES,
)
from app.music.theory import compute_chord_key_compatibility


@dataclass
class KeyEvent:
    """Evento consolidado representando um período estável de tonalidade musical."""
    key: str                    # Nome da tonalidade (ex: "C Major", "A Minor")
    start_time: float           # Início da tonalidade em segundos
    end_time: float             # Fim da tonalidade em segundos
    duration: float             # Duração da tonalidade em segundos (end_time - start_time)
    confidence: float           # Grau médio de confiança estatística (0.0 a 1.0)
    root: str = "--"            # Tônica da escala (ex: "C")
    scale_type: str = "--"      # "Major" ou "Minor"

    def to_dict(self) -> Dict:
        return {
            "time_str": self.format_timestamp(self.start_time),
            "key": self.key,
            "duration_str": f"{self.duration:.1f} s",
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


class KeyHistory:
    """Repositório ordenado de eventos discretos de tonalidade confirmada.
    
    Por padrão armazena todas as modulações sem descarte (max_entries=None).
    """

    def __init__(self, max_entries: Optional[int] = MAX_KEY_HISTORY_ENTRIES):
        self._max_entries = max_entries
        self._events: collections.deque = collections.deque(maxlen=max_entries)
        self._version: int = 0

    @property
    def version(self) -> int:
        return self._version

    def add_event(self, event: KeyEvent) -> None:
        """Registra uma modulação ou período tonal finalizado."""
        if event.key == "--" or event.duration < 0.5:
            return
        self._events.append(event)
        self._version += 1

    def get_events(self) -> List[KeyEvent]:
        return list(self._events)

    def get_recent(self, count: int = 5) -> List[KeyEvent]:
        return list(self._events)[-count:]

    def get_summary_text(self) -> str:
        if not self._events:
            return "Aguardando definição tonal..."
        items = [f"{e.key} ({e.duration:.1f}s)" for e in list(self._events)[-3:]]
        return " → ".join(items)

    def clear(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)


class KeyStabilizer:
    """Estabilizador contextual de tonalidade com rejeição de falsas modulações.
    
    Implementa pontuação contextual ponderada:
      score = w_inst * S_inst + w_chroma * S_chroma + w_chord * S_chord + w_stab * S_stab
      
    Rejeita mudanças espúrias causadas por acordes diatônicos individuais (ex: Em, C ou D dentro
    de G Major). Confirma modulações apenas se uma candidata mantiver margem superior persistente
    ao longo de KEY_CHANGE_CONFIRMATION_TIME.
    """

    def __init__(
        self,
        min_confidence: float = MIN_KEY_CONFIDENCE,
        confirmation_time: float = KEY_CHANGE_CONFIRMATION_TIME,
        score_margin: float = KEY_SCORE_MARGIN,
        history: Optional[KeyHistory] = None,
        w_inst: float = WEIGHT_KEY_INSTANTANEOUS,
        w_chroma: float = WEIGHT_KEY_HISTORICAL_CHROMA,
        w_chord: float = WEIGHT_KEY_CHORD_CONTEXT,
        w_stab: float = WEIGHT_KEY_STABILITY,
    ):
        self._min_confidence = min_confidence
        self._confirmation_time = confirmation_time
        self._score_margin = score_margin
        self._history = history if history is not None else KeyHistory()

        self._w_inst = w_inst
        self._w_chroma = w_chroma
        self._w_chord = w_chord
        self._w_stab = w_stab

        # Estado atual confirmado
        self._current_key: str = "--"
        self._key_start_time: float = 0.0
        self._key_duration: float = 0.0
        self._key_confidence: float = 0.0
        self._key_root: str = "--"
        self._scale_type: str = "--"

        # Tonalidade anterior
        self._previous_key: str = "--"

        # Acumulador de confiança da tonalidade ativa
        self._confidences: List[float] = []

        # Estado da tonalidade candidata a modulação
        self._candidate_key: Optional[str] = None
        self._candidate_start_time: float = 0.0
        self._candidate_duration: float = 0.0
        self._candidate_confidence: float = 0.0
        self._candidate_confidences: List[float] = []
        self._candidate_root: str = "--"
        self._candidate_type: str = "--"

        # Centro tonal local (preparado para futura diferenciação global vs local)
        self._local_tonal_center: str = "--"

    @property
    def history(self) -> KeyHistory:
        return self._history

    @property
    def current_key(self) -> str:
        return self._current_key

    @property
    def previous_key(self) -> str:
        return self._previous_key

    @property
    def key_start_time(self) -> float:
        return self._key_start_time

    @property
    def key_duration(self) -> float:
        return self._key_duration

    @property
    def key_confidence(self) -> float:
        return self._key_confidence

    @property
    def candidate_key(self) -> str:
        return self._candidate_key if self._candidate_key is not None else "--"

    @property
    def candidate_confidence(self) -> float:
        return self._candidate_confidence

    @property
    def candidate_duration(self) -> float:
        return self._candidate_duration

    @property
    def local_tonal_center(self) -> str:
        return self._local_tonal_center

    def reset(self) -> None:
        self._finalize_current_key()
        self._current_key = "--"
        self._key_start_time = 0.0
        self._key_duration = 0.0
        self._key_confidence = 0.0
        self._key_root = "--"
        self._scale_type = "--"
        self._previous_key = "--"
        self._confidences.clear()
        self._candidate_key = None
        self._candidate_start_time = 0.0
        self._candidate_duration = 0.0
        self._candidate_confidence = 0.0
        self._candidate_confidences.clear()
        self._candidate_root = "--"
        self._candidate_type = "--"
        self._local_tonal_center = "--"
        self._history.clear()

    def _finalize_current_key(self, end_time: Optional[float] = None) -> None:
        if self._current_key != "--" and self._confidences:
            actual_end = end_time if end_time is not None else (self._key_start_time + self._key_duration)
            dur = max(0.1, actual_end - self._key_start_time)
            if dur >= 0.5:
                avg_conf = float(np.mean(self._confidences))
                event = KeyEvent(
                    key=self._current_key,
                    start_time=self._key_start_time,
                    end_time=actual_end,
                    duration=dur,
                    confidence=avg_conf,
                    root=self._key_root,
                    scale_type=self._scale_type
                )
                self._history.add_event(event)

    def _compute_key_score(
        self,
        candidate_key: str,
        raw_key: str,
        raw_conf: float,
        recent_chords: Optional[List],
        all_key_scores: Optional[Dict[str, Tuple[float, float]]]
    ) -> float:
        """Calcula a pontuação composta contextual para uma tonalidade candidata."""
        parts = candidate_key.split()
        if len(parts) != 2:
            return 0.0
        cand_root, cand_scale = parts[0], parts[1]

        # 1. Correlação Instantânea e 2. Cromagrama Histórico
        if all_key_scores and candidate_key in all_key_scores:
            corr_short, corr_long = all_key_scores[candidate_key]
            s_inst = float(np.clip((corr_short + 1.0) / 2.0, 0.0, 1.0))
            s_chroma = float(np.clip((corr_long + 1.0) / 2.0, 0.0, 1.0))
        else:
            s_inst = raw_conf if candidate_key == raw_key else (raw_conf * 0.4)
            s_chroma = s_inst

        # 3. Contexto Harmônico de Acordes Recentes com Ponderação Temporal
        if recent_chords:
            chord_weights = []
            chord_scores = []
            n_chords = len(recent_chords)
            for idx, ch in enumerate(recent_chords):
                ch_root = getattr(ch, "root", "")
                ch_qual = getattr(ch, "quality", "major")
                ch_dur = max(0.2, getattr(ch, "duration", 1.0))
                # Recência: acordes mais recentes têm maior peso (fator 0.85 por acorde prévio)
                recency = 0.85 ** (n_chords - 1 - idx)
                compat = compute_chord_key_compatibility(ch_root, ch_qual, cand_root, cand_scale)
                chord_weights.append(ch_dur * recency)
                chord_scores.append(compat)
            if chord_weights and sum(chord_weights) > 0:
                s_chord = float(np.average(chord_scores, weights=chord_weights))
            else:
                s_chord = 0.50
        else:
            s_chord = 0.50

        # Pontuação base de evidência musical (instantânea + cromagrama histórico + acordes)
        base_score = (
            self._w_inst * s_inst +
            self._w_chroma * s_chroma +
            self._w_chord * s_chord
        )
        # Bônus de estabilidade (inércia harmônica conferida à tonalidade atual consolidada)
        stab_bonus = self._w_stab if candidate_key == self._current_key else 0.0

        total = base_score + stab_bonus
        return float(np.clip(total, 0.0, 1.0))


    def process(
        self,
        raw_key: str,
        confidence: float,
        timestamp: float,
        root: str = "--",
        scale_type: str = "--",
        recent_chords: Optional[List] = None,
        all_key_scores: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> str:
        """Processa a estimativa instantânea e contextual de tonalidade."""
        timestamp = max(0.0, float(timestamp))

        # Atualiza centro tonal local imediato
        if raw_key != "--" and confidence >= 0.30:
            self._local_tonal_center = root

        # Se o sinal for completamente nulo
        if raw_key == "--" and (all_key_scores is None or len(all_key_scores) == 0):
            if self._current_key != "--":
                self._key_duration = max(0.0, timestamp - self._key_start_time)
            return self._current_key

        # Determina a melhor candidata avaliando as 24 tonalidades
        candidate_pool = list(all_key_scores.keys()) if all_key_scores else []
        if raw_key != "--" and raw_key not in candidate_pool:
            candidate_pool.append(raw_key)
        if self._current_key != "--" and self._current_key not in candidate_pool:
            candidate_pool.append(self._current_key)

        best_key = raw_key
        best_score = 0.0

        scored_keys: Dict[str, float] = {}
        for k in candidate_pool:
            sc = self._compute_key_score(k, raw_key, confidence, recent_chords, all_key_scores)
            scored_keys[k] = sc
            if sc > best_score:
                best_score = sc
                best_key = k

        # Caso 1: Primeira tonalidade confirmada da música
        if self._current_key == "--":
            if best_key != "--" and best_score >= self._min_confidence:
                parts = best_key.split()
                self._current_key = best_key
                self._key_start_time = timestamp
                self._key_duration = 0.0
                self._key_confidence = best_score
                self._key_root = parts[0] if len(parts) > 0 else root
                self._scale_type = parts[1] if len(parts) > 1 else scale_type
                self._confidences = [best_score]
                self._candidate_key = None
                self._candidate_duration = 0.0
                self._candidate_confidence = 0.0
            return self._current_key

        # Caso 2: A tonalidade atual continua vencendo ou a margem da adversária é insuficiente
        curr_score = scored_keys.get(self._current_key, 0.0)
        margin = best_score - curr_score

        if best_key == self._current_key or margin < self._score_margin:
            # Tonalidade atual consolidada permanece estável
            self._key_duration = max(0.0, timestamp - self._key_start_time)
            self._confidences.append(curr_score)
            self._key_confidence = (self._key_confidence * 0.92) + (curr_score * 0.08)

            # Se havia uma candidata efêmera, zera sua vigência
            self._candidate_key = None
            self._candidate_duration = 0.0
            self._candidate_confidence = 0.0
            self._candidate_confidences.clear()
            return self._current_key

        # Caso 3: Uma adversária genuína supera a tonalidade atual por KEY_SCORE_MARGIN
        if best_score >= self._min_confidence:
            if self._candidate_key != best_key:
                # Nova candidata a modulação detectada
                parts = best_key.split()
                self._candidate_key = best_key
                self._candidate_start_time = timestamp
                self._candidate_duration = 0.0
                self._candidate_confidences = [best_score]
                self._candidate_confidence = best_score
                self._candidate_root = parts[0] if len(parts) > 0 else "--"
                self._candidate_type = parts[1] if len(parts) > 1 else "--"
                self._key_duration = max(0.0, timestamp - self._key_start_time)
                return self._current_key

            # Candidata continua mantendo liderança consistente ao longo do tempo
            self._candidate_confidences.append(best_score)
            self._candidate_duration = max(0.0, timestamp - self._candidate_start_time)
            self._candidate_confidence = float(np.mean(self._candidate_confidences))

            # Confirma modulação somente após persistência por KEY_CHANGE_CONFIRMATION_TIME
            if self._candidate_duration >= self._confirmation_time:
                self._finalize_current_key(self._candidate_start_time)

                self._previous_key = self._current_key
                self._current_key = self._candidate_key
                self._key_start_time = self._candidate_start_time
                self._key_duration = max(0.0, timestamp - self._candidate_start_time)
                self._confidences = list(self._candidate_confidences)
                self._key_confidence = self._candidate_confidence
                self._key_root = self._candidate_root
                self._scale_type = self._candidate_type

                self._candidate_key = None
                self._candidate_duration = 0.0
                self._candidate_confidence = 0.0
                self._candidate_confidences.clear()
                return self._current_key

        self._key_duration = max(0.0, timestamp - self._key_start_time)
        return self._current_key

