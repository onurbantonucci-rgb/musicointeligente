"""Segue a LETRA cantada, ao vivo, como segunda referência de posição na cifra.

Os acordes dizem QUAL harmonia soa, mas progressões se repetem (verso 1 e 2,
refrões iguais). A letra desfaz a ambiguidade: reconhecendo algumas palavras
cantadas, sabemos em que LINHA da cifra a banda está — e cada palavra da cifra
sabe em que compasso cai.

Peças:
- ``LyricsIndex``: as linhas da letra da cifra, palavra a palavra, com o
  compasso de cada palavra (a cifra guarda a letra em pedaços cortados nas
  posições dos acordes).
- ``LyricsMatcher``: compara as últimas palavras ouvidas com as linhas da letra
  (tolerante a erros de reconhecimento) e diz onde estamos.
- ``LyricsFollower``: reconhecimento de fala em fluxo (Vosk, local, sem
  internet) numa thread própria, com vocabulário restrito às palavras da letra.

O reconhecimento leva algumas centenas de ms: serve para CONFIRMAR/CORRIGIR a
linha e a seção, não para marcar o tempo (isso continua com as batidas).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import queue
import re
import threading
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                                 "MusicoInteligente", "models", "vosk-model-small-pt-0.3")
_RECOGNITION_RATE = 16000
_PARTIAL_DELAY = 0.35        # atraso típico entre cantar a palavra e ela aparecer no parcial


def normalize_word(word: str) -> str:
    text = unicodedata.normalize("NFD", word.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]", "", text)


def tokenize(text: str) -> List[str]:
    return [w for w in (normalize_word(t) for t in re.split(r"\s+", text)) if w]


@dataclass(frozen=True)
class LyricWord:
    word: str
    bar: int                 # compasso da cifra em que a palavra é cantada


@dataclass
class LyricLine:
    key: Tuple[str, int]     # (seção, nº da linha no texto)
    section_id: str
    section_name: str
    section_index: int
    first_in_section: bool
    words: List[LyricWord]

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)


class LyricsIndex:
    """Linhas da letra da cifra com o compasso de cada palavra."""

    def __init__(self, lines: Sequence[LyricLine]):
        self.lines = list(lines)

    @classmethod
    def from_chart(cls, chart) -> "LyricsIndex":
        lines: List[LyricLine] = []
        for section_index, section in enumerate(getattr(chart, "sections", []) or []):
            segments = sorted(getattr(section, "lyrics", []) or [],
                              key=lambda s: (s.line_number, s.bar_offset, s.beat_offset))
            by_line: Dict[int, list] = {}
            for segment in segments:
                by_line.setdefault(segment.line_number, []).append(segment)
            for order, (line_number, parts) in enumerate(sorted(by_line.items())):
                # Os pedaços são fatias da mesma linha cortadas nas posições dos
                # acordes ("Antes de eu fal" + "ar"): reconstrói a linha e lembra
                # em que compasso começa cada palavra.
                text, bars = "", []
                for part in parts:
                    piece = part.text.strip() + " "      # cortes caem no início de palavras
                    text += piece
                    bars += [part.bar_offset] * len(piece)
                words = []
                for match in re.finditer(r"\S+", text):
                    normalized = normalize_word(match.group())
                    if normalized:
                        words.append(LyricWord(normalized, int(bars[match.start()])))
                if words:
                    lines.append(LyricLine((section.id, line_number), section.id, section.name,
                                           section_index, order == 0, words))
        return cls(lines)

    def vocabulary(self) -> List[str]:
        return sorted({w.word for line in self.lines for w in line.words})

    def __bool__(self) -> bool:
        return bool(self.lines)


@dataclass
class LyricsEvidence:
    time: float              # instante (áudio) em que a última palavra casada foi cantada
    bar: int                 # compasso da cifra dessa palavra
    line: LyricLine
    matched: int
    confidence: float
    alternatives: List[Tuple[int, LyricLine]] = field(default_factory=list)  # outras linhas iguais
    weight: float = 0.0      # força da evidência (soma dos pesos das palavras casadas)

    @property
    def section_start_line(self) -> bool:
        return self.line.first_in_section


def word_weight(word: str) -> float:
    """Palavras longas identificam a linha; monossílabos ("e", "de", "oh") quase nada."""
    return min(1.0, len(word) / 5.0)


def _similar(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if len(a) >= 4 and len(b) >= 4 and (a.startswith(b[:4]) or b.startswith(a[:4])):
        return 0.7
    return 0.0


class LyricsMatcher:
    """Acha a linha da letra que melhor explica as últimas palavras ouvidas."""

    def __init__(self, index: LyricsIndex, window_words: int = 8, horizon: float = 8.0):
        self._index = index
        self._window = window_words
        self._horizon = horizon
        self._heard: List[Tuple[str, float]] = []

    def hear(self, word: str, time: float) -> None:
        word = normalize_word(word)
        if not word or word == "unk":
            return
        self._heard.append((word, time))
        self._heard = [(w, t) for w, t in self._heard if time - t <= self._horizon][-self._window:]

    def clear(self) -> None:
        self._heard.clear()

    def best(self, min_words: int = 2, min_score: float = 0.6,
             min_weight: float = 1.9) -> Optional[LyricsEvidence]:
        """``min_weight``: soma dos pesos das palavras casadas (2 palavras longas ou
        3–4 médias); ``min_score``: fração da parte final ouvida que casou."""
        heard = self._heard
        if len(heard) < min_words:
            return None
        best_by_word: Dict[int, tuple] = {}
        lines = self._index.lines
        owner = {id(word): line for line in lines for word in line.words}
        for i, line in enumerate(lines):
            # Uma janela pode atravessar o fim de uma linha e o começo da seguinte.
            target = list(line.words) + (list(lines[i + 1].words[:3]) if i + 1 < len(lines) else [])
            words = [w for w, _ in heard]
            # O que importa é o que foi cantado AGORA: testa os finais da janela
            # (as últimas k palavras) para palavras antigas não diluírem a nota.
            best_item = None
            for k in range(min_words, len(words) + 1):
                window = words[-k:]
                score, matched, last = self._align(window, target)
                total = sum(word_weight(w) for w in window)
                if matched >= min_words and last is not None and score >= min_weight:
                    candidate = (score / total, matched, last, score)
                    if best_item is None or candidate[0] > best_item[0]:
                        best_item = candidate
            if best_item is not None:
                normalized, matched, last, weight = best_item
                # A evidência é da linha onde está a ÚLTIMA palavra casada; janelas
                # diferentes que chegam à mesma palavra são o mesmo candidato.
                item = (normalized, matched, last, owner[id(last)], weight)
                previous = best_by_word.get(id(last))
                if previous is None or item[0] > previous[0]:
                    best_by_word[id(last)] = item
        scored = list(best_by_word.values())
        if not scored:
            return None
        scored.sort(key=lambda item: -item[0])
        top_score, matched, last, line, weight = scored[0]
        if top_score < min_score:
            return None
        same_text = [(item[2].bar, item[3]) for item in scored[1:]
                     if item[3].text == line.text and abs(item[0] - top_score) < 1e-6]
        different = [item for item in scored[1:] if item[3].text != line.text]
        if different and top_score - different[0][0] < 0.15:
            return None                      # duas linhas diferentes explicam igual: ambíguo
        return LyricsEvidence(time=heard[-1][1], bar=last.bar, line=line, matched=matched,
                              confidence=min(1.0, top_score), alternatives=same_text,
                              weight=float(weight))

    @staticmethod
    def _align(heard: Sequence[str], target: Sequence[LyricWord]) -> Tuple[float, int, Optional[LyricWord]]:
        """Alinhamento em ordem (estilo LCS) com semelhança aproximada entre palavras."""
        n, m = len(heard), len(target)
        score = np.zeros((n + 1, m + 1))
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                gain = _similar(heard[i - 1], target[j - 1].word) * word_weight(target[j - 1].word)
                score[i, j] = max(score[i - 1, j], score[i, j - 1],
                                  score[i - 1, j - 1] + gain if gain > 0 else 0.0)
        # A última palavra OUVIDA tem de casar com alguma palavra da linha.
        best_j, best_value = None, 0.0
        for j in range(1, m + 1):
            gain = _similar(heard[-1], target[j - 1].word) * word_weight(target[j - 1].word)
            if gain > 0:
                value = score[n - 1, j - 1] + gain
                if value > best_value:
                    best_value, best_j = value, j
        if best_j is None:
            return float(score[n, m]), 0, None
        # Quantas palavras ouvidas casaram (contagem simples, sem pesos).
        matched = sum(1 for w in heard if any(_similar(w, t.word) > 0 for t in target[:best_j]))
        return best_value, matched, target[best_j - 1]


class LyricsFollower:
    """Reconhecimento de fala em fluxo (Vosk) + casamento com a letra da cifra."""

    def __init__(self, index: LyricsIndex, model_dir: str = DEFAULT_MODEL_DIR):
        self._index = index
        self._matcher = LyricsMatcher(index)
        self._queue: "queue.Queue[Optional[Tuple[np.ndarray, int, float]]]" = queue.Queue(maxsize=200)
        self._evidence: Optional[LyricsEvidence] = None
        self._lock = threading.Lock()
        self._fed_until: Optional[float] = None
        self._recognizer = None
        self._error: Optional[str] = None
        try:
            import vosk  # noqa: WPS433 — dependência opcional
            vosk.SetLogLevel(-1)
            model = vosk.Model(model_dir)
            grammar = json.dumps(index.vocabulary() + ["[unk]"], ensure_ascii=False)
            self._recognizer = vosk.KaldiRecognizer(model, _RECOGNITION_RATE, grammar)
        except Exception as exc:                      # sem pacote/modelo: recurso desligado
            self._error = str(exc)
            logger.info("Seguimento da letra desligado: %s", exc)
            return
        self._last_partial: List[str] = []
        self._stream_time = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def available(self) -> bool:
        return self._recognizer is not None

    @property
    def error(self) -> Optional[str]:
        return self._error

    def feed(self, audio: np.ndarray, sample_rate: int, start_time: float) -> None:
        """Entrega áudio NOVO (sem sobreposição) para o reconhecimento, sem bloquear."""
        if self._recognizer is None or sample_rate <= 0:
            return
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        end_time = start_time + len(audio) / sample_rate
        if self._fed_until is not None:
            if start_time < self._fed_until - 1.0:
                self._fed_until = None                    # seek para trás: recomeça o fluxo
            else:
                skip = int(round(max(0.0, self._fed_until - start_time) * sample_rate))
                audio = audio[skip:]
                start_time = max(start_time, self._fed_until)
        if audio.size == 0:
            return
        self._fed_until = end_time
        try:
            self._queue.put_nowait((audio, sample_rate, start_time))
        except queue.Full:
            pass                                          # CPU atrasada: perde um pedaço, não trava

    def take_evidence(self) -> Optional[LyricsEvidence]:
        with self._lock:
            evidence, self._evidence = self._evidence, None
        return evidence

    def reset(self) -> None:
        self._fed_until = None
        self._matcher.clear()
        with self._lock:
            self._evidence = None
        if self._recognizer is not None:
            self._queue.put(None)

    def close(self) -> None:
        if self._recognizer is not None:
            self._queue.put(("__stop__", 0, 0.0))

    # ------------------------------------------------------------ thread
    def _run(self) -> None:
        from scipy.signal import resample_poly
        while True:
            item = self._queue.get()
            if item is None:
                self._recognizer.Reset()
                self._last_partial = []
                continue
            audio, rate, start = item
            if isinstance(audio, str):
                return
            if rate != _RECOGNITION_RATE:
                g = np.gcd(int(rate), _RECOGNITION_RATE)
                audio = resample_poly(audio, _RECOGNITION_RATE // g, int(rate) // g)
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            now = start + len(audio) / _RECOGNITION_RATE
            final = self._recognizer.AcceptWaveform(pcm)
            text = json.loads(self._recognizer.Result() if final else
                              self._recognizer.PartialResult()).get("text" if final else "partial", "")
            words = [w for w in text.split() if w and w != "[unk]"]
            # Palavras novas em relação ao último parcial (o parcial cresce).
            common = 0
            while (common < len(words) and common < len(self._last_partial) and
                   words[common] == self._last_partial[common]):
                common += 1
            for word in words[common:]:
                self._matcher.hear(word, now - _PARTIAL_DELAY)
            self._last_partial = [] if final else words
            if len(words) > 12 and not final:
                self._recognizer.Reset()                  # mantém os parciais curtos
                self._last_partial = []
            if len(words) > common:
                evidence = self._matcher.best()
                if evidence is not None:
                    with self._lock:
                        self._evidence = evidence
