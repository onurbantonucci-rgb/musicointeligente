"""Banco de levadas de baixo reais (compassos tocados por baixistas, em MIDI).

Montado por ``tools/bass_bank/build_bank.py`` a partir do Lakh MIDI Dataset
(Colin Raffel, CC-BY 4.0 — https://colinraffel.com/projects/lmd/). Cada
compasso guarda o CONTEXTO que um baixista ao vivo já conhece antes de tocá-lo
(levada da banda nos compassos anteriores, andamento, tendência de
intensidade, onde os acordes trocam) e as notas que o baixista real tocou,
como FUNÇÃO harmônica (R, R8, 5, 3, 7, APP, U±n, CHR).

A busca é vetorizada: roda a cada compasso, inclusive ao vivo.
"""

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import List, Optional, Sequence

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BANK_FILES = {"GOSPEL_WORSHIP": "bass_bank_gospel_worship.npz"}


@dataclass(frozen=True)
class BankNote:
    slot: int
    duration: int
    role: str
    velocity: int


@dataclass
class BankQuery:
    band_grid: Optional[np.ndarray]     # (16,) levada recente da banda, 0..1 (None = ainda sem histórico)
    bpm: float
    trend: float                        # energia do último compasso / média recente
    changes: int                        # máscara: bit (4*b) = troca de acorde no tempo b+1


class BassBank:
    def __init__(self, path: str):
        data = np.load(path, allow_pickle=False)
        self.roles: List[str] = [str(r) for r in data["roles"]]
        self.song_names = data["song_names"]
        self.song_id = data["song_id"]
        self.bar_index = data["bar_index"]
        self.weight = data["weight"].astype(np.float64)
        prev = data["prev_grid"].astype(np.float64) / 255.0
        # Quantos ataques a banda dava por compasso antes deste compasso do baixo.
        self.prev_density = np.sum(prev >= 0.35, axis=1).astype(np.float64)
        norms = np.linalg.norm(prev, axis=1, keepdims=True)
        self.prev_unit = np.divide(prev, norms, out=np.zeros_like(prev), where=norms > 0)
        self.log_bpm = np.log2(np.maximum(1.0, data["bpm"].astype(np.float64)))
        self.log_trend = np.log2(np.clip(data["trend"].astype(np.float64), 0.05, 20.0))
        self.changes = data["changes"].astype(np.int64)
        order = np.argsort(data["note_bar"], kind="stable")
        self._note_bar = data["note_bar"][order]
        self._note_slot = data["note_slot"][order]
        self._note_dur = data["note_dur"][order]
        self._note_role = data["note_role"][order]
        self._note_vel = data["note_vel"][order]
        count = len(self.song_id)
        self._starts = np.searchsorted(self._note_bar, np.arange(count), side="left")
        self._ends = np.searchsorted(self._note_bar, np.arange(count), side="right")
        # Máscara rítmica (16 bits) e riqueza melódica de cada compasso.
        self.rhythm = np.zeros((count, 16), dtype=np.float64)
        self.rhythm[self._note_bar, self._note_slot] = 1.0
        # Fração das notas que NÃO são a fundamental: riqueza melódica, independente
        # de quantas notas o compasso tem.
        root_id = self.roles.index("R") if "R" in self.roles else -1
        notes_per_bar = np.bincount(self._note_bar, minlength=count).astype(np.float64)
        non_root = np.bincount(self._note_bar[self._note_role != root_id], minlength=count)
        self.notes_per_bar = notes_per_bar
        self.melodic = np.divide(non_root, notes_per_bar, out=np.zeros(count), where=notes_per_bar > 0)
        self._catalog(count, root_id)
        # Índice (música, compasso) -> posição, para achar o compasso seguinte da mesma linha.
        self._keys = self.song_id.astype(np.int64) * 100_000 + self.bar_index.astype(np.int64)
        self._key_order = np.argsort(self._keys, kind="stable")
        self._sorted_keys = self._keys[self._key_order]
        self.source = str(data["source"])

    # Subdivisão: 0 = semínimas, 1 = colcheias, 2 = semicolcheias.
    SUBDIVISION_NAMES = ("QUARTER", "EIGHTH", "SIXTEENTH")

    def _catalog(self, count: int, root_id: int) -> None:
        """Metadados de cada compasso (para casar frase × momento da música).

        - ``notes_per_bar``: densidade (ataques do baixo no compasso);
        - ``subdivision``: menor subdivisão usada (semínima/colcheia/semicolcheia);
        - ``syncopation``: fração de ataques em contratempo sem o tempo seguinte;
        - ``space_ratio``: fração das 16 posições sem ataque;
        - ``mean_duration``: duração média das notas (semicolcheias) — articulação;
        - ``fill_level``: notas não-fundamentais na 2ª metade (virada/condução);
        - ``bpm`` (``log_bpm``), ``trend`` (energia), ``changes`` (ritmo harmônico) e
          ``prev_grid`` (levada da banda) já vêm do construtor do banco.
        """
        rhythm = self.rhythm
        odd = rhythm[:, 1::2].sum(axis=1) > 0
        offbeat_eighths = rhythm[:, 2::4].sum(axis=1) > 0
        self.subdivision = np.where(odd, 2, np.where(offbeat_eighths, 1, 0)).astype(np.int64)
        downbeats = rhythm[:, 0::4]
        following = np.roll(downbeats, -1, axis=1)
        following[:, -1] = 1.0          # o tempo 1 do próximo compasso conta como atacado
        syncopated = (rhythm[:, 2::4] * (1.0 - following)).sum(axis=1)
        self.syncopation = np.divide(syncopated, self.notes_per_bar, out=np.zeros(count),
                                     where=self.notes_per_bar > 0)
        self.space_ratio = 1.0 - rhythm.mean(axis=1)
        durations = np.bincount(self._note_bar, weights=self._note_dur.astype(np.float64),
                                minlength=count)
        self.mean_duration = np.divide(durations, self.notes_per_bar, out=np.zeros(count),
                                       where=self.notes_per_bar > 0)
        late = (self._note_slot >= 8) & (self._note_role != root_id)
        self.fill_level = np.bincount(self._note_bar[late], minlength=count).astype(np.float64)

    def describe(self, index: int) -> dict:
        """Ficha de um compasso do banco (diagnóstico / painel)."""
        return {
            "song": self.song_name(index), "bpm": round(float(2 ** self.log_bpm[index]), 1),
            "notes": int(self.notes_per_bar[index]),
            "subdivision": self.SUBDIVISION_NAMES[int(self.subdivision[index])],
            "syncopation": round(float(self.syncopation[index]), 2),
            "space_ratio": round(float(self.space_ratio[index]), 2),
            "mean_duration": round(float(self.mean_duration[index]), 1),
            "fill_level": int(self.fill_level[index]),
            "melodic": round(float(self.melodic[index]), 2),
        }

    def __len__(self) -> int:
        return len(self.song_id)

    def notes(self, index: int) -> List[BankNote]:
        a, b = self._starts[index], self._ends[index]
        return [BankNote(int(s), int(d), self.roles[int(r)], int(v))
                for s, d, r, v in zip(self._note_slot[a:b], self._note_dur[a:b],
                                      self._note_role[a:b], self._note_vel[a:b])]

    def successor(self, index: int) -> Optional[int]:
        """Compasso seguinte do MESMO baixista real (continuidade da frase)."""
        key = int(self._keys[index]) + 1
        position = int(np.searchsorted(self._sorted_keys, key))
        if position < len(self._sorted_keys) and self._sorted_keys[position] == key:
            return int(self._key_order[position])
        return None

    def song_name(self, index: int) -> str:
        return str(self.song_names[int(self.song_id[index])])

    def score(self, query: BankQuery) -> np.ndarray:
        """Quanto cada compasso do banco combina com o contexto (maior = melhor)."""
        score = np.zeros(len(self))
        if query.band_grid is not None and float(np.max(query.band_grid)) > 0:
            grid = np.asarray(query.band_grid, dtype=np.float64)
            unit = grid / np.linalg.norm(grid)
            score += 3.0 * (self.prev_unit @ unit)
            # Densidade da banda (quantos ataques): o baixo acompanha o quanto ela está cheia.
            density = float(np.sum(grid / float(np.max(grid)) >= 0.35))
            score -= 1.5 * np.abs(np.log2((self.prev_density + 1.0) / (density + 1.0)))
        if query.bpm > 0:
            score -= 4.0 * np.abs(self.log_bpm - np.log2(query.bpm))
        score -= 0.6 * np.abs(self.log_trend - np.log2(np.clip(query.trend, 0.05, 20.0)))
        differing = np.zeros(len(self))
        for beat in range(4):
            bit = 1 << (beat * 4)
            differing += ((self.changes & bit) > 0) != bool(query.changes & bit)
        score -= 0.8 * differing
        score += 0.15 * np.log1p(self.weight)
        return score


@lru_cache(maxsize=4)
def load_bank(style_name: str) -> Optional[BassBank]:
    filename = BANK_FILES.get(style_name)
    if not filename:
        return None
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    return BassBank(path)


def band_lock_penalty(rhythm: np.ndarray, band_grid: Optional[np.ndarray],
                      threshold: float = 0.3) -> np.ndarray:
    """Fração de ataques de cada compasso que cairiam onde a banda não ataca."""
    if band_grid is None or float(np.max(band_grid)) <= 0:
        return np.zeros(len(rhythm))
    silent = (np.asarray(band_grid) / float(np.max(band_grid)) < threshold).astype(np.float64)
    silent[0] = 0.0                       # o tempo 1 é sempre do baixo
    hits = rhythm.sum(axis=1)
    return np.divide(rhythm @ silent, hits, out=np.zeros(len(rhythm)), where=hits > 0)


def top_candidates(scores: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(scores))
    best = np.argpartition(-scores, count - 1)[:count]
    return best[np.argsort(-scores[best])]


def roles_used(bank: BassBank, indices: Sequence[int]) -> List[set]:
    return [{note.role for note in bank.notes(i)} for i in indices]
