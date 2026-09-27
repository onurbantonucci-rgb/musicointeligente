"""Gerenciador de Repertórios e Navegação de Músicas (SetlistManager v0.4).

Controla a seleção, troca de música, histórico isolado e navegação
(primeira, última, próxima, anterior) entre as músicas de um Setlist.
"""

from typing import Optional, List, Dict
import json
import os

from app.setlist.setlist import Setlist
from app.song.song import Song


class SetlistManager:
    """Gerenciador de alto nível para operações em Setlists."""

    def __init__(self, default_setlist: Optional[Setlist] = None):
        self._setlists: Dict[str, Setlist] = {}
        if default_setlist:
            self._setlists[default_setlist.id] = default_setlist
            self._active_setlist_id: Optional[str] = default_setlist.id
        else:
            first_set = Setlist(name="Repertório Principal")
            self._setlists[first_set.id] = first_set
            self._active_setlist_id = first_set.id

    @property
    def active_setlist(self) -> Setlist:
        if self._active_setlist_id not in self._setlists:
            first_set = Setlist(name="Repertório Principal")
            self._setlists[first_set.id] = first_set
            self._active_setlist_id = first_set.id
        return self._setlists[self._active_setlist_id]

    @property
    def all_setlists(self) -> List[Setlist]:
        return list(self._setlists.values())

    def create_setlist(self, name: str, description: str = "") -> Setlist:
        """Cria e cadastra um novo Setlist."""
        new_set = Setlist(name=name, description=description)
        self._setlists[new_set.id] = new_set
        return new_set

    def select_setlist(self, setlist_id: str) -> Optional[Setlist]:
        if setlist_id in self._setlists:
            self._active_setlist_id = setlist_id
            return self._setlists[setlist_id]
        return None

    def add_song(self, song: Song, setlist_id: Optional[str] = None) -> None:
        target = self._setlists.get(setlist_id) if setlist_id else self.active_setlist
        if target:
            target.add_song(song)

    def remove_song(self, song_id: str, setlist_id: Optional[str] = None) -> bool:
        target = self._setlists.get(setlist_id) if setlist_id else self.active_setlist
        return target.remove_song(song_id) if target else False

    def reorder_songs(self, song_ids: List[str], setlist_id: Optional[str] = None) -> None:
        target = self._setlists.get(setlist_id) if setlist_id else self.active_setlist
        if target:
            target.reorder(song_ids)

    def select_song(self, song_id: str, setlist_id: Optional[str] = None) -> Optional[Song]:
        target = self._setlists.get(setlist_id) if setlist_id else self.active_setlist
        if target:
            s = target.get_song(song_id)
            if s:
                target.active_song_id = s.id
                return s
        return None

    def get_active_song(self) -> Optional[Song]:
        return self.active_setlist.get_active_song()

    # ============================================================
    # Navegação entre canções
    # ============================================================
    def next_song(self) -> Optional[Song]:
        """Avança para a próxima música no Setlist ativo."""
        songs = self.active_setlist.songs
        if not songs:
            return None
        active_id = self.active_setlist.active_song_id
        current_idx = next((i for i, s in enumerate(songs) if s.id == active_id), -1)
        next_idx = (current_idx + 1) if (current_idx + 1) < len(songs) else current_idx
        self.active_setlist.active_song_id = songs[next_idx].id
        return songs[next_idx]

    def prev_song(self) -> Optional[Song]:
        """Retrocede para a música anterior no Setlist ativo."""
        songs = self.active_setlist.songs
        if not songs:
            return None
        active_id = self.active_setlist.active_song_id
        current_idx = next((i for i, s in enumerate(songs) if s.id == active_id), 0)
        prev_idx = max(0, current_idx - 1)
        self.active_setlist.active_song_id = songs[prev_idx].id
        return songs[prev_idx]

    def first_song(self) -> Optional[Song]:
        songs = self.active_setlist.songs
        if not songs:
            return None
        self.active_setlist.active_song_id = songs[0].id
        return songs[0]

    def last_song(self) -> Optional[Song]:
        songs = self.active_setlist.songs
        if not songs:
            return None
        self.active_setlist.active_song_id = songs[-1].id
        return songs[-1]

    # ============================================================
    # Persistência Local de Setlist
    # ============================================================
    def save_setlist_to_file(self, file_path: str, setlist_id: Optional[str] = None) -> str:
        target = self._setlists.get(setlist_id) if setlist_id else self.active_setlist
        abs_path = os.path.abspath(file_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(target.to_dict(), f, indent=4, ensure_ascii=False)
        return abs_path

    def load_setlist_from_file(self, file_path: str) -> Setlist:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded = Setlist.from_dict(data)
        self._setlists[loaded.id] = loaded
        self._active_setlist_id = loaded.id
        return loaded
