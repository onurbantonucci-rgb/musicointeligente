"""Gerenciador Central de Projeto, Persistência Atômica e Ciclo de Sessões (ProjectManager v0.4).

Controla o salvamento e carregamento de projetos em JSON, importação de músicas,
orquestração de setlists e transição atômica entre SongSessions sem vazamento de estado.
"""

from typing import Optional, List, Any, Tuple
import json
import os
import tempfile
from datetime import datetime

from app.project.project import Project
from app.setlist.setlist_manager import SetlistManager
from app.song.song import Song
from app.song.song_session import SongSession
from app.input.chart_parser import ChartParser


DEFAULT_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "virtual_band_project.json"))

# Marca as músicas cujas passagens só de acordes já viraram blocos próprios — a separação
# roda uma única vez por música, para não desfazer uma junção feita depois pelo usuário.
INSTRUMENTAL_SPLIT_FLAG = "chord_only_passages_split"


class ProjectManager:
    """Gerenciador global da aplicação para projetos, setlists e sessões musicais."""

    def __init__(self, project_file_path: Optional[str] = None):
        self._project_file_path = os.path.abspath(project_file_path) if project_file_path else DEFAULT_PROJECT_PATH
        self._project: Project = Project()
        self._setlist_manager: SetlistManager = SetlistManager()
        self._active_session: Optional[SongSession] = None

        if project_file_path and os.path.exists(project_file_path) and os.path.getsize(project_file_path) > 0:
            self.load_project(project_file_path)
        else:
            self._sync_setlist_manager()

    @property
    def project(self) -> Project:
        return self._project

    def get_project(self) -> Project:
        """Retorna a instância do projeto gerenciado."""
        return self._project

    @property
    def setlist_manager(self) -> SetlistManager:
        return self._setlist_manager

    @property
    def active_session(self) -> Optional[SongSession]:
        return self._active_session

    @property
    def project_file_path(self) -> str:
        return self._project_file_path

    @project_file_path.setter
    def project_file_path(self, path: str) -> None:
        self._project_file_path = os.path.abspath(path)

    def _sync_setlist_manager(self) -> None:
        """Sincroniza o SetlistManager interno com os setlists do projeto ativo."""
        active_set = self._project.get_active_setlist()
        self._setlist_manager = SetlistManager(active_set)
        for s in self._project.setlists:
            self._setlist_manager._setlists[s.id] = s
        self._setlist_manager._active_setlist_id = active_set.id

    def create_project(self, name: str, description: str = "") -> Project:
        """Inicia um novo projeto vazio com um setlist padrão."""
        if self._active_session:
            self._active_session.close()
            self._active_session = None

        self._project = Project(name=name, description=description)
        self._sync_setlist_manager()
        return self._project

    # ============================================================
    # Troca Hermética de Música (Zero Vazamento de Estado)
    # ============================================================
    def open_song(self, song: Song, mode: str = "PLAYBACK") -> SongSession:
        """Encerra com segurança a sessão anterior e inicializa uma nova SongSession isolada."""
        if self._active_session is not None:
            # 1. Encerra a sessão anterior
            self._active_session.close()
            self._active_session = None

        # 2. Registra como música ativa no setlist corrente
        active_set = self._project.get_active_setlist()
        active_set.active_song_id = song.id

        # 3. Inicializa nova SongSession limpa
        self._active_session = SongSession(song=song, mode=mode)
        return self._active_session

    def open_song_by_id(self, song_id: str, mode: str = "PLAYBACK") -> Optional[SongSession]:
        """Abre uma música do setlist ativo pelo seu ID."""
        active_set = self._project.get_active_setlist()
        song = active_set.get_song(song_id)
        if song:
            return self.open_song(song, mode=mode)
        return None

    def remove_songs_from_active_setlist(self, song_ids: List[str]) -> List[Song]:
        """Remove entradas selecionadas da setlist sem apagar seus arquivos de origem."""
        active_set = self._project.get_active_setlist()
        selected = set(song_ids)
        removed = [song for song in active_set.songs if song.id in selected]
        if not removed:
            return []
        old_active_id = active_set.active_song_id
        old_songs = list(active_set.songs)
        old_active_index = next((index for index, song in enumerate(active_set.songs)
                                 if song.id == old_active_id), 0)
        active_set.songs = [song for song in active_set.songs if song.id not in selected]
        if old_active_id in selected:
            next_index = min(old_active_index, len(active_set.songs) - 1)
            active_set.active_song_id = (active_set.songs[next_index].id
                                         if active_set.songs else None)
        if self._project.settings.auto_save:
            try:
                self.save_project()
            except Exception:
                active_set.songs = old_songs
                active_set.active_song_id = old_active_id
                raise
        if (self._active_session is not None and
                self._active_session.song.id in selected):
            self._active_session.close()
            self._active_session = None
        return removed

    # ============================================================
    # Importação Flexível de Músicas
    # ============================================================
    def import_song(
        self,
        title: str = "Nova Canção",
        artist: str = "Artista",
        audio_path: str = "",
        chart_text: str = "",
        key: str = "C Major",
        bpm: float = 120.0,
        meter: str = "4/4",
        duration: float = 0.0,
        auto_add_to_setlist: bool = True
    ) -> Song:
        """Importa uma música aceitando dados parciais (só áudio, só cifra, ou ambos)."""
        from app.music.sectionizer import split_chord_only_passages

        chart_data = None
        if chart_text.strip():
            chart_text = split_chord_only_passages(chart_text)
            parsed_chart = ChartParser.parse(
                text=chart_text,
                default_key=key,
                default_bpm=bpm,
                default_meter=meter,
                title=title,
                artist=artist
            )
            chart_data = parsed_chart.to_dict()
            key = parsed_chart.key
            bpm = parsed_chart.bpm
            meter = parsed_chart.meter

        new_song = Song(
            title=title,
            artist=artist,
            audio_path=audio_path,
            duration=duration,
            key=key,
            bpm=bpm,
            meter=meter,
            chart_text=chart_text,
            chart_data=chart_data,
            lyrics_text=chart_text if not chart_data else "\n".join(
                l.get("text", "") if isinstance(l, dict) else getattr(l, "text", str(l))
                for s in chart_data.get("sections", []) for l in s.get("lyrics", [])
            )
        )

        new_song.metadata[INSTRUMENTAL_SPLIT_FLAG] = True

        if auto_add_to_setlist:
            self._project.get_active_setlist().add_song(new_song)
            self._sync_setlist_manager()

        return new_song

    def import_from_source(
        self,
        source_input: Any,
        title: str = "Nova Canção",
        artist: str = "Artista",
        audio_path: str = "",
        key: str = "C Major",
        bpm: float = 120.0,
        meter: str = "4/4",
        duration: float = 0.0,
        auto_add_to_setlist: bool = True,
        auto_detect_sections: bool = True
    ) -> Tuple[Song, Any]:
        """Importa uma música a partir de arquivo TXT, DOCX, PDF, Imagem (PNG/JPG/WEBP) ou texto puro.

        Se ``auto_detect_sections`` e a cifra NÃO trouxer partes marcadas, o programa
        identifica e rotula automaticamente Intro/Verso/Refrão/Ponte etc.
        """
        from app.input.chart_sources import TextChartSource, DocxChartSource, ImageChartSource, PdfChartSource, RawChartDocument
        from app.music.sectionizer import auto_sectionize_text, text_has_section_headers

        doc: Optional[RawChartDocument] = None

        if isinstance(source_input, RawChartDocument):
            doc = source_input
        elif isinstance(source_input, str) and os.path.isfile(source_input):
            ext = os.path.splitext(source_input)[1].lower()
            if ext == ".docx":
                doc = DocxChartSource().load(source_input)
            elif ext == ".pdf":
                doc = PdfChartSource().load(source_input)
            elif ext in (".png", ".jpg", ".jpeg", ".webp"):
                doc = ImageChartSource().load(source_input)
            else:
                doc = TextChartSource().load(source_input)
            if title == "Nova Canção":
                title = os.path.splitext(os.path.basename(source_input))[0]
        else:
            doc = TextChartSource().load(str(source_input))

        chart_text = doc.text
        section_types_detected = False
        if auto_detect_sections and chart_text and not text_has_section_headers(chart_text):
            chart_text = auto_sectionize_text(chart_text)
            section_types_detected = True

        song = self.import_song(
            title=title,
            artist=artist,
            audio_path=audio_path,
            chart_text=chart_text,
            key=key,
            bpm=bpm,
            meter=meter,
            duration=duration,
            auto_add_to_setlist=auto_add_to_setlist
        )
        if doc.metadata:
            song.metadata.update(doc.metadata)
        song.metadata["chart_source_type"] = doc.source_type
        song.metadata["chart_overall_confidence"] = doc.overall_confidence
        song.metadata["low_confidence_count"] = len(doc.low_confidence_tokens)
        song.metadata["sections_auto_detected"] = section_types_detected

        return song, doc

    def transpose_song(self, song: Song, target_key: str) -> Song:
        """Transpõe a cifra de uma música para ``target_key`` e persiste a mudança.

        Se a música estiver na sessão ativa, usa o caminho da sessão (que também reconstrói
        alinhamento/estimador); caso contrário, transpõe diretamente o chart_data/chart_text.
        """
        from app.music.chord_chart import ChordChart

        if self._active_session and self._active_session.song.id == song.id:
            self._active_session.transpose_to(target_key)
        else:
            if song.chart_data:
                chart = ChordChart.from_dict(song.chart_data)
            elif song.chart_text:
                chart = ChartParser.parse(text=song.chart_text, default_key=song.key,
                                          title=song.title, artist=song.artist)
            else:
                return song
            chart.transpose_to_key(target_key)
            song.chart_data = chart.to_dict()
            if song.chart_text:
                song.chart_text = chart.raw_text
            song.key = target_key

        song.update_timestamp()
        if self._project.settings.auto_save:
            self.save_project()
        return song

    def update_song_chart(self, song_id: str, new_chart_text: str) -> Optional[Song]:
        """Atualiza a cifra de uma música no projeto e propaga para a sessão de execução ativa."""
        song = self._project.get_song_by_id(song_id)
        if not song:
            active_song = self._project.get_active_setlist().get_active_song()
            if active_song and (active_song.id == song_id or not song_id):
                song = active_song
            else:
                return None

        self._apply_chart_text(song, new_chart_text)

        if self._active_session and self._active_session.song.id == song.id:
            self._active_session.update_chart_text(new_chart_text)

        if self._project.settings.auto_save:
            self.save_project()

        return song

    @staticmethod
    def _apply_chart_text(song: Song, new_chart_text: str) -> None:
        """Grava o texto na música e recalcula cifra, letra, tom, BPM e compasso."""
        parsed_chart = ChartParser.parse(
            text=new_chart_text,
            default_key=song.key,
            default_bpm=song.bpm,
            default_meter=song.meter,
            title=song.title,
            artist=song.artist
        )
        song.chart_text = new_chart_text
        song.chart_data = parsed_chart.to_dict()
        song.lyrics_text = "\n".join(
            l.get("text", "") if isinstance(l, dict) else getattr(l, "text", str(l))
            for s in song.chart_data.get("sections", []) for l in s.get("lyrics", [])
        )
        if parsed_chart.key and parsed_chart.key != song.key:
            song.key = parsed_chart.key
        if parsed_chart.bpm and parsed_chart.bpm != song.bpm:
            song.bpm = parsed_chart.bpm
        if parsed_chart.meter and parsed_chart.meter != song.meter:
            song.meter = parsed_chart.meter
            song.time_signature = parsed_chart.meter

        song.update_timestamp()

    def _split_chord_only_passages_once(self) -> None:
        """Músicas importadas antes da separação ganham os blocos [Reintro]/[Instrumental]."""
        from app.music.sectionizer import split_chord_only_passages

        for setlist in self._project.setlists:
            for song in setlist.songs:
                if song.metadata.get(INSTRUMENTAL_SPLIT_FLAG):
                    continue
                song.metadata[INSTRUMENTAL_SPLIT_FLAG] = True
                if not (song.chart_text or "").strip():
                    continue
                new_text = split_chord_only_passages(song.chart_text)
                if new_text != song.chart_text:
                    self._apply_chart_text(song, new_text)

    # ============================================================
    # Persistência Atômica Local (JSON)
    # ============================================================
    def save_project(self, target_path: Optional[str] = None) -> str:
        """Salva o projeto em disco usando escrita atômica segura para evitar corrupção."""
        dest_path = os.path.abspath(target_path or self._project_file_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        self._project.updated_at = datetime.now().isoformat()
        data = self._project.to_dict()

        dir_name = os.path.dirname(dest_path)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
                temp_name = tf.name
                json.dump(data, tf, indent=4, ensure_ascii=False)
                tf.flush()
                os.fsync(tf.fileno())
            # Mantém a versão anterior intacta se a substituição falhar.
            os.replace(temp_name, dest_path)
        finally:
            if temp_name is not None and os.path.exists(temp_name):
                os.unlink(temp_name)
        self._project_file_path = dest_path
        return dest_path

    def load_project(self, source_path: str) -> Project:
        """Carrega e valida um projeto salvo localmente em arquivo JSON."""
        abs_path = os.path.abspath(source_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Arquivo de projeto não encontrado: {abs_path}")

        if self._active_session:
            self._active_session.close()
            self._active_session = None

        with open(abs_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._project = Project.from_dict(data)
        self._project_file_path = abs_path
        self._split_chord_only_passages_once()
        self._sync_setlist_manager()

        # Se houver música ativa, inicializa a sessão
        active_song = self._project.get_active_setlist().get_active_song()
        if active_song:
            self.open_song(active_song)

        return self._project
