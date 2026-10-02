"""Leitor mínimo de arquivos MIDI (Standard MIDI File, formatos 0 e 1).

Só o necessário para montar o banco de levadas de baixo: notas (início/fim em
ticks, altura, intensidade, canal, programa), mapa de andamento e fórmulas de
compasso. Sem dependências externas.
"""

from dataclasses import dataclass, field
import struct
from typing import Dict, List, Optional, Tuple


@dataclass
class MidiNote:
    start: int
    end: int
    pitch: int
    velocity: int
    channel: int
    program: int
    track: int


@dataclass
class MidiTrack:
    index: int
    name: str = ""
    notes: List[MidiNote] = field(default_factory=list)


@dataclass
class MidiSong:
    ticks_per_beat: int
    tracks: List[MidiTrack]
    tempos: List[Tuple[int, float]]              # (tick, BPM)
    time_signatures: List[Tuple[int, int, int]]  # (tick, numerador, denominador)

    def bpm_at(self, tick: int) -> float:
        bpm = 120.0
        for at, value in self.tempos:
            if at <= tick:
                bpm = value
            else:
                break
        return bpm


def _read_varlen(data: bytes, pos: int) -> Tuple[int, int]:
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, pos


def read_midi(path: str) -> Optional[MidiSong]:
    """Lê o arquivo; retorna None se não for um MIDI válido/suportado."""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError:
        return None
    if data[:4] != b"MThd" or len(data) < 14:
        return None
    header_len = struct.unpack(">I", data[4:8])[0]
    fmt, ntracks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000 or fmt > 1:
        return None                      # SMPTE ou formato 2: não usados no banco
    pos = 8 + header_len
    tracks: List[MidiTrack] = []
    tempos: List[Tuple[int, float]] = []
    signatures: List[Tuple[int, int, int]] = []
    for track_index in range(ntracks):
        if data[pos:pos + 4] != b"MTrk":
            break
        length = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 8 + length
        track = MidiTrack(index=track_index)
        try:
            _parse_track(chunk, track, tempos, signatures)
        except (IndexError, struct.error):
            pass                         # trilha truncada: mantém o que foi lido
        tracks.append(track)
    tempos.sort()
    signatures.sort()
    return MidiSong(ticks_per_beat=division, tracks=tracks, tempos=tempos,
                    time_signatures=signatures)


def _parse_track(chunk: bytes, track: MidiTrack, tempos, signatures) -> None:
    pos, tick, status = 0, 0, 0
    programs: Dict[int, int] = {}
    pending: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    while pos < len(chunk):
        delta, pos = _read_varlen(chunk, pos)
        tick += delta
        byte = chunk[pos]
        if byte & 0x80:
            status = byte
            pos += 1
        if status == 0xFF:
            meta = chunk[pos]
            length, pos = _read_varlen(chunk, pos + 1)
            payload = chunk[pos:pos + length]
            pos += length
            if meta == 0x51 and length == 3:
                mpqn = (payload[0] << 16) | (payload[1] << 8) | payload[2]
                if mpqn > 0:
                    tempos.append((tick, 60_000_000.0 / mpqn))
            elif meta == 0x58 and length >= 2:
                signatures.append((tick, payload[0], 2 ** payload[1]))
            elif meta == 0x03:
                track.name = payload.decode("latin-1", errors="ignore").strip()
            elif meta == 0x2F:
                break
            continue
        if status in (0xF0, 0xF7):
            length, pos = _read_varlen(chunk, pos)
            pos += length
            continue
        kind, channel = status & 0xF0, status & 0x0F
        if kind in (0xC0, 0xD0):
            value = chunk[pos]
            pos += 1
            if kind == 0xC0:
                programs[channel] = value
            continue
        first, second = chunk[pos], chunk[pos + 1]
        pos += 2
        if kind == 0x90 and second > 0:
            pending.setdefault((channel, first), []).append((tick, second))
        elif kind == 0x80 or (kind == 0x90 and second == 0):
            starts = pending.get((channel, first))
            if starts:
                start, velocity = starts.pop(0)
                if tick > start:
                    track.notes.append(MidiNote(start, tick, first, velocity, channel,
                                                programs.get(channel, 0), track.index))
