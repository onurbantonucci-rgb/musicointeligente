"""Gerador de áudio sintético padronizado para testes controlados de DSP e reprodução.

Gera a progressão clássica: C -> G -> Am -> F a 120 BPM com cliques de metrônomo.
"""

import os
import numpy as np
import soundfile as sf


def generate_tone(freq: float, duration: float, sr: int = 44100, harmonics: int = 4) -> np.ndarray:
    """Gera um tom harmônico rico com envelope ADSR suave para evitar estalos (clicks)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)

    # Adicionar fundamental e harmônicos com decaimento natural
    for h in range(1, harmonics + 1):
        amp = 1.0 / (h ** 1.2)
        signal += amp * np.sin(2 * np.pi * freq * h * t)

    # Normalizar sinal bruto
    if np.max(np.abs(signal)) > 0:
        signal /= np.max(np.abs(signal))

    # Envelope ADSR simplificado (fade in 20ms, fade out 40ms)
    fade_in_len = min(int(sr * 0.02), len(signal) // 4)
    fade_out_len = min(int(sr * 0.04), len(signal) // 4)

    fade_in = np.linspace(0, 1, fade_in_len)
    fade_out = np.linspace(1, 0, fade_out_len)

    signal[:fade_in_len] *= fade_in
    signal[-fade_out_len:] *= fade_out

    return signal.astype(np.float32)


def generate_metronome_click(duration: float = 0.03, sr: int = 44100, freq: float = 1200.0) -> np.ndarray:
    """Gera um pulso curto percussivo para o clique do metrônomo."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    click = np.sin(2 * np.pi * freq * t) * np.exp(-t * 120.0)
    return click.astype(np.float32)


def generate_chord(notes_hz: list, duration: float, sr: int = 44100) -> np.ndarray:
    """Sintetiza um acorde somando as notas individuais."""
    chord = np.zeros(int(sr * duration), dtype=np.float32)
    for freq in notes_hz:
        chord += generate_tone(freq, duration, sr=sr, harmonics=3)
    if np.max(np.abs(chord)) > 0:
        chord /= np.max(np.abs(chord))
    return chord


def generate_test_song(output_path: str = "test_song_120bpm.wav", sr: int = 44100, bpm: float = 120.0, total_cycles: int = 2) -> str:
    """Gera arquivo WAV estéreo com a progressão C -> G -> Am -> F a 120 BPM e metrônomo.
    
    Canal Esquerdo: Acordes harmônicos
    Canal Direito: Acordes + Metrônomo
    """
    beat_duration = 60.0 / bpm  # 0.5s a 120 BPM
    chord_duration = beat_duration * 4.0  # 1 compasso de 4/4 por acorde (2.0s)

    # Frequências em Hz das notas dos acordes
    # C4: C (261.63), E (329.63), G (392.00)
    # G3: G (196.00), B (246.94), D (293.66)
    # Am3: A (220.00), C (261.63), E (329.63)
    # F3: F (174.61), A (220.00), C (261.63)
    progression = [
        ("C", [261.63, 329.63, 392.00]),
        ("G", [196.00, 246.94, 293.66]),
        ("Am", [220.00, 261.63, 329.63]),
        ("F", [174.61, 220.00, 261.63]),
    ]

    total_measures = total_cycles  # Número de repetições da progressão (cada ciclo = 4 compassos = 8s)
    full_audio_chords = []
    # Cada acorde tem int(sr * chord_duration) amostras (truncado por acorde, como em
    # generate_chord); o buffer de cliques precisa da MESMA soma, senão BPMs que não
    # dividem o sample rate exato (ex.: 92, 138) geram buffers de tamanhos diferentes.
    samples_per_chord = int(sr * chord_duration)
    full_clicks = np.zeros(samples_per_chord * len(progression) * total_measures, dtype=np.float32)

    sample_idx = 0
    for _ in range(total_measures):
        for chord_name, freqs in progression:
            # Gerar acorde de 1 compasso
            chord_audio = generate_chord(freqs, chord_duration, sr=sr)
            full_audio_chords.append(chord_audio)

            # Inserir 4 cliques de metrônomo no compasso
            for beat in range(4):
                click_freq = 1600.0 if beat == 0 else 1000.0  # Tom mais alto no tempo 1
                click = generate_metronome_click(duration=0.025, sr=sr, freq=click_freq) * 0.7
                start_sample = sample_idx + int(beat * beat_duration * sr)
                end_sample = min(start_sample + len(click), len(full_clicks))
                full_clicks[start_sample:end_sample] += click[:end_sample - start_sample]

            sample_idx += len(chord_audio)

    harmonic_audio = np.concatenate(full_audio_chords)

    # Montar sinal estéreo: L com harmonia pura, R com harmonia + metrônomo sutil
    left_channel = harmonic_audio * 0.75
    right_channel = (harmonic_audio * 0.65) + (full_clicks * 0.35)

    # Normalizar
    stereo = np.column_stack([left_channel, right_channel]).astype(np.float32)
    max_amp = np.max(np.abs(stereo))
    if max_amp > 0:
        stereo /= max_amp

    abs_path = os.path.abspath(output_path)
    sf.write(abs_path, stereo, sr, subtype="PCM_16")
    print(f"[AudioGenerator] Arquivo de teste gerado com sucesso: {abs_path}")
    return abs_path


def generate_plucked_string(freq: float, duration: float, sr: int = 44100, plectrum_noise: float = 0.15) -> np.ndarray:
    """Modela o timbre acústico de uma corda de violão dedilhada ou palhetada.
    
    Combina ataque percussivo de palheta/unha, decaimento diferencial de harmônicos
    e ressonância acústica do tampo de madeira.
    """
    total_samples = int(sr * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)
    string_sound = np.zeros(total_samples, dtype=np.float32)

    # Série harmônica da corda de violão (com decaimento acelerado nas frequências agudas)
    harmonics = 8
    for h in range(1, harmonics + 1):
        # Damping dependente da frequência (harmônicos agudos morrem antes)
        decay_rate = 1.8 + (h * 1.5)
        decay = np.exp(-decay_rate * t)
        h_amp = (1.0 / (h ** 0.95))
        string_sound += h_amp * decay * np.sin(2 * np.pi * freq * h * t)

    # Ruído percussivo de ataque da palheta (burst de ruído passa-alta nos primeiros 15ms)
    noise_samples = min(int(sr * 0.018), total_samples)
    noise_env = np.linspace(1.0, 0.0, noise_samples) ** 2
    noise = np.random.uniform(-1.0, 1.0, noise_samples) * noise_env * plectrum_noise
    string_sound[:noise_samples] += noise

    # Normalizar corda
    max_amp = np.max(np.abs(string_sound))
    if max_amp > 0:
        string_sound /= max_amp

    return string_sound.astype(np.float32)


def generate_acoustic_guitar_chord(notes_hz: list, duration: float, sr: int = 44100, strum_delay_ms: float = 16.0) -> np.ndarray:
    """Simula o 'rasqueado' ou batida de violão, onde as cordas são tocadas sucessivamente."""
    total_samples = int(sr * duration)
    chord_audio = np.zeros(total_samples, dtype=np.float32)
    delay_samples = int((strum_delay_ms / 1000.0) * sr)

    for i, freq in enumerate(notes_hz):
        string_audio = generate_plucked_string(freq, duration, sr=sr)
        start_idx = i * delay_samples
        if start_idx < total_samples:
            available = min(len(string_audio), total_samples - start_idx)
            chord_audio[start_idx:start_idx + available] += string_audio[:available]

    max_amp = np.max(np.abs(chord_audio))
    if max_amp > 0:
        chord_audio /= max_amp

    return chord_audio


def generate_acoustic_guitar_sample(output_path: str = "violao_teste_120bpm.wav", sr: int = 44100, bpm: float = 120.0, total_cycles: int = 2) -> str:
    """Gera faixa realista de Violão Acústico executando a progressão C -> G -> Am -> F a 120 BPM.
    
    Contém afinações reais das cordas de violão (voicings abertos populares):
      - C (C3, E3, G3, C4, E4)
      - G (G2, B2, D3, G3, D4, G4)
      - Am (A2, E3, A3, C4, E4)
      - F (F2, C3, F3, A3, C4)
    """
    beat_sec = 60.0 / bpm
    bar_sec = beat_sec * 4.0  # 2.0s por acorde

    # Voicings reais de violão acústico (frequências em Hz das cordas tocadas)
    guitar_voicings = [
        ("C", [130.81, 164.81, 196.00, 261.63, 329.63]),           # C3, E3, G3, C4, E4
        ("G", [98.00, 123.47, 146.83, 196.00, 293.66, 392.00]),    # G2, B2, D3, G3, D4, G4
        ("Am", [110.00, 164.81, 220.00, 261.63, 329.63]),         # A2, E3, A3, C4, E4
        ("F", [87.31, 130.81, 174.61, 220.00, 261.63]),           # F2, C3, F3, A3, C4
    ]

    total_bars = total_cycles  # Cada ciclo = 4 acordes de 2s = 8s
    all_chunks = []

    for _ in range(total_bars):
        for name, freqs in guitar_voicings:
            # 4 batidas por compasso (estilo levada de violão: batidas nos 4 tempos do compasso a 120 BPM)
            bar_parts = []
            for b in range(4):
                dur = beat_sec
                # Variação de dinâmica e direção de palhetada (tempo forte no 1 e 3)
                vol = 1.0 if b in [0, 2] else 0.75
                strum = generate_acoustic_guitar_chord(freqs, duration=dur, sr=sr, strum_delay_ms=14.0 + (b * 2)) * vol
                bar_parts.append(strum)
            bar_audio = np.concatenate(bar_parts)
            all_chunks.append(bar_audio)

    full_audio = np.concatenate(all_chunks).astype(np.float32)

    # Leve espacialização estéreo (reverberação acústica de corpo do violão)
    left = full_audio * 0.90
    right = np.roll(full_audio, int(sr * 0.008)) * 0.85  # 8ms Haas delay

    stereo = np.column_stack([left, right]).astype(np.float32)
    max_amp = np.max(np.abs(stereo))
    if max_amp > 0:
        stereo /= max_amp

    abs_path = os.path.abspath(output_path)
    sf.write(abs_path, stereo, sr, subtype="PCM_16")
    print(f"[AudioGenerator] Áudio de violão acústico gerado: {abs_path}")
    return abs_path


def generate_modulating_song(
    output_path: str = "test_modulation_g_to_a.wav",
    sr: int = 44100,
    bpm: float = 120.0,
    cycles_per_key: int = 2
) -> str:
    """Gera um áudio de teste com modulação explícita: G Major (16s) -> A Major (16s).
    
    Parte 1: G -> Em -> C -> D (G Major)
    Parte 2: A -> F#m -> D -> E (A Major)
    """
    beat_dur = 60.0 / bpm
    chord_dur = beat_dur * 4.0  # 2.0s por acorde

    # Seção 1: G Major
    g_chords = [
        ("G", [196.00, 246.94, 293.66]),       # G, B, D
        ("Em", [164.81, 196.00, 246.94]),      # E, G, B
        ("C", [261.63, 329.63, 392.00]),       # C, E, G
        ("D", [146.83, 185.00, 220.00]),       # D, F#, A
    ]

    # Seção 2: A Major
    a_chords = [
        ("A", [220.00, 277.18, 329.63]),       # A, C#, E
        ("F#m", [185.00, 220.00, 277.18]),     # F#, A, C#
        ("D", [146.83, 185.00, 220.00]),       # D, F#, A
        ("E", [164.81, 207.65, 246.94]),       # E, G#, B
    ]


    all_audio = []

    # Gerar ciclos em G Major
    for _ in range(cycles_per_key):
        for name, freqs in g_chords:
            all_audio.append(generate_chord(freqs, chord_dur, sr=sr))

    # Gerar ciclos em A Major
    for _ in range(cycles_per_key):
        for name, freqs in a_chords:
            all_audio.append(generate_chord(freqs, chord_dur, sr=sr))

    mono = np.concatenate(all_audio).astype(np.float32)
    stereo = np.column_stack([mono * 0.8, mono * 0.8]).astype(np.float32)

    abs_path = os.path.abspath(output_path)
    sf.write(abs_path, stereo, sr, subtype="PCM_16")
    print(f"[AudioGenerator] Áudio modulante gerado: {abs_path}")
    return abs_path


if __name__ == "__main__":
    generate_test_song("test_song_120bpm.wav")
    generate_acoustic_guitar_sample("violao_teste_120bpm.wav")


