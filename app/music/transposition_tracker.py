"""Rastreador de Transposição / Capotraste (TranspositionTracker v0.5).

Detecta automaticamente quando o instrumento real está sendo tocado deslocado de um
número constante de semitons em relação à cifra — o caso típico de uso de CAPOTRASTE ou
de uma cifra escrita em um tom diferente do executado.

Princípio: se o áudio detecta, de forma consistente, acordes que são sempre a mesma
distância em semitons dos acordes esperados pela cifra (ex.: cifra diz C, G, Am e o áudio
ouve D, A, Bm = +2 semitons), então há uma transposição global, e NÃO uma sequência de
erros harmônicos. Ao reconhecer isso, o sistema para de reportar falsas divergências e
pode informar o deslocamento aos músicos virtuais.
"""

from collections import deque, Counter

from app.music.theory import note_to_pc


class TranspositionTracker:
    """Infere um deslocamento global de semitons entre a cifra e a execução."""

    def __init__(self, window: int = 8, min_observations: int = 3, min_ratio: float = 0.6):
        # Janela de observações recentes (delta de semitons por acorde confiável)
        self._deltas: deque = deque(maxlen=window)
        self._min_observations = min_observations
        self._min_ratio = min_ratio
        self._offset: int = 0  # Deslocamento consolidado atual (0 = sem transposição)

    def reset(self) -> None:
        """Zera o histórico e o deslocamento consolidado."""
        self._deltas.clear()
        self._offset = 0

    @property
    def semitones(self) -> int:
        """Deslocamento global consolidado em semitons (0 se não há transposição estável)."""
        return self._offset

    @property
    def is_transposed(self) -> bool:
        return self._offset != 0

    def observe(self, expected_chord: str, detected_chord: str) -> int:
        """Registra um par (esperado, detectado) confiável e reconsolida o deslocamento.

        Retorna o deslocamento global atual em semitons.
        """
        exp_pc = note_to_pc(expected_chord)
        det_pc = note_to_pc(detected_chord)
        if exp_pc < 0 or det_pc < 0:
            return self._offset

        delta = (det_pc - exp_pc) % 12
        self._deltas.append(delta)
        self._consolidate()
        return self._offset

    def _consolidate(self) -> None:
        """Reavalia se há um deslocamento dominante e consistente na janela recente."""
        if len(self._deltas) < self._min_observations:
            self._offset = 0
            return

        counts = Counter(self._deltas)
        best_delta, best_count = counts.most_common(1)[0]
        ratio = best_count / len(self._deltas)

        # Só assume transposição quando um mesmo deslocamento NÃO-zero domina a janela.
        if best_delta != 0 and best_count >= self._min_observations and ratio >= self._min_ratio:
            # Normaliza para o intervalo mais próximo de zero: +7 é mais natural como -5, etc.
            self._offset = best_delta - 12 if best_delta > 6 else best_delta
        else:
            self._offset = 0

    def matches_under_transposition(self, expected_chord: str, detected_chord: str) -> bool:
        """Indica se o acorde detectado corresponde ao esperado sob o deslocamento atual."""
        if self._offset == 0:
            return False
        exp_pc = note_to_pc(expected_chord)
        det_pc = note_to_pc(detected_chord)
        if exp_pc < 0 or det_pc < 0:
            return False
        return (exp_pc + self._offset) % 12 == det_pc % 12
