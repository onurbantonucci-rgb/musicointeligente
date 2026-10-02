# Banco de levadas de baixo — atribuição

`bass_bank_gospel_worship.npz` foi gerado por `tools/bass_bank/build_bank.py` a partir do
**Lakh MIDI Dataset** (subconjunto `clean_midi`), de Colin Raffel, distribuído sob a licença
**Creative Commons Attribution 4.0 International (CC-BY 4.0)**:
https://colinraffel.com/projects/lmd/

> Colin Raffel. *Learning-Based Methods for Comparing Sequences, with Applications to
> Audio-to-MIDI Alignment and Matching.* PhD Thesis, 2016.

O arquivo guarda apenas padrões rítmicos/funcionais de compassos de baixo (semicolcheia,
duração, intensidade e função harmônica da nota) e o contexto rítmico da banda — não guarda
melodias, letras nem as músicas originais. Os nomes dos arquivos de origem são mantidos só
para mostrar de onde veio cada linha.

Para regenerar (dados brutos fora do repositório):

    python -m tools.bass_bank.build_bank "<pasta clean_midi>" app/instruments/data/bass_bank_gospel_worship.npz GOSPEL_WORSHIP
