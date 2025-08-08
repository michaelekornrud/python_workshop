# 01 — Pythonic thinking

Mål
- Øve på idiomatisk Python: collections, comprehensions, unpacking, EAFP, truthiness

Gjør dette
- Implementer funksjonene i `pythonic_thinking.py` slik at testene passerer
- Kjør `pytest -q tests/test_01_pythonic.py`

Oppgaver
- normalize_whitespace: kollaps whitespace til enkelt mellomrom
- unique_preserve_order: behold første forekomst, bevar rekkefølge
- pairwise_sum: summer parvis, trunkér til korteste sekvens
- transpose: transponer rektangulær matrise
- head_tail: pakk ut første element og resten
- safe_get: EAFP nested-oppslag i dict/list
- flatten_once: flat ut ett nivå

Tips
- Bruk `str.split()`/`' '.join(...)` for whitespace
- Bruk `zip`, `zip(*matrix)`, `enumerate`, `set` for medlemskap
- Sekvens-unpacking: `head, *tail = seq`
- EAFP: `try/except` framfor `if`-kjeder

Se `xample.py` for hint om du står fast.