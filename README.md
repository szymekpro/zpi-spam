# zpi-spam

## Instalacja

```bash
pip install -r requirements.txt
```

## Aplikacja konsolowa

```bash
python run_experiment.py [opcje]
```

| Opcja | Opis | Status |
|---|---|---|
| `--help` | lista opcji | |
| `--preprocess-only` | tylko preprocessing: dane, usunięcie duplikatów, podział 80/20, TF-IDF | |
| `--plain-only` | część jawna: preprocessing, Multinomial Naive Bayes, dobór `alpha` i progu, metryki na zbiorze testowym | |
| `--quick` | szybki przebieg próbny (klucz 1024 bity, 3 SMS) | not implemented |
| `--samples N` | liczba SMS w eksperymencie szyfrowanym | not implemented |
| `--key-bits N` | długość klucza Pailliera w bitach | not implemented |
| `--output DIR` | katalog na wyniki | not implemented |
| bez opcji | pełny eksperyment | not implemented |
