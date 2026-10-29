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
| `--encrypt-only` | etap II: generowanie kluczy Pailliera (`phe`), szyfrowanie wektorów TF-IDF po stronie klienta (kodowanie stałoprzecinkowe), kontrola odszyfrowania | |
| `--quick` | szybki przebieg próbny (klucz 1024 bity, 3 SMS) | |
| `--samples N` | liczba SMS w eksperymencie szyfrowanym | |
| `--key-bits N` | długość klucza Pailliera w bitach | |
| `--output DIR` | katalog na wyniki | not implemented |
| bez opcji | pełny eksperyment | not implemented |
