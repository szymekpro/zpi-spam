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
| `--sensitivity` | analiza wrażliwości: wpływ liczby słów TF-IDF (100/200/300/500) i parametru `alpha` (0.01–5) na recall, F1 i ROC-AUC dla wielu podziałów danych; zapisuje `wrazliwosc_*.png` i `wrazliwosc_*.csv` | |
| `--seeds N` | liczba podziałów danych w analizie wrażliwości (domyślnie 10) | |
| `--quick` | szybki przebieg próbny (klucz 1024 bity, 3 SMS; do demonstracji) | |
| `--samples N` | liczba SMS w eksperymencie szyfrowanym | |
| `--key-bits N` | długość klucza Pailliera w bitach | |
| `--output DIR` | katalog na wyniki (domyślnie `wyniki`) | |
| bez opcji | etapy I–IV: trening, szyfrowanie, klasyfikacja i porównanie wyników oraz czasów (domyślnie 20 SMS, klucz 2048 bitów) | |

## Przykłady

```bash
python run_experiment.py --quick
python run_experiment.py --samples 5 --key-bits 2048
python run_experiment.py --sensitivity --seeds 10
```

## Testy

```bash
python -m unittest discover -s tests -v
```
