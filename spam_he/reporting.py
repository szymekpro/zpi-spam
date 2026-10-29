from pathlib import Path

import pandas as pd

from spam_he.config import Config


def plot_class_distribution(class_counts: pd.Series, path: Path) -> None:
    raise NotImplementedError


def plot_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    raise NotImplementedError


def plot_times(mean_times: dict[str, float], path: Path) -> None:
    raise NotImplementedError


def requirement_checks(results: dict, config: Config) -> list[dict]:
    summary = results["summary"]
    times = summary["mean_times_s"]
    agreement_ok = summary["agreement"] >= config.target_agreement
    slower = times["encrypted_total_s"] > times["plain_s"]
    return [
        {"name": f"Recall spamu na całym teście >= {config.target_recall:.0%}",
         "passed": results["plain_metrics"]["recall_spam"] >= config.target_recall},
        {"name": f"Zgodność decyzji na próbce >= {config.target_agreement:.0%}",
         "passed": agreement_ok},
        {"name": "Wariant szyfrowany trwa dłużej niż jawny", "passed": slower},
        {"name": "Hipoteza badawcza potwierdzona na tej próbce", "passed": agreement_ok and slower},
    ]


def render_report(results: dict) -> str:
    summary = results["summary"]
    metrics = results["plain_metrics"]
    lines = [
        "Etap IV - testy i porównanie",
        f"Klucz: {results['key_bits']} bitów, cechy TF-IDF: {results['feature_count']}, "
        f"ziarno losowania: {results['seed']}",
        f"Próbka: {summary['sample_count']} SMS "
        f"(ham: {summary['class_counts']['ham']}, spam: {summary['class_counts']['spam']})",
        "",
        "Model jawny - cały zbiór testowy:",
        f"Accuracy: {metrics['accuracy']:.2%}, Precision (spam): {metrics['precision_spam']:.2%}, "
        f"Recall (spam): {metrics['recall_spam']:.2%}, F1 (spam): {metrics['f1_spam']:.4f}",
        f"Macierz pomyłek (wiersze: rzeczywiste, kolumny: przewidziane; ham, spam): "
        f"{metrics['confusion_matrix']}",
        "",
        f"Zgodność decyzji na próbce: {summary['agreement']:.2%}, "
        f"rozbieżności: {summary['disagreements']}",
        f"Accuracy na próbce: jawnie {summary['plain_accuracy']:.2%}, "
        f"szyfrowane {summary['encrypted_accuracy']:.2%}",
        "",
        "Średni czas dla jednego SMS:",
    ]
    for label, column in [
        ("Wariant jawny łącznie", "plain_s"),
        ("TF-IDF w wariancie szyfrowanym", "vectorize_s"),
        ("Szyfrowanie", "encrypt_s"),
        ("Klasyfikacja na serwerze", "classify_s"),
        ("Odszyfrowanie i decyzja klienta", "decrypt_s"),
        ("Wariant szyfrowany łącznie", "encrypted_total_s"),
    ]:
        lines.append(f"  {label}: {summary['mean_times_s'][column] * 1000:.3f} ms")
    lines += [
        "Czasy łączne obejmują TF-IDF i decyzję; pomiar lokalny bez treningu, "
        "generowania kluczy i transmisji sieciowej.",
        "",
        "Weryfikacja:",
    ]
    for check in results["checks"]:
        lines.append(f"  {check['name']}: {'TAK' if check['passed'] else 'NIE'}")
    lines += [
        "",
        "Ograniczenia:",
        "- Mała próbka nie pozwala uogólnić wyników na wszystkie SMS-y.",
        "- Ograniczony słownik TF-IDF może pogarszać skuteczność modelu.",
        "- Zaokrąglenia stałoprzecinkowe mogą zmienić decyzję blisko progu.",
        "- Czasy zależą od sprzętu, liczby cech i długości klucza.",
        "- Paillier obsługuje dodawanie i mnożenie przez jawne wagi; decyzję podejmuje klient.",
        "- Zakładamy uczciwe wykonywanie obliczeń przez serwer; klucz prywatny ma tylko klient.",
    ]
    return "\n".join(lines) + "\n"


def markdown_to_pdf(markdown: str, path: Path) -> Path:
    raise NotImplementedError


def write_outputs(results: dict, config: Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results["comparison"].to_csv(config.output_dir / "porownanie.csv", index=False)
    (config.output_dir / "raport.txt").write_text(render_report(results), encoding="utf-8")
