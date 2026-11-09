from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

from spam_he.config import Config


LABELS = ("ham", "spam")
TIME_LABELS = {
    "plain_s": "Jawny",
    "vectorize_s": "TF-IDF",
    "encrypt_s": "Szyfrowanie",
    "classify_s": "Klasyfikacja",
    "decrypt_s": "Odszyfrowanie",
    "encrypted_total_s": "Szyfrowany",
}


def plot_class_distribution(class_counts: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(class_counts.index, class_counts.values, color=["tab:blue", "tab:red"])
    for x, value in enumerate(class_counts.values):
        ax.text(x, value, str(value), ha="center", va="bottom")
    ax.set_title("Rozkład klas w zbiorze")
    ax.set_ylabel("Liczba SMS")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(matrix, cmap="Blues")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value > max(map(max, matrix)) / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Przewidziana")
    ax.set_ylabel("Rzeczywista")
    ax.set_title("Macierz pomyłek (model jawny)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_times(mean_times: dict[str, float], path: Path) -> None:
    names = [TIME_LABELS[key] for key in TIME_LABELS]
    values = [mean_times[key] * 1000 for key in TIME_LABELS]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(names, values, color="tab:green")
    ax.set_yscale("log")
    ax.set_ylabel("Średni czas na SMS [ms]")
    ax.set_title("Czas wariantu jawnego i szyfrowanego")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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


def markdown_to_pdf(markdown: str, path: Path, images: list[Path] = ()) -> Path:
    font = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(font)))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "DejaVuSans"
    story = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            story.append(Paragraph(line[2:], styles["Heading1"]))
        elif line.strip():
            story.append(Paragraph(line.replace("&", "&amp;").replace("<", "&lt;"), styles["Normal"]))
        else:
            story.append(Spacer(1, 0.3 * cm))
    for image in images:
        story.append(Image(str(image), width=12 * cm, height=7 * cm, kind="proportional"))
    SimpleDocTemplate(str(path)).build(story)
    return path


def write_outputs(results: dict, config: Config) -> None:
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    results["comparison"].to_csv(out / "porownanie.csv", index=False)
    report = render_report(results)
    (out / "raport.txt").write_text(report, encoding="utf-8")
    images = [out / "rozklad_klas.png", out / "macierz_pomylek.png", out / "czasy.png"]
    plot_class_distribution(pd.Series(results["class_counts"]), images[0])
    plot_confusion_matrix(results["plain_metrics"]["confusion_matrix"], images[1])
    plot_times(results["summary"]["mean_times_s"], images[2])
    markdown_to_pdf("# " + report, out / "raport.pdf", images)
